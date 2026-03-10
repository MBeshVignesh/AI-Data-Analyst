import re
import traceback
from typing import Tuple, Any, Dict, List, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from ..rag.retriever import ContextRetriever
from ..utils.safe_code_executor import execute_analysis_code


class DataAgent:
    def __init__(self, retriever: ContextRetriever, datasets: Dict[str, Any], model_name: str = "llama3.2"):
        self.retriever = retriever
        self.datasets = datasets
        self.llm = OllamaLLM(model=model_name, temperature=0.3, max_tokens=6000)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a professional **Data Analyst**. Your job is to answer the user with **accurate, detailed insights** and to run code when needed. Always use the **exact dataset and column names** from the context below.

---

## Dataset Context (use these names exactly)
{context}

If the context includes **"User uploaded the following file(s)"** (e.g. PDF text or an image), answer the question using that content: for PDFs use the extracted text; for images respond based on filename/context (image analysis may be limited).

---

## Behavior by Request Type

### 1. SQL-only requests
If the user asks **only** for SQL (e.g. "write a SQL query", "give me the SQL", "SQL only", "just the query", "convert this to SQL"):
- Respond with **only** a one-line description and a single ```sql code block. Do NOT generate any Python.
- Use only table/column names that exist in the dataset context above.

### 2. Plots / charts / visualizations
If the user asks for a **plot, chart, graph, or visualization** (e.g. "plot", "chart", "visualize", "graph", "bar chart", "line plot"):
- Generate **Python** that loads the data, creates the plot, and assigns the figure: `fig = plt.gcf()` before the block ends.
- Use: `import pandas as pd`, `import matplotlib.pyplot as plt`, `datasets = datasets`, `df = datasets["EXACT_DATASET_NAME"]`.
- Write 1–2 sentences describing what the plot shows and one concrete insight (e.g. a trend or outlier).
- Do NOT repeat the code in your narrative.

### 3. "What is the output of this SQL?" / "Run this query" / user pastes SQL and wants results
- The user wants to **see the result table**, not just the SQL again. Generate **Python** that replicates the SQL logic using pandas (e.g. groupby, agg, merge) on the dataset, then **print the result** with `print(result_df)` or `print(df.to_string())` so the actual data is shown.
- Map SQL to pandas: GROUP BY → df.groupby(), AVG/SUM/COUNT → .agg(), ORDER BY → .sort_values(). Use the exact column names from the context.
- One short line (e.g. "Result of the query:") then the code. The printed output will be the answer.

### 4. Show records / sample data / view table / data table
- Generate **Python** that **prints** the data so the user sees the table: e.g. `print(df.head(10))` or `print(df.to_string())`. The system captures print output and shows it.
- Say briefly what you are showing (e.g. "First 10 rows of the sales table:") then let the code output speak.

### 5. General analysis (stats, comparisons, "how many", "what is", "summarize")
- Generate **Python** to compute the answer, then give **detailed insights**:
  - 2–4 sentences: what you did, the main numbers (counts, sums, percentages), and a clear takeaway.
  - Use specific numbers from the data; do not give vague answers.
- Use only column names from the schema. No guessing values.

---

## Strict rules
- **EXACT COLUMNS**: Use only column names and dataset names from the context. Copy them character-for-character.
- **NO HALLUCINATION**: Do not invent data. If you need a number, compute it in code.
- **ACTION OVER INSTRUCTION**: Provide runnable code and results, not "you can run this" instructions.
- **ONE FIG PER PLOT REQUEST**: For plots, exactly one Python block ending with `fig = plt.gcf()`.
- **OUTPUT CODE ONCE**: In your reply, include the code block **only once**. Do not repeat the same code after the narrative or insights.

---

## Response structure
1. Short intro (what you are doing).
2. code block: ```python or ```sql as above. Do not duplicate the code.
3. **Insights**: For analysis and plots, add 2–4 sentences with specific numbers. Do not repeat or paste the code again.

---

## User question
{question}
"""
        )

        # Used for self-healing retry when Python code execution fails
        self.retry_prompt_template = PromptTemplate(
            input_variables=["context", "original_question", "failed_code", "error"],
            template="""
You are a Python data analyst. Your previous code failed.
Fix the Python code so it works correctly.

## Dataset Schema (EXACT column names — use them character-for-character)

{context}

## Original Question
{original_question}

## Failed Code
```python
{failed_code}
```

## Error
{error}

Write ONLY the corrected ```python code block. No explanation.
Use only the column names shown above.
"""
        )

    def extract_all_code_blocks(self, response: str) -> List[Tuple[str, str]]:
        """
        Extracts ALL code blocks from an LLM response.
        Returns list of (language, code) tuples in order of appearance.
        If no language is specified, attempts to detect (SQL vs Python).
        """
        blocks = []
        # Support optional language label
        pattern = re.compile(r"```(\w*)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(response):
            lang = match.group(1).lower()
            code = match.group(2).strip()
            
            if not lang:
                # Fallback detection
                lower_code = code.lower()
                if "select" in lower_code and "from" in lower_code:
                    lang = "sql"
                else:
                    lang = "python"
            
            if code:
                blocks.append((lang, code))
        return blocks

    def strip_code_blocks(self, response: str) -> str:
        """Removes all code blocks from a string, returning clean narrative text."""
        # Match any code block with optional language prefix
        clean = re.sub(r"```\w*.*?```", "", response, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()

    def run_stream(
        self,
        question: str,
        session_id: str = "default",
        prefetched_context: Optional[str] = None,
        preferred_dataset_names: Optional[List[str]] = None,
        upload_context: Optional[str] = None,
    ):
        """
        Streaming version of the pipeline. Yields (text_token, fig, code, error, run_output).
        preferred_dataset_names: when user just uploaded tabular data, prefer it for this question.
        upload_context: when user uploaded PDF/image, text or placeholder to answer with respect to that file.
        """
        try:
            # 1. Build context (dataset + memory + optional upload content)
            if prefetched_context:
                combined_context = prefetched_context
                print(f"[DataAgent] Using pre-fetched context.")
            else:
                dataset_context = self.retriever.get_relevant_datasets(
                    question, top_k=5, preferred_dataset_names=preferred_dataset_names
                )
                memory_context = self.retriever.vector_db.get_memory(session_id, question)
                combined_context = dataset_context + ("\n" + memory_context if memory_context else "")
            if upload_context:
                combined_context = combined_context + "\n\n" + upload_context

            # 2. Stream LLM
            prompt = self.prompt_template.format(context=combined_context, question=question)
            print(f"[DataAgent] Streaming LLM (session={session_id[:8]}…)")
            full_response = ""
            for chunk in self.llm.stream(prompt):
                full_response += chunk
                yield chunk, None, "", "", ""

            # 3. Extract and execute code (same as run())
            code_blocks = self.extract_all_code_blocks(full_response)
            narrative = self.strip_code_blocks(full_response)
            sql_blocks = []
            python_blocks = []
            for lang, code in code_blocks:
                if lang == "sql":
                    sql_blocks.append(code)
                elif lang == "python":
                    python_blocks.append(code)

            fig = None
            py_text_output = ""
            internal_error = ""
            display_code_parts = []

            if python_blocks:
                combined_python = "\n\n".join(python_blocks)
                display_code_parts.append(f"python\n{combined_python}")
                py_text_output, fig, internal_error = execute_analysis_code(combined_python, self.datasets)

                if internal_error:
                    print(f"[DataAgent] Attempting self-heal...")
                    retry_prompt = self.retry_prompt_template.format(
                        context=combined_context,
                        original_question=question,
                        failed_code=combined_python,
                        error=internal_error
                    )
                    fixed_response = self.llm.invoke(retry_prompt)
                    extracted = self.extract_all_code_blocks(fixed_response)
                    if extracted:
                        _, fixed_code = extracted[0]
                        display_code_parts[0] = f"python\n{fixed_code}"
                        py_text_output, fig, _ = execute_analysis_code(fixed_code, self.datasets)

            for sql in sql_blocks:
                display_code_parts.append(f"sql\n{sql}")

            display_code = "\n\n---\n\n".join(display_code_parts) if display_code_parts else ""
            yield "", fig, display_code, "", py_text_output or ""

        except Exception as e:
            yield f"I encountered an issue analyzing the data: {str(e)}", None, "", "", ""

    def run(self, question: str, session_id: str = "default") -> Tuple[str, Any, str, str]:
        """
        Runs the agent pipeline:
        1. Build context (dataset schemas + session memory)
        2. Prompt LLM
        3. Extract ALL code blocks (SQL + Python)
        4. Execute Python blocks (with self-healing retry on error)
        5. Display SQL as formatted text
        6. Save turn to session memory

        Returns:
            Tuple of (text_output, fig, code_for_display, error_message)
        """
        try:
            # 1. Build rich context
            dataset_context = self.retriever.get_relevant_datasets(question, top_k=5)
            memory_context = self.retriever.vector_db.get_memory(session_id, question)
            combined_context = dataset_context + ("\n" + memory_context if memory_context else "")

            # 2. Prompt LLM
            prompt = self.prompt_template.format(context=combined_context, question=question)
            print(f"[DataAgent] Prompting LLM (session={session_id[:8]}…)")
            llm_response = self.llm.invoke(prompt)

            # 3. Extract all code blocks
            code_blocks = self.extract_all_code_blocks(llm_response)
            narrative = self.strip_code_blocks(llm_response)

            # 4. Process blocks
            sql_blocks: List[str] = []
            python_blocks: List[str] = []
            for lang, code in code_blocks:
                if lang == "sql":
                    sql_blocks.append(code)
                elif lang == "python":
                    python_blocks.append(code)

            # 5. Execute Python blocks (merge into one execution for shared scope)
            fig = None
            py_text_output = ""
            internal_error = "" # Hide this from user if self-heal works
            display_code_parts = []

            if python_blocks:
                combined_python = "\n\n".join(python_blocks)
                display_code_parts.append(f"python\n{combined_python}")
                py_text_output, fig, internal_error = execute_analysis_code(combined_python, self.datasets)

                # Self-healing retry on error
                if internal_error:
                    print(f"[DataAgent] Execution failed, attempting self-heal (internal error suppressed)...")
                    retry_prompt = self.retry_prompt_template.format(
                        context=dataset_context,
                        original_question=question,
                        failed_code=combined_python,
                        error=internal_error
                    )
                    fixed_response = self.llm.invoke(retry_prompt)
                    # Extract from result
                    extracted = self.extract_all_code_blocks(fixed_response)
                    if extracted:
                        _, fixed_code = extracted[0]
                        display_code_parts[0] = f"python\n{fixed_code}"
                        py_text_output, fig, final_error = execute_analysis_code(fixed_code, self.datasets)
                        
                        if final_error:
                            print(f"[DataAgent] Self-heal failed.")
                            # Only return error if the FINAL attempt failed
                            return narrative, fig, "\n\n---\n\n".join(display_code_parts), "Analysis failed after internal correction. Please try rephrasing."
                        else:
                            print(f"[DataAgent] Self-heal succeeded.")
                            # Success! internal_error is ignored
                            internal_error = "" 
                    else:
                        return narrative, fig, "\n\n---\n\n".join(display_code_parts), "Analysis failed. Please try rephrasing."

            for sql in sql_blocks:
                display_code_parts.append(f"sql\n{sql}")

            # 6. Build final outputs
            final_text = narrative
            if py_text_output:
                final_text = (final_text + "\n\n" + py_text_output).strip()

            display_code = "\n\n---\n\n".join(display_code_parts) if display_code_parts else ""

            # 7. Save to memory
            save_text = final_text or llm_response
            if not internal_error:
                self.retriever.vector_db.add_memory(session_id, question, save_text)

            return final_text, fig, display_code, "" # Clean return

        except Exception as e:
            return f"I encountered an issue analyzing the data. Please try again.", None, "", ""
