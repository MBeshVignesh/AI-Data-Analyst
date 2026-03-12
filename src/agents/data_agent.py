import re
import time
import traceback
from typing import Tuple, Any, Dict, List, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from ..rag.retriever import ContextRetriever
from ..utils.safe_code_executor import execute_analysis_code, auto_plot_from_code


class DataAgent:
    def __init__(self, retriever: ContextRetriever, datasets: Dict[str, Any], model_name: str = "llama3.2"):
        self.retriever = retriever
        self.datasets = datasets
        self.llm = OllamaLLM(model=model_name, temperature=0.1, max_tokens=6000)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a highly capable, flexible, and conversational **Data Analyst**. Your top priority is to provide **Business Insights** and communicate naturally with the user. You are NOT just a rigid code generator.

IMPORTANT ABILITIES:
1. Provide rich business insights in warm, natural language BEFORE writing any code. Explain patterns, trends, and significance.
2. Show sample data automatically when asked or when it adds clarity. Briefly interpret what the data represents.
3. Generate plots that render purely in the conversation window when asked (NO EXTERNAL WINDOWS).
4. Generate SQL code if asked. You are fluent in SQL and can provide raw queries if requested.
5. Use Python to compute answers or plot data, but always surround it with rich conversational context. NEVER output just code!

---

## Dataset Context (use these names exactly)
{context}

If the context includes **"User uploaded the following file(s)"** (e.g. PDF text or an image), answer the question using that content: for PDFs use the extracted text; for images respond based on filename/context (image analysis may be limited).

---

## Behavior Guidelines by Request Type

### 1. Business Insights / Open-Ended Analysis
- If the user asks a general question (e.g., "What can you tell me?", "Analyze this data", "Give me some insights"):
  - Give a highly conversational, detailed breakdown.
  - Generate **Python** to compute the answer. 
  - Provide a thorough, well-formatted response with bullet points detailing multiple insights (e.g., trends, outliers, high/low values).
  - Do NOT just drop a snippet of code and leave. Be a true analyst and interpret the numbers!

### 2. Show records / sample data / view table
- If the user explicitly asks to see data, or if showing data helps answer their question:
  - Generate **Python** that **prints** the data: e.g. `print(df.head(10).to_markdown())` or `print(df.tail(5))`.
  - Speak conversationally: "Here are the first few rows of the data for you to review." Then the output of the print will show up.

### 3. Plots / charts / visualizations
- If asked for a plot (e.g., "visualize", "plot", "chart", "graph"):
  - Generate **Python** that assigns the figure to `fig`.
  - Prefer Plotly (`import plotly.express as px`) for interactive graphs, or Matplotlib/Seaborn.
  - Access the data dynamically using: `df = datasets["EXACT_DATASET_NAME"]`.
  - Describe the plot conversationally and point out at least one interesting insight from the chart. 

### 4. SQL-only requests
- If the user asks **only** for SQL (e.g. "write a SQL query", "give me the SQL"):
  - Respond with a conversational intro, then provide the SQL code block. Make sure you use the exact table/column names from the context. Do NOT generate Python code unless they also ask to execute it.

### 5. Executing SQL ("Run this query")
- Generate **Python** using `pandas` to replicate the SQL logic. 
  - E.g., `GROUP BY` -> `df.groupby()`, `ORDER BY` -> `.sort_values()`.
  - Print the result using `print(result_df)`. Say something like, "Here is the result of your query:"

---

## Strict Rules
- **TEXT FIRST**: Start with a brief 1-sentence natural-language lead-in. If you include code, place code blocks immediately after that sentence, then continue with the narrative.
- **BUSINESS INSIGHTS**: Do not just spit out numbers. Be a real analyst interpreting the data organically.
- **EXACT COLUMNS**: Copy column and dataset names character-for-character from the Context.
- **CASE SENSITIVE**: Column names are case-sensitive. Preserve the exact capitalization from the Context.
- **FLEXIBLE OUTPUT**: Provide SQL for the user to copy, based precisely on their request.
- **USE CODE AS NEEDED**: Write Python when you need to calculate an answer, plot data, or show data.
- **NO FILE I/O**: Use `datasets["<name>"]` exclusively. No `pd.read_csv()`!
- **ONE FIG PER REQUEST**: If plotting, assign exactly one figure to the `fig` variable.
- **CONVERSATIONAL TONE**: Provide friendly, non-rigid, informative analysis.

---

## User Question
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
- Use only the column names shown above.
- Access data using `df = datasets["EXACT_DATASET_NAME"]`. Do NOT use `pd.read_csv()`.
- Never read local files. Use `datasets[...]` only.
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
        allowed_dataset_names: Optional[List[str]] = None,
        allowed_sources: Optional[List[str]] = None,
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
                dataset_context = self.retriever.get_relevant_context(
                    question,
                    top_k_datasets=8,
                    top_k_files=8,
                    preferred_dataset_names=preferred_dataset_names,
                    allowed_dataset_names=allowed_dataset_names,
                    allowed_sources=allowed_sources,
                )
                memory_context = self.retriever.vector_db.get_memory(session_id, question)
                combined_context = dataset_context + ("\n" + memory_context if memory_context else "")
            if upload_context:
                combined_context = combined_context + "\n\n" + upload_context

            # 2. Stream LLM
            prompt = self.prompt_template.format(context=combined_context, question=question)
            print(f"[DataAgent] Streaming LLM (session={session_id[:8]}…)")
            full_response = ""
            last_emitted_code = ""
            for chunk in self.llm.stream(prompt):
                full_response += chunk
                yield chunk, None, "", "", ""

                if "```" in chunk:
                    partial_blocks = self.extract_all_code_blocks(full_response)
                    if partial_blocks:
                        display_code_parts = [f"{lang}\n{code}" for lang, code in partial_blocks]
                        display_code = "\n\n---\n\n".join(display_code_parts)
                        if display_code and display_code != last_emitted_code:
                            last_emitted_code = display_code
                            yield "", None, display_code, "", ""

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
            combined_python = None

            if python_blocks:
                combined_python = python_blocks[0]
                display_code_parts.append(f"python\n{combined_python}")

            for sql in sql_blocks:
                display_code_parts.append(f"sql\n{sql}")

            display_code = "\n\n---\n\n".join(display_code_parts) if display_code_parts else ""
            if display_code and display_code != last_emitted_code:
                last_emitted_code = display_code
                yield "", None, display_code, "", ""

            if combined_python:
                py_text_output, fig, internal_error = execute_analysis_code(combined_python, self.datasets)
                plot_requested = bool(re.search(r"\\b(px|sns|plt)\\.|plotly", combined_python, flags=re.IGNORECASE))

                if plot_requested and (fig is None or internal_error):
                    # Retry once after a short pause for more robust chart generation
                    time.sleep(0.4)
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
                        display_code = "\n\n---\n\n".join(display_code_parts)
                        if display_code and display_code != last_emitted_code:
                            last_emitted_code = display_code
                            yield "", None, display_code, "", ""
                        py_text_output, fig, internal_error = execute_analysis_code(fixed_code, self.datasets)

                if plot_requested and fig is None:
                    fallback_fig = auto_plot_from_code(combined_python, self.datasets)
                    if fallback_fig is not None:
                        fig = fallback_fig
                        internal_error = ""

            yield "", fig, "", internal_error or "", py_text_output or ""

        except Exception as e:
            yield f"I encountered an issue analyzing the data: {str(e)}", None, "", "", ""

    def run(
        self,
        question: str,
        session_id: str = "default",
        allowed_dataset_names: Optional[List[str]] = None,
        allowed_sources: Optional[List[str]] = None,
    ) -> Tuple[str, Any, str, str]:
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
            dataset_context = self.retriever.get_relevant_context(
                question,
                top_k_datasets=8,
                top_k_files=8,
                allowed_dataset_names=allowed_dataset_names,
                allowed_sources=allowed_sources,
            )
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
                combined_python = python_blocks[0]
                display_code_parts.append(f"python\n{combined_python}")
                py_text_output, fig, internal_error = execute_analysis_code(combined_python, self.datasets)
                plot_requested = bool(re.search(r"\\b(px|sns|plt)\\.|plotly", combined_python, flags=re.IGNORECASE))

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

                if plot_requested and fig is None:
                    fallback_fig = auto_plot_from_code(combined_python, self.datasets)
                    if fallback_fig is not None:
                        fig = fallback_fig
                        internal_error = ""

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
