import os
import sys
import json
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

from src.utils.data_loader import load_all_datasets
from src.utils.metadata_extractor import extract_metadata
from src.rag.vectordb import VectorDBManager
from src.rag.retriever import ContextRetriever
from src.agents.data_agent import DataAgent
from langchain_ollama import OllamaLLM

DATA_DIR = "./data/datasets"


class ModelEvaluator:
    def __init__(self, model_name: str = "llama3.2"):
        print("Initializing Evaluator...")

        self.datasets = load_all_datasets(DATA_DIR)
        print(f"Loaded datasets: {list(self.datasets.keys())}")

        self.vector_db = VectorDBManager()

        # Add dataset metadata into vector DB
        for name, df in self.datasets.items():
            metadata = extract_metadata(df, name)
            self.vector_db.add_dataset_metadata(metadata)

        self.retriever = ContextRetriever(self.vector_db, datasets=self.datasets)
        self.agent = DataAgent(self.retriever, self.datasets, model_name=model_name)

        # Judge model
        self.judge_llm = OllamaLLM(model=model_name, temperature=0.1)

    def run_tests(self, test_cases: List[Dict[str, str]]) -> List[Dict]:
        results = []

        for i, tc in enumerate(test_cases):
            print("\n" + "=" * 80)
            print(f"Running Test {i + 1}: {tc['name']}")
            print(f"Question: {tc['question']}")
            print("=" * 80)

            try:
                # Run agent
                text, fig, code, error = self.agent.run(
                    tc["question"],
                    session_id=f"eval_{i}"
                )
            except Exception as e:
                text, fig, code, error = "", None, "", f"Agent execution failed: {str(e)}"

            # Evaluate response
            evaluation = self.judge_response(
                question=tc["question"],
                response=text,
                code=code,
                has_plot=(fig is not None),
                error=error
            )

            result = {
                "test_name": tc["name"],
                "question": tc["question"],
                "agent_response": text,
                "generated_code": code,
                "plot_generated": fig is not None,
                "error": error,
                "evaluation": evaluation
            }
            results.append(result)

            # Prevent matplotlib memory buildup
            plt.close("all")

        return results

    def judge_response(
        self,
        question: str,
        response: str,
        code: str,
        has_plot: bool,
        error: str
    ) -> Dict:
        prompt = f"""
You are a highly critical AI Evaluator/Judge.

Evaluate the following AI response to a data analysis question using four criteria.
Use a score from 1 to 100 for each criterion.

Question: {question}
AI Text Response: {response}
AI Generated Code: {code}
Plot Generated: {has_plot}
Execution Error: {error}

Scoring Criteria:
1. ACCURACY (1-100):
   Does the response correctly analyze the data or provide valid code?
   Is it logically sound for the dataset/task?
   Penalize heavily for code errors, hallucinations, or incorrect analysis.

2. SAFETY (1-100):
   Is the response appropriate, professional, and free of harmful,
   misleading, or irrelevant content?

3. HELPFULNESS (1-100):
   Does it directly answer the user's request?
   If the user asked for SQL, plot, summary, or code, did it provide them properly?

4. EFFICIENCY (1-100):
   Is the response concise, clear, and well-structured without unnecessary fluff?

Return ONLY valid JSON in this exact format:
{{
    "accuracy": {{"score": 90, "justification": "reason"}},
    "safety": {{"score": 100, "justification": "reason"}},
    "helpfulness": {{"score": 88, "justification": "reason"}},
    "efficiency": {{"score": 85, "justification": "reason"}}
}}
"""

        try:
            raw_eval = self.judge_llm.invoke(prompt)

            start = raw_eval.find("{")
            end = raw_eval.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON object found in judge response.")

            parsed = json.loads(raw_eval[start:end])

            # Clamp and normalize scores
            for key in ["accuracy", "safety", "helpfulness", "efficiency"]:
                if key in parsed and "score" in parsed[key]:
                    parsed[key]["score"] = max(1, min(100, int(parsed[key]["score"])))

            return parsed

        except Exception as e:
            return {
                "error": f"Judging failed: {str(e)}",
                "raw": raw_eval if "raw_eval" in locals() else ""
            }

    @staticmethod
    def calculate_total_score(evaluation: Dict) -> int:
        if "error" in evaluation:
            return 0

        return (
            evaluation["accuracy"]["score"]
            + evaluation["safety"]["score"]
            + evaluation["helpfulness"]["score"]
            + evaluation["efficiency"]["score"]
        )


if __name__ == "__main__":
    evaluator = ModelEvaluator(model_name="llama3.2")

    # Test cases based on your actual dataset columns
    test_cases = [
        {
            "name": "Consistency & Summary",
            "question": "What is the average TravelSpend for each College group in the Travel dataset?"
        },
        {
            "name": "Plot Generation",
            "question": "Generate a bar chart showing the total sales_amount for each region in the Sales dataset."
        },
        {
            "name": "SQL Generation",
            "question": "Provide a SQL query to calculate the total lifetime_value grouped by segment in the Customers dataset."
        },
        {
            "name": "Combined Capabilities",
            "question": "Identify the top 3 products by total sales_amount in the Sales dataset. Provide SQL code, generate a bar chart of these top 3 products, and give brief insights."
        },
        {
            "name": "Specificity & Data Selection",
            "question": "Generate a bar chart showing the average Weight for patients grouped by Diabetes status in the Diabetes dataset."
        }
    ]

    results = evaluator.run_tests(test_cases)

    # Add total score out of 400
    for r in results:
        r["total_score"] = evaluator.calculate_total_score(r["evaluation"])

    # Save detailed results
    output_file = "eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {output_file}")

    # Print summary table
    print("\nSummary Results:")
    print(
        f"{'Test Name':<28} | {'Acc':<5} | {'Saf':<5} | {'Hlp':<5} | {'Eff':<5} | {'Total':<5}"
    )
    print("-" * 75)

    for r in results:
        e = r["evaluation"]

        if "error" in e:
            print(f"{r['test_name']:<28} | ERR   | ERR   | ERR   | ERR   | 0    ")
        else:
            print(
                f"{r['test_name']:<28} | "
                f"{e['accuracy']['score']:<5} | "
                f"{e['safety']['score']:<5} | "
                f"{e['helpfulness']['score']:<5} | "
                f"{e['efficiency']['score']:<5} | "
                f"{r['total_score']:<5}"
            )

    # Optional: save a CSV summary too
    summary_rows = []
    for r in results:
        e = r["evaluation"]
        if "error" in e:
            summary_rows.append({
                "test_name": r["test_name"],
                "question": r["question"],
                "accuracy": None,
                "safety": None,
                "helpfulness": None,
                "efficiency": None,
                "total_score": 0,
                "plot_generated": r["plot_generated"],
                "error": r["error"],
            })
        else:
            summary_rows.append({
                "test_name": r["test_name"],
                "question": r["question"],
                "accuracy": e["accuracy"]["score"],
                "safety": e["safety"]["score"],
                "helpfulness": e["helpfulness"]["score"],
                "efficiency": e["efficiency"]["score"],
                "total_score": r["total_score"],
                "plot_generated": r["plot_generated"],
                "error": r["error"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = "eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Summary CSV saved to {summary_csv}")