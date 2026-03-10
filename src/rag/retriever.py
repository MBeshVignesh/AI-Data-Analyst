from .vectordb import VectorDBManager
from typing import List, Dict, Any, Optional
import pandas as pd


class ContextRetriever:
    def __init__(self, vector_db: VectorDBManager, datasets: Dict[str, pd.DataFrame] = None):
        self.vector_db = vector_db
        self.datasets = datasets or {}

    def _format_dataset_block(self, dataset_name: str, rank: int) -> str:
        """Format a single dataset's schema and sample for LLM context."""
        df = self.datasets.get(dataset_name)
        if df is None:
            return ""
        exact_cols = list(df.columns)
        dtypes_str = ", ".join(f"{c} ({df[c].dtype})" for c in exact_cols)
        sample = df.head(2).to_dict(orient="records")
        schema_block = (
            f"EXACT COLUMNS (use these names exactly in code):\n"
            f"  {exact_cols}\n"
            f"Column types: {dtypes_str}\n"
            f"Sample rows: {sample}\n"
        )
        return (
            f"--- Dataset {rank}: `{dataset_name}` (use this for the user's question) ---\n"
            f"Access via: datasets[\"{dataset_name}\"]\n"
            f"{schema_block}"
        )

    def get_relevant_datasets(
        self,
        query: str,
        top_k: int = 5,
        preferred_dataset_names: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieves the most relevant datasets for the given query and
        formats a rich context string. If preferred_dataset_names is given (e.g. from
        just-uploaded files), those datasets are included first so the question
        is answered with respect to that data.
        """
        context_parts = []
        seen = set()

        # Prefer recently uploaded / user-specified datasets so the question is about that data
        if preferred_dataset_names:
            for i, name in enumerate(preferred_dataset_names):
                if name in self.datasets and name not in seen:
                    seen.add(name)
                    block = self._format_dataset_block(name, len(context_parts) + 1)
                    if block:
                        context_parts.append(block)

        # Then add semantic search results (skip already included)
        results = self.vector_db.search(query, n_results=top_k)
        if results:
            for res in results:
                dataset_name = res["metadata"].get("name", "")
                if not dataset_name or dataset_name in seen:
                    continue
                seen.add(dataset_name)
                block = self._format_dataset_block(dataset_name, len(context_parts) + 1)
                if block:
                    context_parts.append(block)

        if not context_parts:
            return "No relevant dataset context found."
        return "\n\n".join(context_parts)
