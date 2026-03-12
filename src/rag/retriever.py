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

    def _format_file_block(self, doc: str, metadata: Dict[str, Any], rank: int) -> str:
        name = metadata.get("name") or metadata.get("path") or metadata.get("file_id") or "file"
        source = metadata.get("source", "unknown")
        ftype = metadata.get("type", "text")
        return (
            f"--- File {rank}: {name} (source={source}, type={ftype}) ---\n"
            f"{doc}\n"
        )

    def get_relevant_datasets(
        self,
        query: str,
        top_k: int = 5,
        preferred_dataset_names: Optional[List[str]] = None,
        allowed_dataset_names: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieves the most relevant datasets for the given query and
        formats a rich context string. If preferred_dataset_names is given (e.g. from
        just-uploaded files), those datasets are included first so the question
        is answered with respect to that data.
        """
        context_parts = []
        seen = set()
        allowed = set(allowed_dataset_names) if allowed_dataset_names else None

        # Prefer recently uploaded / user-specified datasets so the question is about that data
        if preferred_dataset_names:
            for i, name in enumerate(preferred_dataset_names):
                if allowed is not None and name not in allowed:
                    continue
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
                if allowed is not None and dataset_name not in allowed:
                    continue
                if not dataset_name or dataset_name in seen:
                    continue
                seen.add(dataset_name)
                block = self._format_dataset_block(dataset_name, len(context_parts) + 1)
                if block:
                    context_parts.append(block)

        if not context_parts:
            return "No relevant dataset context found."
        return "\n\n".join(context_parts)

    def get_relevant_context(
        self,
        query: str,
        top_k_datasets: int = 5,
        top_k_files: int = 6,
        preferred_dataset_names: Optional[List[str]] = None,
        allowed_dataset_names: Optional[List[str]] = None,
        allowed_sources: Optional[List[str]] = None,
    ) -> str:
        dataset_context = self.get_relevant_datasets(
            query,
            top_k=top_k_datasets,
            preferred_dataset_names=preferred_dataset_names,
            allowed_dataset_names=allowed_dataset_names,
        )
        file_results = self.vector_db.search_files(
            query,
            n_results=top_k_files,
            allowed_sources=allowed_sources,
        )
        file_blocks = []
        if file_results:
            for i, res in enumerate(file_results, start=1):
                doc = res.get("document") or ""
                meta = res.get("metadata") or {}
                block = self._format_file_block(doc, meta, i)
                if block:
                    file_blocks.append(block)
        if file_blocks:
            return dataset_context + "\n\n" + "\n".join(file_blocks)
        return dataset_context
