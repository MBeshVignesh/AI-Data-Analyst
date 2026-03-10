import pandas as pd
from typing import Dict, Any

def extract_metadata(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Extracts metadata from a pandas DataFrame to be used for LLM context."""
    
    # Get basic info
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Get a sample of the data (up to 3 rows)
    sample_df = df.head(min(3, num_rows))
    sample_data = sample_df.to_dict(orient='records')
    
    # Get basic summary statistics for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    stats = {}
    if not numeric_df.empty:
        stats_df = numeric_df.describe()
        # Keep only mean, min, max to save space
        for col in stats_df.columns:
            stats[col] = {
                "mean": stats_df.loc["mean", col] if "mean" in stats_df.index and not pd.isna(stats_df.loc["mean", col]) else None,
                "min": stats_df.loc["min", col] if "min" in stats_df.index and not pd.isna(stats_df.loc["min", col]) else None,
                "max": stats_df.loc["max", col] if "max" in stats_df.index and not pd.isna(stats_df.loc["max", col]) else None,
            }
            
    summary = f"Dataset Name: {dataset_name}\n"
    summary += f"Shape: {num_rows} rows, {num_cols} columns\n"
    summary += f"Columns: {', '.join(columns)}\n"
    summary += f"Data Types: {', '.join([f'{k} ({v})' for k, v in dtypes.items()])}\n"
    summary += f"Sample Data: {sample_data}\n"

    # Richer searchable document for vector DB: natural-language style so retrieval matches queries better
    sample_text = " ".join(str(v) for r in sample_data[:3] for v in (list(r.values())[:10] if isinstance(r, dict) else []))[:500]
    searchable_parts = [
        f"Dataset {dataset_name} with {num_rows} rows and columns: " + ", ".join(columns),
        "Column names: " + " ".join(columns),
        "Sample values: " + sample_text,
    ]
    if stats:
        searchable_parts.append("Numeric columns: " + " ".join(stats.keys()))
    searchable_doc = ". ".join(searchable_parts) + ".\n" + summary

    return {
        "name": dataset_name,
        "summary": summary,
        "searchable_doc": searchable_doc,
        "columns": columns,
        "dtypes": dtypes,
        "stats": stats,
        "sample": sample_data
    }
