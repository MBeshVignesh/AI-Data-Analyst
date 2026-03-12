from typing import List
import pandas as pd


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def tabular_to_chunks(
    df: pd.DataFrame,
    dataset_name: str,
    max_rows: int = 200,
    rows_per_chunk: int = 20,
) -> List[str]:
    if df is None or df.empty:
        return []
    rows = min(max_rows, len(df))
    if rows <= 0:
        return []
    header = ", ".join([str(c) for c in df.columns])
    chunks = []
    for start in range(0, rows, rows_per_chunk):
        end = min(rows, start + rows_per_chunk)
        sample = df.iloc[start:end]
        lines = [header]
        for _, row in sample.iterrows():
            lines.append(", ".join([str(v) for v in row.values.tolist()]))
        chunk = (
            f"Dataset {dataset_name} rows {start + 1}-{end} (CSV-like):\n" + "\n".join(lines)
        )
        chunks.append(chunk)
    return chunks
