import os
import pandas as pd
from typing import Dict, Union

def load_dataset(file_path: str) -> Union[pd.DataFrame, None]:
    """Loads a CSV, JSON, or XLSX file into a pandas DataFrame."""
    try:
        path_lower = file_path.lower()
        if path_lower.endswith(".csv"):
            return pd.read_csv(file_path)
        if path_lower.endswith(".json"):
            return pd.read_json(file_path)
        if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
            return pd.read_excel(file_path)
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_all_datasets(datasets_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads all CSV, JSON, and XLSX files from a directory."""
    datasets = {}
    if not os.path.exists(datasets_dir):
        return datasets
    for filename in os.listdir(datasets_dir):
        if filename.lower().endswith((".csv", ".json", ".xlsx", ".xls")):
            file_path = os.path.join(datasets_dir, filename)
            df = load_dataset(file_path)
            if df is not None:
                name = os.path.splitext(filename)[0]
                datasets[name] = df
    return datasets
