import difflib
import io
import os
import re
import sys
import traceback
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as MplAxes
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class SchemaResolver:
    """
    Generic schema resolver that adapts to any dataset.
    It resolves dataset aliases, column case differences, derived date parts,
    and best-effort column matching for plotting/groupby operations.
    """
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.dataset_aliases = self._build_dataset_aliases()

    @staticmethod
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    def _build_dataset_aliases(self) -> Dict[str, str]:
        aliases = {}
        for name in self.datasets.keys():
            aliases[self.normalize(name)] = name
            base = str(name).split("__")[-1]
            aliases[self.normalize(base)] = name
        return aliases

    def resolve_dataset(self, key: str) -> str | None:
        if key in self.datasets:
            return key
        norm = self.normalize(key)
        return self.dataset_aliases.get(norm)

    def _find_datetime_column(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            try:
                if np.issubdtype(df[col].dtype, np.datetime64):
                    return col
            except Exception:
                continue
        for col in df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() >= 0.6:
                        df[col] = parsed
                        return col
                except Exception:
                    continue
        return None

    def _ensure_date_part(self, df: pd.DataFrame, key_norm: str) -> str | None:
        date_col = self._find_datetime_column(df)
        if date_col is None:
            return None
        if key_norm in ("month", "month_name"):
            if "Month" not in df.columns:
                df["Month"] = df[date_col].dt.month_name()
            return "Month"
        if key_norm in ("month_num", "month_number"):
            if "Month" not in df.columns:
                df["Month"] = df[date_col].dt.month
            return "Month"
        if key_norm in ("year",):
            if "Year" not in df.columns:
                df["Year"] = df[date_col].dt.year
            return "Year"
        if key_norm in ("day", "day_of_month"):
            if "Day" not in df.columns:
                df["Day"] = df[date_col].dt.day
            return "Day"
        if key_norm in ("week", "week_of_year"):
            if "Week" not in df.columns:
                df["Week"] = df[date_col].dt.isocalendar().week
            return "Week"
        if key_norm in ("quarter",):
            if "Quarter" not in df.columns:
                df["Quarter"] = df[date_col].dt.quarter
            return "Quarter"
        return None

    def prepare_dataframe(self, df: pd.DataFrame) -> None:
        """Pre-compute common date parts when a datetime column exists."""
        if hasattr(df, "attrs") and "original_columns" not in df.attrs:
            try:
                df.attrs["original_columns"] = list(df.columns)
            except Exception:
                pass
        date_col = self._find_datetime_column(df)
        if date_col is None:
            return
        if "Month" not in df.columns:
            df["Month"] = df[date_col].dt.month_name()
        if "Year" not in df.columns:
            df["Year"] = df[date_col].dt.year
        if "Day" not in df.columns:
            df["Day"] = df[date_col].dt.day

    def _is_id_like(self, name: str) -> bool:
        norm = self.normalize(name)
        return any(token in norm for token in ("id", "uuid", "key"))

    def _is_numeric(self, df: pd.DataFrame, col: str) -> bool:
        try:
            return pd.api.types.is_numeric_dtype(df[col])
        except Exception:
            return False

    def resolve_column(self, df: pd.DataFrame, key: str, allow_fallback: bool = True) -> str | None:
        if not isinstance(key, str) or df is None:
            return None
        if key in df.columns:
            return key

        # Case-insensitive exact
        ci_matches = [c for c in df.columns if str(c).lower() == key.lower()]
        if len(ci_matches) == 1:
            return ci_matches[0]

        # Normalized exact
        key_norm = self.normalize(key)
        norm_map = {}
        for col in df.columns:
            norm_map.setdefault(self.normalize(col), col)
        if key_norm in norm_map:
            return norm_map[key_norm]

        # Derived date parts
        derived = self._ensure_date_part(df, key_norm)
        if derived:
            return derived

        # Fuzzy match
        norm_keys = list(norm_map.keys())
        close = difflib.get_close_matches(key_norm, norm_keys, n=1, cutoff=0.82)
        if close:
            return norm_map[close[0]]

        if not allow_fallback:
            return None

        # Best-effort fallback using generic similarity scoring
        best_col = None
        best_score = 0.0
        for col in df.columns:
            norm_col = self.normalize(col)
            score = difflib.SequenceMatcher(None, key_norm, norm_col).ratio()
            if self._is_id_like(col):
                score -= 0.15
            if score > best_score:
                best_score = score
                best_col = col

        if best_col and best_score >= 0.35:
            return best_col

        numeric_cols = [c for c in df.columns if self._is_numeric(df, c) and not self._is_id_like(c)]
        if numeric_cols:
            return numeric_cols[0]

        non_numeric = [c for c in df.columns if not self._is_numeric(df, c)]
        if non_numeric:
            return non_numeric[0]

        return best_col


class DatasetProxy:
    def __init__(self, datasets: Dict[str, pd.DataFrame], resolver: SchemaResolver):
        self._datasets = datasets
        self._resolver = resolver

    def __getitem__(self, key):
        resolved = self._resolver.resolve_dataset(key)
        if resolved is None:
            raise KeyError(key)
        return self._datasets[resolved]

    def get(self, key, default=None):
        resolved = self._resolver.resolve_dataset(key)
        if resolved is None:
            return default
        return self._datasets[resolved]

    def keys(self):
        return self._datasets.keys()

    def values(self):
        return self._datasets.values()

    def items(self):
        return self._datasets.items()

    def __iter__(self):
        return iter(self._datasets)

    def __len__(self):
        return len(self._datasets)


def _code_mentions_plot(code: str) -> bool:
    if not code:
        return False
    return bool(re.search(r"\\b(px|sns|plt|plotly|go)\\.", code, flags=re.IGNORECASE))


def _extract_kwarg_value(code: str, key: str) -> str | None:
    if not code:
        return None
    match = re.search(rf"{key}\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", code)
    if match:
        return match.group(1)
    return None


def _infer_dataset_key(code: str, resolver: SchemaResolver, datasets: Dict[str, pd.DataFrame]) -> str | None:
    if not datasets:
        return None
    if code:
        matches = re.findall(r"datasets\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
        matches += re.findall(r"datasets\.get\(\s*['\"]([^'\"]+)['\"]", code)
        for name in matches:
            resolved = resolver.resolve_dataset(name)
            if resolved:
                return resolved
    return next(iter(datasets.keys()), None)


def _pick_x_column(df: pd.DataFrame, resolver: SchemaResolver) -> str | None:
    date_col = resolver._find_datetime_column(df)
    if date_col:
        return date_col
    for col in df.columns:
        if not resolver._is_numeric(df, col):
            return col
    return df.columns[0] if len(df.columns) > 0 else None


def _pick_y_column(df: pd.DataFrame, resolver: SchemaResolver, exclude: list[str] | None = None) -> str | None:
    excluded = set(exclude or [])
    numeric_cols = [
        c for c in df.columns
        if c not in excluded and resolver._is_numeric(df, c) and not resolver._is_id_like(c)
    ]
    if numeric_cols:
        return numeric_cols[0]
    numeric_any = [
        c for c in df.columns
        if c not in excluded and resolver._is_numeric(df, c)
    ]
    if numeric_any:
        return numeric_any[0]
    return None


def _infer_chart_type(code: str, df: pd.DataFrame, x_col: str | None, y_col: str | None, resolver: SchemaResolver) -> str:
    code_lower = (code or "").lower()
    if "line" in code_lower or "trend" in code_lower or "time" in code_lower:
        return "line"
    if "hist" in code_lower:
        return "hist"
    if "bar" in code_lower:
        return "bar"
    if "scatter" in code_lower or "correl" in code_lower or " vs " in code_lower:
        return "scatter"
    if "box" in code_lower:
        return "box"
    if "pie" in code_lower:
        return "pie"

    if x_col and y_col:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[x_col]) or "date" in str(x_col).lower():
                return "line"
        except Exception:
            pass
        if resolver._is_numeric(df, x_col) and resolver._is_numeric(df, y_col):
            return "scatter"
        if not resolver._is_numeric(df, x_col) and resolver._is_numeric(df, y_col):
            return "bar"
    return "bar"


def _auto_plot_from_code(code: str, datasets: Dict[str, pd.DataFrame], resolver: SchemaResolver):
    dataset_key = _infer_dataset_key(code, resolver, datasets)
    if not dataset_key:
        return None
    df = datasets.get(dataset_key)
    if df is None or not hasattr(df, "columns") or df.empty:
        return None

    x_hint = _extract_kwarg_value(code, "x")
    y_hint = _extract_kwarg_value(code, "y")
    x_col = resolver.resolve_column(df, x_hint) if x_hint else None
    y_col = resolver.resolve_column(df, y_hint) if y_hint else None

    if not x_col:
        x_col = _pick_x_column(df, resolver)
    if not y_col:
        y_col = _pick_y_column(df, resolver, exclude=[x_col] if x_col else None)

    chart = _infer_chart_type(code, df, x_col, y_col, resolver)

    try:
        if y_col is None:
            if x_col is None:
                return None
            counts = df[x_col].astype(str).value_counts().reset_index()
            counts.columns = [x_col, "count"]
            return px.bar(counts, x=x_col, y="count")

        df_plot = df
        if x_col is None:
            df_plot = df.reset_index().rename(columns={"index": "index"})
            x_col = "index"

        if chart == "line":
            return px.line(df_plot, x=x_col, y=y_col)
        if chart == "scatter":
            return px.scatter(df_plot, x=x_col, y=y_col)
        if chart == "box":
            return px.box(df_plot, x=x_col, y=y_col)
        if chart == "pie":
            return px.pie(df_plot, names=x_col, values=y_col)
        if chart == "hist":
            return px.histogram(df_plot, x=x_col)
        return px.bar(df_plot, x=x_col, y=y_col)
    except Exception:
        return None


def auto_plot_from_code(code: str, datasets: Dict[str, pd.DataFrame]):
    """Generic fallback plot builder that works across datasets and schemas."""
    if not _code_mentions_plot(code):
        return None
    resolver = SchemaResolver(datasets)
    for df in datasets.values():
        if hasattr(df, "columns"):
            resolver.prepare_dataframe(df)
    return _auto_plot_from_code(code, datasets, resolver)


def execute_analysis_code(code: str, datasets: Dict[str, pd.DataFrame]) -> Tuple[str, Any, str]:
    """
    Executes Python code safely using a restricted local environment.
    Returns: (text_output, fig, error_message)
    """
    resolver = SchemaResolver(datasets)

    for df in datasets.values():
        if hasattr(df, "columns"):
            resolver.prepare_dataframe(df)

    local_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
        "datasets": DatasetProxy(datasets, resolver),
        "fig": None,
        "result_df": None,
        "result_text": None,
    }

    class PXProxy:
        def __init__(self, px_module, env):
            self._px = px_module
            self._env = env

        def __getattr__(self, name):
            attr = getattr(self._px, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    df = kwargs.get("data_frame")
                    if df is None and args and hasattr(args[0], "columns"):
                        df = args[0]

                    def _map(value):
                        if isinstance(value, str):
                            match = resolver.resolve_column(df, value)
                            return match or value
                        if isinstance(value, (list, tuple)):
                            return [resolver.resolve_column(df, v) or v for v in value]
                        if isinstance(value, dict):
                            return {resolver.resolve_column(df, k) or k: v for k, v in value.items()}
                        return value

                    if df is not None:
                        for key in (
                            "x", "y", "color", "size", "symbol", "facet_row", "facet_col",
                            "hover_name", "hover_data", "text", "animation_frame", "animation_group",
                            "line_group", "pattern_shape"
                        ):
                            if key in kwargs:
                                kwargs[key] = _map(kwargs[key])

                    fig = attr(*args, **kwargs)
                    self._env["fig"] = fig
                    return fig
                return wrapper
            return attr

    local_env["px"] = PXProxy(px, local_env)

    def _blocked_file_access(*args, **kwargs):
        raise ValueError("File access is disabled. Use datasets[\"<name>\"] instead of file paths.")

    _orig_read_csv = pd.read_csv
    _orig_read_excel = pd.read_excel
    _orig_read_json = pd.read_json
    _orig_read_table = getattr(pd, "read_table", None)
    _orig_df_getitem = pd.DataFrame.__getitem__
    _orig_df_setitem = pd.DataFrame.__setitem__
    _orig_df_groupby = pd.DataFrame.groupby

    pd.read_csv = _blocked_file_access
    pd.read_excel = _blocked_file_access
    pd.read_json = _blocked_file_access
    if _orig_read_table is not None:
        pd.read_table = _blocked_file_access
    local_env["open"] = _blocked_file_access

    def _ci_getitem(self, key):
        if isinstance(key, str) and key not in self.columns:
            match = resolver.resolve_column(self, key)
            if match:
                key = match
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            mapped = []
            changed = False
            for k in key:
                if k in self.columns:
                    mapped.append(k)
                    continue
                match = resolver.resolve_column(self, k)
                if match:
                    mapped.append(match)
                    changed = True
                else:
                    mapped.append(k)
            if changed:
                key = mapped
        return _orig_df_getitem(self, key)

    def _ci_setitem(self, key, value):
        if isinstance(key, str) and key not in self.columns:
            match = resolver.resolve_column(self, key)
            if match:
                key = match
        return _orig_df_setitem(self, key, value)

    def _ci_groupby(self, by=None, *args, **kwargs):
        def _map(val):
            if isinstance(val, str):
                return resolver.resolve_column(self, val) or val
            if isinstance(val, list):
                return [_map(v) for v in val]
            return val
        return _orig_df_groupby(self, by=_map(by), *args, **kwargs)

    pd.DataFrame.__getitem__ = _ci_getitem
    pd.DataFrame.__setitem__ = _ci_setitem
    pd.DataFrame.groupby = _ci_groupby

    # Strip interactive show calls and rewrite file reads when possible
    try:
        code = re.sub(r"\.show\(\)\s*", "", code)
        code = re.sub(r"^\s*import\s+plotly\.express\s+as\s+px\s*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*from\s+plotly\s+import\s+express\s+as\s+px\s*$", "", code, flags=re.MULTILINE)

        def _rewrite_read(match):
            var_name = match.group(1)
            path = match.group(2)
            stem = os.path.splitext(os.path.basename(path))[0]
            resolved = resolver.resolve_dataset(stem) or resolver.resolve_dataset(path)
            if resolved:
                return f"{var_name} = datasets[{resolved!r}]"
            return match.group(0)

        code = re.sub(r"(\w+)\s*=\s*pd\.read_csv\(\s*['\"]([^'\"]+)['\"]\s*\)", _rewrite_read, code)
        code = re.sub(r"(\w+)\s*=\s*pd\.read_excel\(\s*['\"]([^'\"]+)['\"]\s*\)", _rewrite_read, code)
        code = re.sub(r"(\w+)\s*=\s*pd\.read_json\(\s*['\"]([^'\"]+)['\"]\s*\)", _rewrite_read, code)
        if _orig_read_table is not None:
            code = re.sub(r"(\w+)\s*=\s*pd\.read_table\(\s*['\"]([^'\"]+)['\"]\s*\)", _rewrite_read, code)
    except Exception:
        pass

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    error_msg = None

    try:
        plt.clf()
        plt.close("all")
        exec(code, {}, local_env)
    except Exception:
        error_msg = f"Error during execution:\n{traceback.format_exc()}"
    finally:
        pd.read_csv = _orig_read_csv
        pd.read_excel = _orig_read_excel
        pd.read_json = _orig_read_json
        if _orig_read_table is not None:
            pd.read_table = _orig_read_table
        pd.DataFrame.__getitem__ = _orig_df_getitem
        pd.DataFrame.__setitem__ = _orig_df_setitem
        pd.DataFrame.groupby = _orig_df_groupby
        sys.stdout = old_stdout

    printed_output = redirected_output.getvalue()
    text_output = ""
    if printed_output:
        text_output += printed_output + "\n"
    if local_env.get("result_text") is not None:
        text_output += str(local_env["result_text"])

    fig = local_env.get("fig")
    if fig is None:
        for val in local_env.values():
            if isinstance(val, go.Figure):
                fig = val
                break
    if fig is None:
        for val in local_env.values():
            try:
                if isinstance(val, MplAxes):
                    fig = val.get_figure()
                    break
            except Exception:
                continue
    if fig is None and len(plt.get_fignums()) > 0:
        fig = plt.gcf()

    result_df = local_env.get("result_df")
    if result_df is not None and isinstance(result_df, pd.DataFrame):
        text_output += "\nData Result:\n" + result_df.head(20).to_markdown()
    elif result_df is not None and isinstance(result_df, pd.Series):
        text_output += "\nData Result:\n" + result_df.head(20).to_markdown()

    if error_msg:
        cols = []
        dataset_name = _infer_dataset_key(code, resolver, datasets) if datasets else None
        if dataset_name and dataset_name in datasets:
            df = datasets[dataset_name]
            cols = list(df.attrs.get("original_columns") or df.columns)
            prefix = f"Chart generation failed for dataset `{dataset_name}`. Available columns: "
        else:
            for df in datasets.values():
                if hasattr(df, "columns"):
                    cols = list(df.attrs.get("original_columns") or df.columns)
                    break
            prefix = "Chart generation failed. Available columns: "
        error_msg = (
            prefix +
            ", ".join(map(str, cols[:40])) +
            (" ..." if len(cols) > 40 else "")
        )

    return text_output.strip(), fig, error_msg
