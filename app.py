import os
import re
import time
import uuid
import json
import threading
import base64
import io
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv

from src.utils.data_loader import load_dataset
from src.utils.file_parsers import get_upload_context_for_non_tabular, extract_pdf_text
from src.utils.chunking import chunk_text
from src.utils.metadata_extractor import extract_metadata
from src.rag.vectordb import VectorDBManager
from src.rag.retriever import ContextRetriever
from src.agents.data_agent import DataAgent
from src.integrations.cloud_sync import sync_s3, sync_adls

load_dotenv()

# Initialize paths
BASE_DATA_DIR = "./data/datasets"
LOCAL_UPLOADS_DIR = "./data/uploads_local"
LOCAL_UPLOADS_TABULAR_DIR = os.path.join(LOCAL_UPLOADS_DIR, "tabular")
LOCAL_UPLOADS_NON_TABULAR_DIR = os.path.join(LOCAL_UPLOADS_DIR, "non_tabular")
CLOUD_INGEST_DIR = "./data/cloud_ingest"
PLOTS_DIR = "./data/session_plots"
for d in [BASE_DATA_DIR, LOCAL_UPLOADS_TABULAR_DIR, LOCAL_UPLOADS_NON_TABULAR_DIR, CLOUD_INGEST_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

TABULAR_EXTENSIONS = (".csv", ".json", ".xlsx", ".xls")
NON_TABULAR_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

app_state: Dict[str, Any] = {
    "datasets": {},
    "dataset_sources": {},   # {dataset_name: source}
    "vector_db": None,
    "retriever": None,
    "agent": None,
    "current_session_id": None,
    "sessions": [],
    "prefetch_context_cache": {}  # {session_id: {text: context}}
}
state_lock = threading.Lock()

@app.after_request
def _disable_static_cache(response):
    if request.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "").strip("_")

def _normalize_dataset_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")

def _resolve_dataset_key(name: str) -> Optional[str]:
    if name in app_state["datasets"]:
        return name
    norm = _normalize_dataset_key(name)
    for key in app_state["datasets"].keys():
        if _normalize_dataset_key(key) == norm:
            return key
        base = key.split("__")[-1]
        if _normalize_dataset_key(base) == norm:
            return key
    return None


def _dataset_name_for(source: str, filename: str, container: Optional[str] = None) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = [source]
    if container:
        parts.append(_safe_name(container))
    parts.append(_safe_name(base))
    return "__".join([p for p in parts if p])


def _register_dataset(name: str, df, source: str, path: Optional[str] = None):
    app_state["datasets"][name] = df
    app_state["dataset_sources"][name] = source
    metadata = extract_metadata(df, name)
    app_state["vector_db"].add_dataset_metadata(metadata)
    app_state["vector_db"].add_tabular_chunks(name, df, source=source, path=path)


def _register_file(path: str, source: str):
    if not path or not os.path.isfile(path):
        return
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith(".pdf"):
        text = extract_pdf_text(path)
        if text:
            chunks = chunk_text(text, chunk_size=900, overlap=120)
            app_state["vector_db"].add_file_chunks(
                file_id=f"{source}::pdf::{name}",
                chunks=chunks,
                metadata={"name": name, "path": path, "source": source, "type": "pdf"}
            )
    elif lower.endswith(NON_TABULAR_EXTENSIONS):
        app_state["vector_db"].add_file_chunks(
            file_id=f"{source}::image::{name}",
            chunks=[f"Image file: {name}. Source: {source}. Path: {path}."],
            metadata={"name": name, "path": path, "source": source, "type": "image"}
        )


def _dataset_choices_for_sources(sources: Optional[list]) -> list:
    if not sources:
        return list(app_state["datasets"].keys())
    choices = []
    for name in app_state["datasets"].keys():
        src = app_state["dataset_sources"].get(name, "base")
        if src in sources:
            choices.append(name)
    return choices


def _find_dataset_mentions(question: str) -> list:
    if not question:
        return []
    q = question.lower()
    q_norm = _normalize_dataset_key(q)
    matches = []
    for name in app_state["datasets"].keys():
        name_lower = name.lower()
        base = name_lower.split("__")[-1]
        if name_lower in q or base in q:
            matches.append(name)
            continue
        if _normalize_dataset_key(name_lower) in q_norm or _normalize_dataset_key(base) in q_norm:
            matches.append(name)
    return list(dict.fromkeys(matches))


def _is_overview_query(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "what does", "show columns", "columns", "schema", "preview", "head", "sample",
        "what is in", "what's in", "what data", "data have"
    ]
    return any(k in q for k in keywords)


def _build_overview_response(dataset_name: str) -> str:
    df = app_state["datasets"].get(dataset_name)
    if df is None:
        return "Dataset not found."
    cols = ", ".join([str(c) for c in df.columns])
    sample = df.head(5).to_string(index=False)
    return (
        f"Columns in `{dataset_name}`:\\n{cols}\\n\\n"
        f"Sample rows:\\n{sample}"
    )


def _load_from_dir(datasets_dir: str, source: str, container: Optional[str] = None):
    if not os.path.exists(datasets_dir):
        return
    for filename in os.listdir(datasets_dir):
        if filename.lower().endswith(TABULAR_EXTENSIONS):
            file_path = os.path.join(datasets_dir, filename)
            df = load_dataset(file_path)
            if df is not None:
                name = _dataset_name_for(source, filename, container=container) if source != "base" else os.path.splitext(filename)[0]
                _register_dataset(name, df, source=source, path=file_path)


def _load_cloud_ingest():
    if not os.path.exists(CLOUD_INGEST_DIR):
        return
    for root, _, files in os.walk(CLOUD_INGEST_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            rel = os.path.relpath(root, CLOUD_INGEST_DIR)
            parts = rel.split(os.sep) if rel != "." else []
            source = parts[0] if parts else "cloud"
            container = parts[1] if len(parts) > 1 else None
            if filename.lower().endswith(TABULAR_EXTENSIONS):
                df = load_dataset(file_path)
                if df is not None:
                    name = _dataset_name_for(source, filename, container=container)
                    _register_dataset(name, df, source=source, path=file_path)
            elif filename.lower().endswith(NON_TABULAR_EXTENSIONS):
                _register_file(file_path, source=source)


def init_app_state():
    with state_lock:
        app_state["datasets"] = {}
        app_state["dataset_sources"] = {}
        app_state["vector_db"] = VectorDBManager()
        app_state["retriever"] = ContextRetriever(app_state["vector_db"], datasets=app_state["datasets"])
        app_state["agent"] = DataAgent(app_state["retriever"], app_state["datasets"], model_name="llama3.2")

        _load_from_dir(BASE_DATA_DIR, source="base")
        _load_from_dir(LOCAL_UPLOADS_TABULAR_DIR, source="upload")
        if os.path.exists(LOCAL_UPLOADS_NON_TABULAR_DIR):
            for filename in os.listdir(LOCAL_UPLOADS_NON_TABULAR_DIR):
                if filename.lower().endswith(NON_TABULAR_EXTENSIONS):
                    _register_file(os.path.join(LOCAL_UPLOADS_NON_TABULAR_DIR, filename), source="upload")
        _load_cloud_ingest()

        app_state["sessions"] = app_state["vector_db"].get_sessions()
        if not app_state["sessions"]:
            default_id = str(uuid.uuid4())
            app_state["vector_db"].add_session(default_id, "Default Session")
            app_state["sessions"] = app_state["vector_db"].get_sessions()
        app_state["current_session_id"] = app_state["sessions"][0]["id"]


def _load_cloud_env():
    return {
        "s3_bucket": os.getenv("S3_BUCKET", ""),
        "s3_prefix": os.getenv("S3_PREFIX", ""),
        "aws_profile": os.getenv("AWS_PROFILE", ""),
        "aws_region": os.getenv("AWS_REGION", ""),
        "adls_account_url": os.getenv("ADLS_ACCOUNT_URL", ""),
        "adls_filesystem": os.getenv("ADLS_FILE_SYSTEM", ""),
        "adls_prefix": os.getenv("ADLS_PREFIX", ""),
        "adls_account_key": os.getenv("ADLS_ACCOUNT_KEY", ""),
        "adls_sas": os.getenv("ADLS_SAS_TOKEN", ""),
        "adls_connection_string": os.getenv("ADLS_CONNECTION_STRING", ""),
    }


def sync_cloud_data():
    cfg = _load_cloud_env()

    if cfg["s3_bucket"]:
        container = cfg["s3_bucket"].strip()
        if container:
            dest_dir = os.path.join(CLOUD_INGEST_DIR, "s3", _safe_name(container))
            manifest_path = os.path.join(CLOUD_INGEST_DIR, "_manifests", f"s3_{_safe_name(container)}.json")
            paths, _ = sync_s3(
                bucket=container,
                prefix=cfg["s3_prefix"] or "",
                dest_dir=dest_dir,
                manifest_path=manifest_path,
                aws_profile=cfg["aws_profile"] or None,
                aws_region=cfg["aws_region"] or None,
            )
            _ingest_files_from_paths(paths, source="s3", container=container)

    if cfg["adls_account_url"] and cfg["adls_filesystem"]:
        container = cfg["adls_filesystem"].strip()
        dest_dir = os.path.join(CLOUD_INGEST_DIR, "adls", _safe_name(container))
        manifest_path = os.path.join(CLOUD_INGEST_DIR, "_manifests", f"adls_{_safe_name(container)}.json")
        paths, _ = sync_adls(
            account_url=cfg["adls_account_url"],
            file_system=container,
            prefix=cfg["adls_prefix"] or "",
            dest_dir=dest_dir,
            manifest_path=manifest_path,
            account_key=cfg["adls_account_key"] or None,
            sas_token=cfg["adls_sas"] or None,
            connection_string=cfg["adls_connection_string"] or None,
        )
        _ingest_files_from_paths(paths, source="adls", container=container)


def poll_cloud_sync():
    try:
        sync_cloud_data()
    except Exception as e:
        print(f"[cloud-sync] Poll failed: {e}")


def _start_polling_thread(interval_seconds: int = 60):
    def loop():
        while True:
            poll_cloud_sync()
            time.sleep(interval_seconds)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


def _ingest_files_from_paths(paths: list[str], source: str, container: Optional[str] = None):
    new_dataset_names = []
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()
        if ext in TABULAR_EXTENSIONS:
            df = load_dataset(path)
            if df is None:
                continue
            dataset_name = _dataset_name_for(source, filename, container=container)
            _register_dataset(dataset_name, df, source=source, path=path)
            new_dataset_names.append(dataset_name)
        elif ext in NON_TABULAR_EXTENSIONS:
            _register_file(path, source=source)
    app_state["retriever"].datasets = app_state["datasets"]
    app_state["agent"].datasets = app_state["datasets"]
    return new_dataset_names


@app.get("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/api/plots/<path:filename>")
def api_plots(filename: str):
    return send_from_directory(PLOTS_DIR, filename)


@app.get("/api/init")
def api_init():
    with state_lock:
        datasets = list(app_state["datasets"].keys())
        sessions = app_state["sessions"]
        current_session_id = app_state["current_session_id"]
    return jsonify({
        "datasets": datasets,
        "sessions": sessions,
        "current_session_id": current_session_id
    })


@app.get("/api/dataset/<name>")
def api_dataset_preview(name: str):
    with state_lock:
        df = app_state["datasets"].get(name)
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404
    preview = {
        "columns": list(df.columns),
        "dtypes": [str(t) for t in df.dtypes],
        "non_null": [int(v) for v in df.notnull().sum().values],
        "unique": [int(v) for v in df.nunique().values],
    }
    return jsonify(preview)


@app.get("/api/sessions")
def api_sessions():
    with state_lock:
        return jsonify(app_state["sessions"])


@app.post("/api/sessions")
def api_create_session():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        with state_lock:
            count = len(app_state["vector_db"].get_sessions())
        name = f"Session {count + 1}"
    new_id = str(uuid.uuid4())
    with state_lock:
        app_state["vector_db"].add_session(new_id, name)
        app_state["sessions"] = app_state["vector_db"].get_sessions()
        app_state["current_session_id"] = new_id
    return jsonify({"id": new_id, "name": name})


@app.put("/api/sessions/<sid>")
def api_rename_session(sid: str):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("name") or "").strip()
    if not new_name:
        return jsonify({"error": "Name required"}), 400
    with state_lock:
        app_state["vector_db"].rename_session(sid, new_name)
        app_state["sessions"] = app_state["vector_db"].get_sessions()
    return jsonify({"id": sid, "name": new_name})


@app.delete("/api/sessions/<sid>")
def api_delete_session(sid: str):
    with state_lock:
        app_state["vector_db"].delete_session(sid)
        app_state["sessions"] = app_state["vector_db"].get_sessions()
        if not app_state["sessions"]:
            default_id = str(uuid.uuid4())
            app_state["vector_db"].add_session(default_id, "Default Session")
            app_state["sessions"] = app_state["vector_db"].get_sessions()
        app_state["current_session_id"] = app_state["sessions"][0]["id"]
    return jsonify({"ok": True})


@app.post("/api/sessions/select")
def api_select_session():
    data = request.get_json(silent=True) or {}
    sid = data.get("id")
    if not sid:
        return jsonify({"error": "Session id required"}), 400
    with state_lock:
        app_state["current_session_id"] = sid
        turns = app_state["vector_db"].get_session_turns(sid)
    return jsonify({"turns": turns})


@app.post("/api/upload")
def api_upload():
    files = request.files.getlist("files")
    new_dataset_names = []
    non_tabular_paths = []
    for f in files:
        filename = f.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext in TABULAR_EXTENSIONS:
            new_path = os.path.join(LOCAL_UPLOADS_TABULAR_DIR, filename)
            f.save(new_path)
            df = load_dataset(new_path)
            if df is not None:
                dataset_name = _dataset_name_for("upload", filename)
                with state_lock:
                    _register_dataset(dataset_name, df, source="upload", path=new_path)
                    app_state["retriever"].datasets = app_state["datasets"]
                    app_state["agent"].datasets = app_state["datasets"]
                new_dataset_names.append(dataset_name)
        elif ext in NON_TABULAR_EXTENSIONS:
            new_path = os.path.join(LOCAL_UPLOADS_NON_TABULAR_DIR, filename)
            f.save(new_path)
            non_tabular_paths.append(new_path)
            with state_lock:
                _register_file(new_path, source="upload")
    return jsonify({"datasets": new_dataset_names, "non_tabular": [os.path.basename(p) for p in non_tabular_paths]})


@app.post("/api/chat/stream")
def api_chat_stream():
    message = request.form.get("message", "").strip()
    session_id = request.form.get("session_id") or app_state["current_session_id"]
    source_filter_raw = request.form.get("source_filter")
    source_filter = json.loads(source_filter_raw) if source_filter_raw else None

    files = request.files.getlist("files")
    new_dataset_names = []
    non_tabular_paths = []
    for f in files:
        filename = f.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext in TABULAR_EXTENSIONS:
            new_path = os.path.join(LOCAL_UPLOADS_TABULAR_DIR, filename)
            f.save(new_path)
            df = load_dataset(new_path)
            if df is not None:
                dataset_name = _dataset_name_for("upload", filename)
                with state_lock:
                    _register_dataset(dataset_name, df, source="upload", path=new_path)
                    app_state["retriever"].datasets = app_state["datasets"]
                    app_state["agent"].datasets = app_state["datasets"]
                new_dataset_names.append(dataset_name)
        elif ext in NON_TABULAR_EXTENSIONS:
            new_path = os.path.join(LOCAL_UPLOADS_NON_TABULAR_DIR, filename)
            f.save(new_path)
            non_tabular_paths.append(new_path)
            with state_lock:
                _register_file(new_path, source="upload")

    upload_context = get_upload_context_for_non_tabular(non_tabular_paths) if non_tabular_paths else ""

    if not message:
        return jsonify({"error": "Message required"}), 400

    allowed_sources = source_filter or None
    allowed_dataset_names = None
    if allowed_sources:
        allowed_dataset_names = [
            name for name, src in app_state["dataset_sources"].items() if src in allowed_sources
        ]
    matched_datasets = _find_dataset_mentions(message)
    if matched_datasets:
        resolved = []
        for name in matched_datasets:
            resolved_name = _resolve_dataset_key(name) or name
            resolved.append(resolved_name)
        if allowed_dataset_names is None:
            allowed_dataset_names = resolved
        else:
            allowed_dataset_names = list(set(allowed_dataset_names) | set(resolved))
    if allowed_dataset_names is not None and new_dataset_names:
        allowed_dataset_names = list(set(allowed_dataset_names) | set(new_dataset_names))

    if matched_datasets and _is_overview_query(message):
        overview_text = _build_overview_response(matched_datasets[0])
        def generate_overview():
            yield f"data: {json.dumps({'type': 'final', 'content': overview_text})}\n\n"
        try:
            app_state["vector_db"].add_memory(session_id, message, overview_text)
        except Exception as e:
            print(f"[chat] Failed to save memory: {e}")
        return Response(generate_overview(), mimetype="text/event-stream")

    def generate():
        yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking...'})}\n\n"
        prefetched = app_state["prefetch_context_cache"].get(session_id)
        prefetched_context = None
        if prefetched and prefetched.get("text") == message:
            prefetched_context = prefetched.get("context")
        preferred_names = new_dataset_names if new_dataset_names else (matched_datasets if matched_datasets else None)

        gen = app_state["agent"].run_stream(
            message,
            session_id=session_id,
            prefetched_context=prefetched_context,
            preferred_dataset_names=preferred_names,
            upload_context=upload_context if upload_context else None,
            allowed_dataset_names=allowed_dataset_names,
            allowed_sources=allowed_sources,
        )

        full_narrative = ""
        first_token_received = False
        final_response = ""
        last_code = ""
        plotly_json_path = None
        image_path = None
        for token, fig, code, error_msg, run_output in gen:
            if token:
                if not first_token_received:
                    first_token_received = True
                full_narrative += token
                yield f"data: {json.dumps({'type': 'token', 'content': full_narrative})}\n\n"
            if code and not fig and not error_msg and not run_output:
                if code != last_code:
                    last_code = code
                    yield f"data: {json.dumps({'type': 'code', 'content': code})}\n\n"
                continue
            if fig or code or error_msg:
                if code and code != last_code:
                    last_code = code
                    yield f"data: {json.dumps({'type': 'code', 'content': code})}\n\n"
                # If we have a figure, send it as a separate event
                if fig is not None:
                    try:
                        if hasattr(fig, "to_plotly_json"):
                            from plotly.utils import PlotlyJSONEncoder
                            plotly_json = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
                            yield f"data: {json.dumps({'type': 'plotly', 'content': plotly_json})}\n\n"
                            try:
                                plot_id = f"{session_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
                                plot_path = os.path.join(PLOTS_DIR, plot_id)
                                with open(plot_path, "w") as f:
                                    json.dump(plotly_json, f)
                                plotly_json_path = plot_id
                            except Exception:
                                pass
                        else:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", bbox_inches="tight")
                            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            yield f"data: {json.dumps({'type': 'image', 'content': b64})}\n\n"
                            try:
                                plot_id = f"{session_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
                                plot_path = os.path.join(PLOTS_DIR, plot_id)
                                with open(plot_path, "wb") as f:
                                    f.write(buf.getvalue())
                                image_path = plot_id
                            except Exception:
                                pass
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Plot render failed: {e}'})}\n\n"
                narrative_only = re.sub(r"```\\w*.*?```", "", full_narrative, flags=re.DOTALL | re.IGNORECASE).strip()
                response = narrative_only
                if run_output:
                    response += "\\n\\n" + run_output
                response += "\\n\\n"
                if error_msg:
                    response += f"⚠️ **Error:**\\n```\\n{error_msg}\\n```\\n\\n"
                final_response = response.strip()
                yield f"data: {json.dumps({'type': 'final', 'content': final_response})}\n\n"

        if final_response:
            try:
                app_state["vector_db"].add_memory(
                    session_id,
                    message,
                    final_response,
                    image_path=image_path,
                    assistant_code=last_code or None,
                    plotly_json_path=plotly_json_path
                )
            except Exception as e:
                print(f"[chat] Failed to save memory: {e}")

    resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.post("/api/prefetch")
def api_prefetch():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id") or app_state["current_session_id"]
    source_filter = data.get("source_filter")
    if len(text) < 5:
        return jsonify({"ok": True})

    allowed_sources = source_filter or None
    allowed_dataset_names = None
    if allowed_sources:
        allowed_dataset_names = [
            name for name, src in app_state["dataset_sources"].items() if src in allowed_sources
        ]

    dataset_context = app_state["retriever"].get_relevant_context(
        text,
        top_k_datasets=8,
        top_k_files=8,
        allowed_dataset_names=allowed_dataset_names,
        allowed_sources=allowed_sources,
    )
    memory_context = app_state["vector_db"].get_memory(session_id, text)
    combined_context = dataset_context + ("\n" + memory_context if memory_context else "")
    app_state["prefetch_context_cache"][session_id] = {"text": text, "context": combined_context}
    return jsonify({"ok": True})


def create_app():
    init_app_state()
    interval = int(os.getenv("CLOUD_SYNC_INTERVAL_SECONDS", "60"))
    _start_polling_thread(interval_seconds=interval)
    return app


if __name__ == "__main__":
    create_app()
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=int(os.getenv("PORT", "8051")))
