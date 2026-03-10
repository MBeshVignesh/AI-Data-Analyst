import os
import re
import time
import gradio as gr
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import shutil
import uuid

from src.utils.data_loader import load_all_datasets, load_dataset
from src.utils.file_parsers import get_upload_context_for_non_tabular


def _strip_code_blocks(text: str) -> str:
    """Remove markdown code blocks so we can show narrative + code once (no duplication)."""
    if not text or not text.strip():
        return text
    return re.sub(r"```\w*.*?```", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
from src.utils.metadata_extractor import extract_metadata
from src.rag.vectordb import VectorDBManager
from src.rag.retriever import ContextRetriever
from src.agents.data_agent import DataAgent

# Initialize paths
DATA_DIR = "./data/datasets"
PLOTS_DIR = "./data/session_plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Tabular = show in dataset preview. Non-tabular = don't show in preview but answer from content (PDF text, image).
TABULAR_EXTENSIONS = (".csv", ".json", ".xlsx", ".xls")
NON_TABULAR_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp")
UPLOADS_DIR = "./data/uploads"  # non-tabular files (PDF, images) for context
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global Application State
app_state = {
    "datasets": {},
    "vector_db": None,
    "retriever": None,
    "agent": None,
    "current_session_id": None,
    "sessions": [],
    "prefetch_context_cache": {} # {session_id: {text: context}}
}

def init_app():
    """Initializes the backend components and loads initial datasets and sessions."""
    print("Initializing application...")
    app_state["datasets"] = load_all_datasets(DATA_DIR)

    app_state["vector_db"] = VectorDBManager()

    for name, df in app_state["datasets"].items():
        metadata = extract_metadata(df, name)
        app_state["vector_db"].add_dataset_metadata(metadata)

    app_state["retriever"] = ContextRetriever(app_state["vector_db"], datasets=app_state["datasets"])
    app_state["agent"] = DataAgent(app_state["retriever"], app_state["datasets"], model_name="llama3.2")

    # Load or create a default session
    app_state["sessions"] = app_state["vector_db"].get_sessions()
    if not app_state["sessions"]:
        default_id = str(uuid.uuid4())
        app_state["vector_db"].add_session(default_id, "Default Session")
        app_state["sessions"] = app_state["vector_db"].get_sessions()

    app_state["current_session_id"] = app_state["sessions"][0]["id"]

    print(f"Initialization complete. Loaded datasets: {list(app_state['datasets'].keys())}")
    return list(app_state["datasets"].keys())


# ── Dataset helpers ──────────────────────────────────────────────────────────

def render_dataset_preview(dataset_name):
    """Returns metadata of the selected dataset as a DataFrame."""
    if not dataset_name or dataset_name not in app_state["datasets"]:
        return None
    df = app_state["datasets"][dataset_name]
    return pd.DataFrame({
        "Column":        df.columns,
        "Type":          df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique().values
    })


# ── Session helpers ──────────────────────────────────────────────────────────

def _session_html():
    """Build a compact HTML list of sessions for the sidebar."""
    sessions = app_state["sessions"]
    if not sessions:
        return "<p style='color:#888'>No sessions</p>"

    rows = []
    for s in sessions:
        sid   = s["id"]
        sname = s["name"]
        active = sid == app_state["current_session_id"]
        style = (
            "display:flex;align-items:center;justify-content:space-between;"
            "padding:6px 8px;border-radius:6px;margin-bottom:4px;"
            + ("background:#2563eb22;border:1px solid #2563eb;" if active else "background:#1f2937;border:1px solid transparent;")
        )
        rows.append(
            f"<div style='{style}'>"
            f"<span style='flex:1;cursor:pointer;font-size:13px;color:{'#93c5fd' if active else '#d1d5db'};' "
            f"  onclick=\"document.getElementById('_sess_select').value='{sid}';"
            f"  document.getElementById('_sess_select').dispatchEvent(new Event('input',{{bubbles:true}}))\"> "
            f"{sname}</span>"
            f"<span title='Rename' style='cursor:pointer;margin-left:6px;font-size:14px;' "
            f"  onclick=\"document.getElementById('_sess_edit').value='{sid}|{sname}';"
            f"  document.getElementById('_sess_edit').dispatchEvent(new Event('input',{{bubbles:true}}))\">✏️</span>"
            f"<span title='Delete' style='cursor:pointer;margin-left:6px;font-size:14px;' "
            f"  onclick=\"document.getElementById('_sess_del').value='{sid}';"
            f"  document.getElementById('_sess_del').dispatchEvent(new Event('input',{{bubbles:true}}))\">🗑️</span>"
            f"</div>"
        )
    return "".join(rows)


def _sessions_dropdown_choices():
    return [f"{s['name']} (…{s['id'][-6:]})" for s in app_state["sessions"]]


def _session_id_from_short(short_id: str) -> str:
    for s in app_state["sessions"]:
        if s["id"].endswith(short_id) or s["id"] == short_id:
            return s["id"]
    return app_state["current_session_id"]


def _turns_to_chat_history(turns: list) -> list:
    """Build Gradio chatbot history from stored turns (text + optional persisted plot image)."""
    history = []
    for t in turns:
        history.append({"role": "user", "content": t["user"]})
        history.append({"role": "assistant", "content": t["assistant"]})
        # If this turn had a saved plot, add it as a separate assistant message (image)
        image_path = t.get("image_path")
        if image_path and os.path.isfile(image_path):
            history.append({"role": "assistant", "content": {"path": image_path}})
    return history


def select_session(session_label):
    """Switch active session and restore its full chat history from ChromaDB."""
    if not session_label:
        return [], gr.Dropdown()
    try:
        short = session_label.split("(…")[-1].rstrip(")")
        sid = _session_id_from_short(short)
    except Exception:
        sid = app_state["current_session_id"]
    app_state["current_session_id"] = sid

    # Restore full chat history (and any persisted plots) from ChromaDB
    turns = app_state["vector_db"].get_session_turns(sid)
    history = _turns_to_chat_history(turns)
    return history, gr.Dropdown(choices=_sessions_dropdown_choices(), value=session_label)


def create_new_session(name=None):
    if not name or not name.strip():
        # Get count to generate a reasonable default
        sessions = app_state["vector_db"].get_sessions()
        name = f"Session {len(sessions) + 1}"
    
    new_id = str(uuid.uuid4())
    app_state["vector_db"].add_session(new_id, name.strip())
    app_state["sessions"] = app_state["vector_db"].get_sessions()
    app_state["current_session_id"] = new_id
    label = f"{name.strip()} (…{new_id[-6:]})"
    return (
        gr.Dropdown(choices=_sessions_dropdown_choices(), value=label),
        gr.Textbox(value=""),   # clear input
        []                      # clear chat
    )


def rename_session_fn(new_name, session_label):
    if not new_name or not session_label:
        return gr.Dropdown(), gr.Textbox()
    try:
        short = session_label.split("(…")[-1].rstrip(")")
        sid = _session_id_from_short(short)
    except Exception:
        return gr.Dropdown(), gr.Textbox()
    app_state["vector_db"].rename_session(sid, new_name.strip())
    app_state["sessions"] = app_state["vector_db"].get_sessions()
    new_label = f"{new_name.strip()} (…{sid[-6:]})"
    return gr.Dropdown(choices=_sessions_dropdown_choices(), value=new_label), gr.Textbox(value="")


def delete_session_fn(session_label):
    if not session_label:
        return gr.Dropdown(), []
    try:
        short = session_label.split("(…")[-1].rstrip(")")
        sid = _session_id_from_short(short)
    except Exception:
        return gr.Dropdown(), []
    app_state["vector_db"].delete_session(sid)
    app_state["sessions"] = app_state["vector_db"].get_sessions()
    if not app_state["sessions"]:
        default_id = str(uuid.uuid4())
        app_state["vector_db"].add_session(default_id, "Default Session")
        app_state["sessions"] = app_state["vector_db"].get_sessions()
    app_state["current_session_id"] = app_state["sessions"][0]["id"]
    first_label = _sessions_dropdown_choices()[0]
    return gr.Dropdown(choices=_sessions_dropdown_choices(), value=first_label), []


# ── Chat ─────────────────────────────────────────────────────────────────────

def prefetch_context(message, session_label):
    """Background task to fetch context while user types."""
    if not message or not message.get("text"):
        return
    
    text = message["text"].strip()
    if len(text) < 5: # Don't pre-fetch for very short fragments
        return

    session_id = app_state["current_session_id"]
    if session_label:
        try:
            short = session_label.split("(…")[-1].rstrip(")")
            session_id = _session_id_from_short(short)
        except Exception: pass

    # Perform retrieval
    dataset_context = app_state["retriever"].get_relevant_datasets(text, top_k=5)
    memory_context = app_state["vector_db"].get_memory(session_id, text)
    combined_context = dataset_context + ("\n" + memory_context if memory_context else "")
    
    # Cache it
    app_state["prefetch_context_cache"][session_id] = {
        "text": text,
        "context": combined_context
    }
    print(f"[UI] Prefetched context for '{text[:20]}...'")

def submit_question(message, history, session_label):
    """Handles user question and file uploads, runs the agent, formats output."""
    session_id = app_state["current_session_id"]
    if session_label:
        try:
            short = session_label.split("(…")[-1].rstrip(")")
            session_id = _session_id_from_short(short)
        except Exception:
            pass

    if not history:
        history = []

    text  = message.get("text", "").strip()
    files = message.get("files", [])

    # ── Process uploaded files: tabular (CSV/JSON/XLSX) → dataset preview; PDF/image → context only, no preview ──
    new_dataset_names = []
    non_tabular_paths = []
    for f in files:
        filename = os.path.basename(f)
        ext = os.path.splitext(filename)[1].lower()
        name = os.path.splitext(filename)[0]
        if ext in TABULAR_EXTENSIONS:
            new_path = os.path.join(DATA_DIR, filename)
            shutil.copy(f, new_path)
            df = load_dataset(new_path)
            if df is not None:
                app_state["datasets"][name] = df
                app_state["retriever"].datasets = app_state["datasets"]
                app_state["agent"].datasets = app_state["datasets"]
                metadata = extract_metadata(df, name)
                app_state["vector_db"].add_dataset_metadata(metadata)
                new_dataset_names.append(name)
                history.append({"role": "user",      "content": {"path": f}})
                history.append({"role": "assistant",  "content": f"✅ Loaded dataset **`{name}`** into context. You can ask questions about this data."})
                yield (history,
                       gr.MultimodalTextbox(),
                       gr.Dropdown(choices=list(app_state["datasets"].keys()), value=name))
        elif ext in NON_TABULAR_EXTENSIONS:
            new_path = os.path.join(UPLOADS_DIR, filename)
            shutil.copy(f, new_path)
            non_tabular_paths.append(new_path)
            history.append({"role": "user",      "content": {"path": f}})
            history.append({"role": "assistant",  "content": f"✅ Uploaded **{filename}** for context. Ask your question and I'll answer using this file."})

    upload_context = get_upload_context_for_non_tabular(non_tabular_paths) if non_tabular_paths else ""

    if not text:
        if files:
            yield (history,
                   gr.MultimodalTextbox(value=None),
                   gr.Dropdown(choices=list(app_state["datasets"].keys())))
        return

    history.append({"role": "user",     "content": text})
    history.append({"role": "assistant","content": ""}) # Start empty for streaming
    yield (history,
           gr.MultimodalTextbox(value=None),
           gr.Dropdown(choices=list(app_state["datasets"].keys())))

    # Check for pre-fetched context
    prefetched = app_state["prefetch_context_cache"].get(session_id)
    prefetched_context = None
    if prefetched and prefetched["text"] == text:
        prefetched_context = prefetched["context"]
        print(f"[UI] Using pre-fetched context for session {session_id[:8]}")
    
    # Stream the response (prefer just-uploaded tabular data; include PDF/image context)
    gen = app_state["agent"].run_stream(
        text,
        session_id=session_id,
        prefetched_context=prefetched_context,
        preferred_dataset_names=new_dataset_names if new_dataset_names else None,
        upload_context=upload_context if upload_context else None,
    )
    
    full_narrative = ""
    first_token_received = False
    
    run_output = ""
    for token, fig, code, error_msg, run_output in gen:
        if token:
            if not first_token_received:
                first_token_received = True
            full_narrative += token
            history[-1]["content"] = full_narrative
            yield (history, gr.MultimodalTextbox(value=None), gr.Dropdown(choices=list(app_state["datasets"].keys())))
        
        elif not first_token_received and not fig and not code and not error_msg:
            # If we're waiting for the first token, show thinking
            history[-1]["content"] = " Thinking..."
            yield (history, gr.MultimodalTextbox(value=None), gr.Dropdown(choices=list(app_state["datasets"].keys())))
        
        if fig or code or error_msg:
            # Final yield: show narrative once (strip embedded code to avoid duplication), run output (e.g. table), then code once
            narrative_only = _strip_code_blocks(full_narrative)
            response = narrative_only
            if run_output:
                response += "\n\n" + run_output
            response += "\n\n"
            if error_msg:
                response += f"⚠️ **Error:**\n```\n{error_msg}\n```\n\n"
            if code:
                for block in code.split("\n\n---\n\n"):
                    block = block.strip()
                    if block:
                        response += f"\n```{block}\n```\n"

            final_response = response.strip()
            history[-1]["content"] = final_response

            # Persist plot to a permanent path so it survives refresh
            persistent_plot_path = None
            if fig:
                session_plots_dir = os.path.join(PLOTS_DIR, session_id)
                os.makedirs(session_plots_dir, exist_ok=True)
                turn_id = f"{session_id}_{int(time.time() * 1000)}"
                persistent_plot_path = os.path.join(session_plots_dir, f"{turn_id}.png")
                try:
                    if hasattr(fig, "savefig"):
                        fig.savefig(persistent_plot_path, bbox_inches="tight")
                    elif hasattr(fig, "write_image"):
                        fig.write_image(persistent_plot_path)
                    history.append({"role": "assistant", "content": {"path": persistent_plot_path}})
                except Exception as e:
                    print(f"[UI] Failed to save plot: {e}")
                    persistent_plot_path = None
                    img_path = os.path.join(tempfile.gettempdir(), f"plot_{uuid.uuid4().hex}.png")
                    if hasattr(fig, "savefig"):
                        fig.savefig(img_path, bbox_inches="tight")
                    elif hasattr(fig, "write_image"):
                        fig.write_image(img_path)
                    history.append({"role": "assistant", "content": {"path": img_path}})

            yield (history,
                   gr.MultimodalTextbox(value=None),
                   gr.Dropdown(choices=list(app_state["datasets"].keys())))

            # Save this turn to session memory (text + code; plot path so refresh restores it)
            try:
                app_state["vector_db"].add_memory(
                    session_id, text, final_response, image_path=persistent_plot_path
                )
            except Exception as e:
                print(f"[UI] Failed to save memory: {e}")

    # Clear cache for this session
    if session_id in app_state["prefetch_context_cache"]:
        del app_state["prefetch_context_cache"][session_id]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def create_ui():
    datasets_list = init_app()

    custom_css = """
    #new-session-btn {
        background-color: #f3f4f6 !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        margin-bottom: 8px !important;
    }
    #new-session-btn:hover {
        background-color: #e5e7eb !important;
    }
    .icon-button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-width: 30px !important;
        max-width: 30px !important;
        padding: 0 !important;
        font-size: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .icon-button:hover {
        background-color: #f3f4f6 !important;
    }
    .compact-row {
        align-items: center !important;
        gap: 0 !important;
    }
    /* Fix for dropdown alignment */
    .compact-row > .form {
        margin-bottom: 0 !important;
    }
    """

    with gr.Blocks(title="AI Data Analyst", css=custom_css) as demo:

        gr.Markdown("#  AI Data Analyst\n*Ask questions, upload data, and get instant insights.*")

        with gr.Row():
            # ── Sidebar ──────────────────────────────────
            with gr.Column(scale=1, min_width=280):

                # ── Sessions ──
                with gr.Group(elem_id="session-panel"):
                    gr.Markdown("### Sessions")
                    
                    new_session_btn = gr.Button("＋ New Session", variant="secondary", elem_id="new-session-btn")
                    
                    # Normal view: Dropdown + Rename/Delete icons
                    with gr.Row(variant="compact", visible=True, elem_classes="compact-row") as session_view:
                        sessions_dropdown = gr.Dropdown(
                            choices=_sessions_dropdown_choices(),
                            value=_sessions_dropdown_choices()[0] if _sessions_dropdown_choices() else None,
                            show_label=False,
                            interactive=True,
                            scale=4
                        )
                        rename_btn = gr.Button("✏️", variant="secondary", scale=0, min_width=30, elem_classes="icon-button")
                        delete_btn = gr.Button("🗑️", variant="secondary", scale=0, min_width=30, elem_classes="icon-button")

                    # Edit view: Textbox + Save/Cancel icons
                    with gr.Row(variant="compact", visible=False, elem_classes="compact-row") as rename_view:
                        rename_input = gr.Textbox(
                            placeholder="New name…",
                            show_label=False,
                            scale=4
                        )
                        save_rename_btn = gr.Button("✅", variant="secondary", scale=0, min_width=30, elem_classes="icon-button")
                        cancel_rename_btn = gr.Button("❌", variant="secondary", scale=0, min_width=30, elem_classes="icon-button")

                gr.Markdown("---")

                # ── Dataset management ──
                gr.Markdown("### 📂 Dataset")
                dataset_dropdown = gr.Dropdown(
                    choices=datasets_list,
                    value=datasets_list[0] if datasets_list else None,
                    label="Preview Dataset",
                    interactive=True
                )
                gr.Markdown("#### 📋 Metadata")
                data_preview = gr.Dataframe(interactive=False, max_height=250)

            # ── Main chat area ────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=580,
                    label="Chat",
                    show_label=False
                )
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_types=[".csv", ".json", ".xlsx", ".xls", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp"],
                    placeholder="Ask a question or upload data…",
                    show_label=False
                )

        # ── Event wiring ──────────────────────────────────

        # Session events
        sessions_dropdown.change(
            fn=select_session,
            inputs=[sessions_dropdown],
            outputs=[chatbot, sessions_dropdown]
        )
        new_session_btn.click(
            fn=create_new_session,
            inputs=[],
            outputs=[sessions_dropdown, rename_input, chatbot]
        )

        # Toggle to Rename View
        def show_rename_ui(label):
            current_name = label.split(" (…")[0] if label else ""
            return gr.Row(visible=False), gr.Row(visible=True), gr.Textbox(value=current_name)

        rename_btn.click(
            fn=show_rename_ui,
            inputs=[sessions_dropdown],
            outputs=[session_view, rename_view, rename_input]
        )

        # Cancel Rename
        def hide_rename_ui():
            return gr.Row(visible=True), gr.Row(visible=False)

        cancel_rename_btn.click(
            fn=hide_rename_ui,
            outputs=[session_view, rename_view]
        )

        # Save Rename
        def save_rename_and_hide(new_name, old_label):
            # Reuse the existing rename_session_fn logic
            updated_dropdown, _ = rename_session_fn(new_name, old_label)
            return gr.Row(visible=True), gr.Row(visible=False), updated_dropdown

        save_rename_btn.click(
            fn=save_rename_and_hide,
            inputs=[rename_input, sessions_dropdown],
            outputs=[session_view, rename_view, sessions_dropdown]
        )

        delete_btn.click(
            fn=delete_session_fn,
            inputs=[sessions_dropdown],
            outputs=[sessions_dropdown, chatbot]
        )

        # Dataset preview
        dataset_dropdown.change(
            fn=render_dataset_preview,
            inputs=[dataset_dropdown],
            outputs=[data_preview]
        )
        def on_page_load(session_label):
            """Restore dataset preview, current session's chat history, and session list on refresh/load."""
            # Always refresh sessions from DB so dropdown shows only real, current sessions
            app_state["sessions"] = app_state["vector_db"].get_sessions()
            choices = _sessions_dropdown_choices()
            if not app_state["sessions"]:
                preview = render_dataset_preview(datasets_list[0] if datasets_list else None)
                return preview, [], gr.Dropdown(choices=[], value=None)

            # Resolve selected session from dropdown (or keep current)
            sid = app_state["current_session_id"]
            if session_label:
                try:
                    short = session_label.split("(…")[-1].rstrip(")")
                    sid = _session_id_from_short(short)
                    app_state["current_session_id"] = sid
                except Exception:
                    pass
            # Ensure sid is valid
            if sid not in [s["id"] for s in app_state["sessions"]]:
                sid = app_state["sessions"][0]["id"]
                app_state["current_session_id"] = sid

            current_label = next(
                (f"{s['name']} (…{s['id'][-6:]})" for s in app_state["sessions"] if s["id"] == sid),
                choices[0] if choices else None
            )

            preview = render_dataset_preview(datasets_list[0] if datasets_list else None)
            history = []
            try:
                turns = app_state["vector_db"].get_session_turns(sid)
                history = _turns_to_chat_history(turns)
            except Exception as e:
                print(f"[UI] on_page_load error: {e}")
            return preview, history, gr.Dropdown(choices=choices, value=current_label)

        demo.load(
            fn=on_page_load,
            inputs=[sessions_dropdown],
            outputs=[data_preview, chatbot, sessions_dropdown]
        )

        # Chat: outputs include dataset_dropdown so it refreshes after file uploads
        chat_input.submit(
            fn=submit_question,
            inputs=[chat_input, chatbot, sessions_dropdown],
            outputs=[chatbot, chat_input, dataset_dropdown]
        )
        
        # Performance: Pre-fetch context while typing
        chat_input.change(
            fn=prefetch_context,
            inputs=[chat_input, sessions_dropdown],
            outputs=None,
            show_progress="hidden"
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0")
