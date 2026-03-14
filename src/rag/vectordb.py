import os
import re
import chromadb
from .embeddings import get_embedding_model
from typing import List, Dict, Any, Optional
from ..utils.chunking import tabular_to_chunks

class VectorDBManager:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embeddings
        self.embedding_function = get_embedding_model()
        
        # Wrapper for Chroma
        class LangchainEmbeddingWrapper:
            def __init__(self, lc_embedder):
                self.lc_embedder = lc_embedder
            
            def __call__(self, input: Any) -> List[List[float]]:
                if isinstance(input, str):
                    input = [input]
                return self.lc_embedder.embed_documents(input)

            def name(self) -> str:
                return "langchain-embedding-wrapper"

            def embed_query(self, *args, **kwargs) -> List[List[float]]:
                text = args[0] if args else kwargs.get("input", kwargs.get("text", ""))
                if isinstance(text, list):
                    text = text[0] if len(text) > 0 else ""
                text = str(text)
                return [self.lc_embedder.embed_query(text)]

            def embed_documents(self, *args, **kwargs) -> List[List[float]]:
                texts = args[0] if args else kwargs.get("input", kwargs.get("texts", []))
                if isinstance(texts, str):
                    texts = [texts]
                elif not isinstance(texts, list):
                    texts = [str(texts)]
                texts = [str(t) for t in texts]
                return self.lc_embedder.embed_documents(texts)
        
        self.wrapper = LangchainEmbeddingWrapper(self.embedding_function)
        
        # Collections
        self.metadata_collection = self.client.get_or_create_collection(
            name="dataset_metadata",
            embedding_function=self.wrapper
        )
        
        self.sessions_collection = self.client.get_or_create_collection(
            name="user_sessions",
            embedding_function=self.wrapper
        )
        
        self.memory_collection = self.client.get_or_create_collection(
            name="chat_memory",
            embedding_function=self.wrapper
        )

        # File chunks for RAG (text + tabular chunks)
        self.file_collection = self.client.get_or_create_collection(
            name="file_chunks",
            embedding_function=self.wrapper
        )

        # Cache per-session memory collections to keep sessions fully isolated
        self._session_memory_collections: Dict[str, Any] = {}
        self._cleanup_orphan_session_memory_collections()

    def _session_memory_collection_name(self, session_id: str) -> str:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return f"chat_memory__{safe_id}"

    def _list_collection_names(self) -> List[str]:
        names: List[str] = []
        try:
            for col in self.client.list_collections():
                names.append(col.name if hasattr(col, "name") else str(col))
        except Exception:
            return []
        return names

    def _get_session_memory_collection(self, session_id: str, create_if_missing: bool = True):
        if session_id not in self._session_memory_collections:
            name = self._session_memory_collection_name(session_id)
            if create_if_missing:
                self._session_memory_collections[session_id] = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self.wrapper
                )
            else:
                try:
                    self._session_memory_collections[session_id] = self.client.get_collection(
                        name=name,
                        embedding_function=self.wrapper
                    )
                except Exception:
                    return None
        return self._session_memory_collections[session_id]

    def _delete_session_memory_collection(self, session_id: str):
        name = self._session_memory_collection_name(session_id)
        self._session_memory_collections.pop(session_id, None)
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass

    def _cleanup_orphan_session_memory_collections(self):
        """Delete per-session memory collections that no longer map to active sessions."""
        try:
            session_names = {
                self._session_memory_collection_name(s["id"])
                for s in self.get_sessions()
                if s.get("id")
            }
            for name in self._list_collection_names():
                if not name.startswith("chat_memory__"):
                    continue
                if name not in session_names:
                    try:
                        self.client.delete_collection(name=name)
                    except Exception:
                        pass
        except Exception:
            pass
        
    def add_dataset_metadata(self, metadata: Dict[str, Any]):
        """Adds dataset metadata to the vector DB. Uses a rich searchable document for better retrieval."""
        name = metadata["name"]
        # Embed the searchable doc (natural-language + columns + sample) so queries match better
        document = metadata.get("searchable_doc") or metadata["summary"]
        self.metadata_collection.upsert(
            documents=[document],
            metadatas=[{"name": name}],
            ids=[name]
        )
        print(f"Added/Updated metadata for dataset: {name}")

    def list_indexed_dataset_names(self) -> List[str]:
        try:
            results = self.metadata_collection.get()
            return results.get("ids") or []
        except Exception:
            return []

    def delete_dataset_index(self, dataset_name: str):
        try:
            self.metadata_collection.delete(ids=[dataset_name])
        except Exception:
            pass
        # Remove associated tabular chunks using both modern metadata and legacy file_id keys.
        try:
            self.file_collection.delete(where={"dataset": dataset_name})
        except Exception:
            pass
        try:
            self.file_collection.delete(where={"file_id": f"tabular::{dataset_name}"})
        except Exception:
            pass

    def add_file_chunks(
        self,
        file_id: str,
        chunks: List[str],
        metadata: Dict[str, Any],
    ):
        # Replace file chunks atomically-ish to avoid stale tail chunks when files shrink.
        try:
            self.file_collection.delete(where={"file_id": file_id})
        except Exception:
            pass
        if not chunks:
            return
        ids = [f"{file_id}::{i}" for i in range(len(chunks))]
        metadatas = []
        for i in range(len(chunks)):
            meta = dict(metadata)
            meta["chunk_index"] = i
            meta["file_id"] = file_id
            metadatas.append(meta)
        self.file_collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def add_tabular_chunks(
        self,
        dataset_name: str,
        df: Any,
        source: str,
        path: Optional[str] = None,
    ):
        chunks = tabular_to_chunks(df, dataset_name)
        if not chunks:
            return
        meta = {
            "type": "tabular",
            "dataset": dataset_name,
            "source": source,
        }
        if path:
            meta["path"] = path
        self.add_file_chunks(f"tabular::{dataset_name}", chunks, meta)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Searches for relevant datasets based on a query."""
        results = self.metadata_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        extracted_results = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                extracted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
                
        return extracted_results

    def search_files(
        self,
        query: str,
        n_results: int = 6,
        allowed_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            where = None
            if allowed_sources is not None:
                if len(allowed_sources) == 0:
                    return []
                where = {"source": {"$in": allowed_sources}}
            results = self.file_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            extracted = []
            if results and results.get("documents") and results["documents"]:
                for i in range(len(results["documents"][0])):
                    extracted.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })
            return extracted
        except Exception:
            return []

    # --- Session Management ---
    
    def get_sessions(self) -> List[Dict[str, str]]:
        """Returns a list of all chat sessions."""
        results = self.sessions_collection.get()
        sessions = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                sessions.append({
                    "id": results["ids"][i],
                    "name": results["metadatas"][i]["name"]
                })
        return sessions

    def add_session(self, session_id: str, name: str):
        """Creates a new session."""
        self.sessions_collection.upsert(
            documents=[name],  # Document is the name for searchability if needed
            metadatas=[{"name": name}],
            ids=[session_id]
        )

    def rename_session(self, session_id: str, new_name: str):
        """Renames an existing session."""
        self.sessions_collection.update(
            ids=[session_id],
            documents=[new_name],
            metadatas=[{"name": new_name}]
        )

    def delete_session(self, session_id: str):
        """Deletes a session and its associated memory."""
        self.sessions_collection.delete(ids=[session_id])
        # Also clear any memory associated with this session (legacy + per-session)
        try:
            self.memory_collection.delete(where={"session_id": session_id})
        except Exception:
            pass
        self._delete_session_memory_collection(session_id)

    def add_memory(
        self,
        session_id: str,
        user_input: str,
        assistant_output: str,
        image_path: Optional[str] = None,
        assistant_code: Optional[str] = None,
        plotly_json_path: Optional[str] = None,
    ):
        """Adds a conversation turn to a session's history with a timestamp.
        If image_path or plotly_json_path is provided (e.g. a saved plot), it is stored so the UI can restore it on refresh.
        """
        import time
        ts = int(time.time() * 1000)
        turn_id = f"{session_id}_{ts}"
        combined_text = f"User: {user_input}\nAssistant: {assistant_output}"
        meta = {
            "session_id": session_id,
            "user": user_input,
            "assistant": assistant_output,
            "timestamp": ts,
        }
        if image_path:
            meta["image_path"] = image_path
        if assistant_code:
            meta["assistant_code"] = assistant_code
        if plotly_json_path:
            meta["plotly_json_path"] = plotly_json_path
        # Save to per-session memory collection (isolated)
        self._get_session_memory_collection(session_id).add(
            documents=[combined_text],
            metadatas=[meta],
            ids=[turn_id]
        )

    def get_session_turns(self, session_id: str) -> list:
        """
        Returns all conversation turns for a session in chronological order.
        Each turn is a dict with 'user', 'assistant', and optionally 'image_path', 'assistant_code', 'plotly_json_path'.
        Used to restore chat history (and persisted plots) in the UI.
        """
        try:
            per_results = {"metadatas": []}
            legacy_results = {"metadatas": []}

            try:
                per_collection = self._get_session_memory_collection(session_id, create_if_missing=False)
                if per_collection is not None:
                    per_results = per_collection.get(include=["metadatas"])
            except Exception:
                per_results = {"metadatas": []}

            try:
                legacy_results = self.memory_collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"]
                )
            except Exception:
                legacy_results = {"metadatas": []}

            combined = []
            for bucket in (per_results, legacy_results):
                metas = bucket.get("metadatas") or []
                combined.extend(metas)

            if not combined:
                return []

            # Deduplicate by timestamp + user + assistant + image_path
            seen = set()
            unique = []
            for t in combined:
                key = (t.get("timestamp"), t.get("user"), t.get("assistant"), t.get("image_path"), t.get("assistant_code"), t.get("plotly_json_path"))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(t)

            turns = sorted(unique, key=lambda m: m.get("timestamp", 0))
            return [
                {
                    "user": t["user"],
                    "assistant": t["assistant"],
                    "image_path": t.get("image_path") or None,
                    "assistant_code": t.get("assistant_code") or None,
                    "plotly_json_path": t.get("plotly_json_path") or None,
                }
                for t in turns
            ]
        except Exception:
            return []

    def get_memory(self, session_id: str, query: str, n_results: int = 5) -> str:
        """Retrieves relevant past conversation context for a query."""
        try:
            # Prefer per-session memory, then fall back to legacy if needed
            per_docs = []
            legacy_docs = []

            try:
                per_collection = self._get_session_memory_collection(session_id, create_if_missing=False)
                per_count = per_collection.count() if per_collection is not None else 0
            except Exception:
                per_count = 0

            if per_count > 0:
                per_results = per_collection.query(
                    query_texts=[query],
                    n_results=min(n_results, per_count)
                )
                if per_results and per_results.get("documents") and per_results["documents"]:
                    per_docs = per_results["documents"][0] or []

            # If we don't have enough from per-session, query legacy for back-compat
            remaining = n_results - len(per_docs)
            if remaining > 0:
                try:
                    legacy_count = self.memory_collection.count()
                except Exception:
                    legacy_count = 0
                if legacy_count > 0:
                    legacy_results = self.memory_collection.query(
                        query_texts=[query],
                        n_results=min(remaining, legacy_count),
                        where={"session_id": session_id}
                    )
                    if legacy_results and legacy_results.get("documents") and legacy_results["documents"]:
                        legacy_docs = legacy_results["documents"][0] or []

            docs = []
            for doc in per_docs + legacy_docs:
                if doc and doc not in docs:
                    docs.append(doc)

            context = ""
            if docs:
                context = "\n---\nPast relevant conversation context:\n"
                for doc in docs:
                    context += doc + "\n"
            return context
        except Exception:
            return ""
