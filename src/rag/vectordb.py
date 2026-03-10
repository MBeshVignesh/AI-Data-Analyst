import os
import chromadb
from chromadb.config import Settings
from .embeddings import get_embedding_model
from typing import List, Dict, Any, Optional

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
        # Also clear any memory associated with this session
        self.memory_collection.delete(where={"session_id": session_id})

    def add_memory(
        self,
        session_id: str,
        user_input: str,
        assistant_output: str,
        image_path: Optional[str] = None,
    ):
        """Adds a conversation turn to a session's history with a timestamp.
        If image_path is provided (e.g. a saved plot), it is stored so the UI can restore it on refresh.
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
        self.memory_collection.add(
            documents=[combined_text],
            metadatas=[meta],
            ids=[turn_id]
        )

    def get_session_turns(self, session_id: str) -> list:
        """
        Returns all conversation turns for a session in chronological order.
        Each turn is a dict with 'user', 'assistant', and optionally 'image_path'.
        Used to restore chat history (and persisted plots) in the UI.
        """
        try:
            count = self.memory_collection.count()
            if count == 0:
                return []
            all_results = self.memory_collection.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )
            if not all_results or not all_results["ids"]:
                return []
            turns = sorted(
                all_results["metadatas"],
                key=lambda m: m.get("timestamp", 0)
            )
            return [
                {
                    "user": t["user"],
                    "assistant": t["assistant"],
                    "image_path": t.get("image_path") or None,
                }
                for t in turns
            ]
        except Exception:
            return []

    def get_memory(self, session_id: str, query: str, n_results: int = 5) -> str:
        """Retrieves relevant past conversation context for a query."""
        try:
            count = self.memory_collection.count()
            if count == 0:
                return ""
            results = self.memory_collection.query(
                query_texts=[query],
                n_results=min(n_results, count),
                where={"session_id": session_id}
            )
            context = ""
            if results and results["documents"] and len(results["documents"]) > 0 and results["documents"][0]:
                context = "\n---\nPast relevant conversation context:\n"
                for doc in results["documents"][0]:
                    context += doc + "\n"
            return context
        except Exception:
            return ""
