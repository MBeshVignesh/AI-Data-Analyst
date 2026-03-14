import os
from langchain_huggingface import HuggingFaceEmbeddings


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_embedding_model():
    """Initialize and return the configured HuggingFace embedding model."""
    # Higher-quality open-source default (free). Override in env if needed.
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    normalize = _env_bool("EMBEDDING_NORMALIZE", True)

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize}

    batch_size_raw = os.getenv("EMBEDDING_BATCH_SIZE")
    if batch_size_raw:
        try:
            encode_kwargs["batch_size"] = max(1, int(batch_size_raw))
        except ValueError:
            pass

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
