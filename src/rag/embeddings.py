from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    # Using a fast, local embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
