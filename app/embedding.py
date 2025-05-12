# app/embedding.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(chunks: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.
    """
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings.tolist()
