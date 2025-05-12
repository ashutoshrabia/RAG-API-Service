# app/vector_store.py
import faiss
import pickle
import numpy as np
import os

# Define paths for storing the FAISS index and documents
INDEX_PATH = "tmp/vector_store.index"
DOCUMENTS_PATH = "tmp/documents.pkl"

def save_faiss_index(index, documents):
    """
    Save the FAISS index and associated documents to disk.
    
    Args:
        index: FAISS index containing document embeddings.
        documents: List of document metadata (e.g., [{"source": "path", "content": "text"}]).
    """
    # Ensure the tmp directory exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    # Save the FAISS index
    faiss.write_index(index, INDEX_PATH)
    
    # Save the documents metadata
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)

def get_faiss_index():
    """
    Load the FAISS index from disk.
    
    Returns:
        FAISS index object.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Please ingest documents first.")
    return faiss.read_index(INDEX_PATH)

def load_documents():
    """
    Load the documents metadata from disk.
    
    Returns:
        List of document metadata.
    """
    if not os.path.exists(DOCUMENTS_PATH):
        raise FileNotFoundError("Documents metadata not found. Please ingest documents first.")
    with open(DOCUMENTS_PATH, "rb") as f:
        return pickle.load(f)

def search_index(index, query_embedding, k=3):
    """
    Search the FAISS index for the top-k most similar documents to the query embedding.
    
    Args:
        index: FAISS index to search.
        query_embedding: Embedding of the query (numpy array).
        k: Number of top results to return.
    
    Returns:
        List of (document, distance) tuples for the top-k results.
    """
    distances, indices = index.search(query_embedding, k)
    documents = load_documents()
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1 and idx < len(documents):
            results.append((documents[idx], distance))
    return results