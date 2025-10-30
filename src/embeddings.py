from sentence_transformers import SentenceTransformer
from data_loader import load_documents, split_documents
import numpy as np
import pickle
import os

def create_embeddings(documents, model_name="all-MiniLM-L6-v2"):
    """
    Create embeddings for document chunks using sentence-transformers.
    """
    model = SentenceTransformer(model_name)
    texts = [doc.page_content for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

def save_embeddings(embeddings, documents, filename="models/embeddings.pkl"):
    """
    Save embeddings and documents to a pickle file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'documents': documents}, f)
    print(f"Embeddings saved to {filename}")

def load_embeddings(filename="models/embeddings.pkl"):
    """
    Load embeddings and documents from a pickle file.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['documents']

if __name__ == "__main__":
    # Load and split documents
    docs = load_documents()
    chunks = split_documents(docs)

    # Create embeddings
    embeddings, model = create_embeddings(chunks)

    # Save embeddings
    save_embeddings(embeddings, chunks)

    print(f"Created embeddings for {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")