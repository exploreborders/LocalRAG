import faiss
import numpy as np
try:
    from .embeddings import load_embeddings
except ImportError:
    from embeddings import load_embeddings
import pickle
import os

def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings.astype('float32'))
    return index

def save_faiss_index(index, filename="models/faiss_index.pkl"):
    """
    Save FAISS index to file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")

def load_faiss_index(filename="models/faiss_index.pkl"):
    """
    Load FAISS index from file.
    """
    return faiss.read_index(filename)

def search_similar(query_embedding, index, documents, k=3):
    """
    Search for similar documents using FAISS.
    """
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Valid index
            results.append({
                'document': documents[idx],
                'distance': distances[0][i]
            })
    return results

if __name__ == "__main__":
    # Load embeddings and documents
    embeddings, documents = load_embeddings()

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Save index
    save_faiss_index(index)

    print(f"FAISS index created with {index.ntotal} vectors")

    # Test search with a sample query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "What is RAG?"
    query_embedding = model.encode([query])[0]

    results = search_similar(query_embedding, index, documents, k=2)
    print(f"Search results for '{query}':")
    for result in results:
        print(f"Distance: {result['distance']:.4f}")
        print(f"Content: {result['document'].page_content[:200]}...")
        print("---")