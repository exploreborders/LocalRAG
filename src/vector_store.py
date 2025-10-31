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
    import numpy as np

    dimension = embeddings.shape[1]

    # Use simple L2 index for all cases (most stable)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_faiss_index(index, model_name="all-MiniLM-L6-v2", filename=None):
    """
    Save FAISS index to file.
    """
    if filename is None:
        # Create model-specific filename
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"models/faiss_index_{safe_model_name}.pkl"

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")

def load_faiss_index(model_name="all-MiniLM-L6-v2", filename=None):
    """
    Load FAISS index from file with validation.
    """
    if filename is None:
        # Create model-specific filename
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"models/faiss_index_{safe_model_name}.pkl"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"FAISS index file not found: {filename}. Please process documents with this model first.")

    try:
        index = faiss.read_index(filename)
    except Exception as e:
        raise Exception(f"Failed to load FAISS index from {filename}: {e}")

    # Validate index
    if index.ntotal == 0:
        raise Exception(f"FAISS index {filename} is empty (no vectors added)")

    return index

def search_similar(query_embedding, index, documents, k=3):
    """
    Search for similar documents using FAISS.
    """
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(documents):  # Valid index and within bounds
            results.append({
                'document': documents[idx],
                'distance': distances[0][i]
            })
        else:
            print(f"Warning: Index {idx} out of range for documents list of length {len(documents)}")
    return results

if __name__ == "__main__":
    # Load embeddings and documents
    embeddings, documents = load_embeddings()[:2]

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Save index
    save_faiss_index(index)

    print(f"FAISS index created with {index.ntotal} vectors")

    # Test search with a sample query
    from sentence_transformers import SentenceTransformer
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    query = "What is RAG?"
    query_embedding = model.encode([query])[0]

    results = search_similar(query_embedding, index, documents, k=2)
    print(f"Search results for '{query}':")
    for result in results:
        print(f"Distance: {result['distance']:.4f}")
        print(f"Content: {result['document'].page_content[:200]}...")
        print("---")