from sentence_transformers import SentenceTransformer
try:
    from .data_loader import load_documents, split_documents
except ImportError:
    from data_loader import load_documents, split_documents
import numpy as np
import pickle
import os
import hashlib

def create_embeddings(documents, model_name="all-MiniLM-L6-v2", batch_size=32, show_progress=True):
    """
    Create embeddings for document chunks using sentence-transformers with optimizations.
    """
    import torch

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model with optimizations
    model = SentenceTransformer(model_name, device=device)

    # Optimize model for inference
    model.eval()
    if hasattr(model, 'module'):
        model.module.eval()

    texts = [doc.page_content for doc in documents]

    # Use optimized encoding parameters
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for better similarity search
    )

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings, model

def get_documents_hash(documents):
    """
    Generate a hash of the document contents for change detection.
    """
    content = ''.join(doc.page_content for doc in documents)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def save_embeddings(embeddings, documents, model_name="all-MiniLM-L6-v2", filename=None):
    """
    Save embeddings and documents to a pickle file.
    """
    if filename is None:
        # Create model-specific filename
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"models/embeddings_{safe_model_name}.pkl"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate hash of document contents
    documents_hash = get_documents_hash(documents)

    with open(filename, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'documents': documents,
            'model_name': model_name,
            'documents_hash': documents_hash
        }, f)
    print(f"Embeddings saved to {filename}")

def load_embeddings(model_name="all-MiniLM-L6-v2", filename=None):
    """
    Load embeddings and documents from a pickle file with validation.
    """
    if filename is None:
        # Create model-specific filename
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"models/embeddings_{safe_model_name}.pkl"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Embeddings file not found: {filename}. Please process documents with this model first.")

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load embeddings file {filename}: {e}")

    # Validate data structure
    required_keys = ['embeddings', 'documents', 'model_name']
    for key in required_keys:
        if key not in data:
            raise Exception(f"Invalid embeddings file {filename}: missing '{key}' key")

    # Validate embeddings format
    embeddings = data['embeddings']
    if not isinstance(embeddings, np.ndarray):
        raise Exception(f"Invalid embeddings format in {filename}: expected numpy array")

    if embeddings.ndim != 2:
        raise Exception(f"Invalid embeddings shape in {filename}: expected 2D array, got {embeddings.ndim}D")

    # Validate documents
    documents = data['documents']
    if not isinstance(documents, list):
        raise Exception(f"Invalid documents format in {filename}: expected list")

    if len(documents) != embeddings.shape[0]:
        raise Exception(f"Document count ({len(documents)}) doesn't match embedding count ({embeddings.shape[0]}) in {filename}")

    return embeddings, documents, data.get('documents_hash', None)

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