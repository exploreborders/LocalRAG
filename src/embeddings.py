from sentence_transformers import SentenceTransformer
try:
    from .data_loader import load_documents, split_documents
except ImportError:
    from data_loader import load_documents, split_documents
import numpy as np
import pickle
import os
import hashlib
import sys

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

def get_embedding_model(model_name):
    """Get a sentence transformer model"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device='cpu')

def get_available_models():
    """
    Get list of available sentence-transformers models.
    """
    # Common embedding models to check
    candidate_models = [
        "all-MiniLM-L6-v2",  # Most common and well-tested
        "all-mpnet-base-v2",  # Good performance
        "paraphrase-multilingual-mpnet-base-v2"  # Multilingual support
    ]

    available_models = []

    try:
        from sentence_transformers import SentenceTransformer
        for model_name in candidate_models:
            try:
                model = SentenceTransformer(model_name, device='cpu')
                available_models.append(model_name)
                del model  # Clean up
            except Exception:
                continue
    except ImportError:
        print("sentence-transformers not available")
        return ["all-MiniLM-L6-v2"]

    # Always include at least the default model
    if not available_models:
        available_models = ["all-MiniLM-L6-v2"]

    return available_models

def create_embeddings_for_all_models(documents, batch_size=32, show_progress=True):
    """
    Create embeddings for all available models.
    Returns a dictionary of {model_name: (embeddings, model)} pairs.
    """
    available_models = get_available_models()
    results = {}

    print(f"Processing embeddings for {len(available_models)} models: {available_models}")

    for model_name in available_models:
        try:
            print(f"\n🔄 Processing {model_name}...")
            embeddings, model = create_embeddings(
                documents,
                model_name=model_name,
                batch_size=batch_size,
                show_progress=show_progress
            )
            results[model_name] = (embeddings, model)
            print(f"✅ {model_name}: {embeddings.shape} embeddings created")
        except Exception as e:
            print(f"❌ Failed to process {model_name}: {e}")
            continue

    return results

def save_embeddings_for_all_models(embeddings_dict, documents):
    """
    Save embeddings for all models in the dictionary.
    """
    for model_name, (embeddings, _) in embeddings_dict.items():
        try:
            save_embeddings(embeddings, documents, model_name)
        except Exception as e:
            print(f"❌ Failed to save embeddings for {model_name}: {e}")

def process_all_models(documents=None, batch_size=32, show_progress=True):
    """
    Process embeddings for all available models and save them.
    If documents is None, loads and splits documents automatically.
    """
    if documents is None:
        # Load and split documents
        docs = load_documents()
        documents = split_documents(docs)
        print(f"Loaded {len(docs)} documents, split into {len(documents)} chunks")

    # Create embeddings for all models
    embeddings_dict = create_embeddings_for_all_models(
        documents,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Save all embeddings
    save_embeddings_for_all_models(embeddings_dict, documents)

    print(f"\n🎉 Successfully processed {len(embeddings_dict)} models!")
    return embeddings_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process embeddings for documents')
    parser.add_argument('--all', action='store_true', help='Process all available models')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Specific model to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')

    args = parser.parse_args()

    if args.all:
        # Process all available models
        print("🚀 Processing embeddings for all available models...")
        results = process_all_models(batch_size=args.batch_size)
        print(f"✅ Completed processing {len(results)} models")
    else:
        # Process single model (legacy behavior)
        print(f"🔄 Processing embeddings for {args.model}...")

        # Load and split documents
        docs = load_documents()
        chunks = split_documents(docs)
        print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks")

        # Create embeddings
        embeddings, model = create_embeddings(chunks, model_name=args.model, batch_size=args.batch_size)

        # Save embeddings
        save_embeddings(embeddings, chunks, args.model)

        print(f"✅ Created embeddings for {len(chunks)} chunks using {args.model}")
        print(f"Embedding shape: {embeddings.shape}")