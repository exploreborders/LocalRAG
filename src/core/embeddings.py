import numpy as np
from sentence_transformers import SentenceTransformer


def create_embeddings(
    documents,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    batch_size=32,
    show_progress=True,
):
    """
    Create embeddings for document chunks using nomic-embed-text-v1.5.

    Args:
        documents (list): List of document objects with page_content attribute, or list of strings
        model_name (str): Name of the sentence-transformers model to use (only nomic-embed-text-v1.5 supported)
        batch_size (int): Batch size for encoding
        show_progress (bool): Whether to show progress bar

    Returns:
        tuple: (embeddings_array, model) where embeddings is numpy array
    """
    # Only support nomic-embed-text-v1.5
    if model_name != "nomic-ai/nomic-embed-text-v1.5":
        print(
            f"Warning: Only nomic-embed-text-v1.5 is supported. Using nomic-ai/nomic-embed-text-v1.5 instead of {model_name}"
        )
        model_name = "nomic-ai/nomic-embed-text-v1.5"

    # Load model with trust_remote_code for nomic models
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)

    # Handle both list of objects with page_content and list of strings
    if documents and hasattr(documents[0], "page_content"):
        texts = [doc.page_content for doc in documents]
    else:
        texts = documents

    # Encode documents
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings, model


def get_embedding_model(model_name):
    """
    Get a sentence transformer model instance.

    Args:
        model_name (str): Name of the model to load

    Returns:
        SentenceTransformer: Loaded model instance
    """
    # Use trust_remote_code for nomic models
    if "nomic" in model_name:
        return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
    else:
        return SentenceTransformer(model_name, device="cpu")


def get_available_models():
    """
    Get list of available sentence-transformers models that can be loaded.

    Returns:
        list: List containing only nomic-embed-text-v1.5
    """
    # Only support nomic-embed-text-v1.5 - assume it's available
    # Model will be downloaded on first use
    return ["nomic-ai/nomic-embed-text-v1.5"]
