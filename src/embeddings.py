import numpy as np
from sentence_transformers import SentenceTransformer


def create_embeddings(documents, model_name="all-mpnet-base-v2", batch_size=32, show_progress=True):
    """
    Create embeddings for document chunks using sentence-transformers.

    Args:
        documents (list): List of document objects with page_content attribute
        model_name (str): Name of the sentence-transformers model to use
        batch_size (int): Batch size for encoding
        show_progress (bool): Whether to show progress bar

    Returns:
        tuple: (embeddings_array, model) where embeddings is numpy array
    """
    # Load model
    model = SentenceTransformer(model_name, device='cpu')

    texts = [doc.page_content for doc in documents]

    # Encode documents
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
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
    return SentenceTransformer(model_name, device='cpu')


def get_available_models():
    """
    Get list of available sentence-transformers models that can be loaded.

    Returns:
        list: List of available model names
    """
    candidate_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-mpnet-base-v2"
    ]

    available_models = []

    for model_name in candidate_models:
        try:
            model = SentenceTransformer(model_name, device='cpu')
            available_models.append(model_name)
            del model
        except Exception:
            continue

    if not available_models:
        available_models = ["all-MiniLM-L6-v2"]

    return available_models