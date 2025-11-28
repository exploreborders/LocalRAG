import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def create_embeddings(
    documents,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    batch_size=32,
    show_progress=True,
    use_cache=True,
    cache_backend=None,
):
    """
    Create embeddings for document chunks using nomic-embed-text-v1.5 with optional caching.

    Args:
        documents (list): List of document objects with page_content attribute, or list of strings
        model_name (str): Name of the sentence-transformers model to use (only nomic-embed-text-v1.5 supported)
        batch_size (int): Batch size for encoding
        show_progress (bool): Whether to show progress bar
        use_cache (bool): Whether to use embedding cache
        cache_backend: Cache backend instance (e.g., RedisCache)

    Returns:
        tuple: (embeddings_array, model) where embeddings is numpy array
    """
    # Only support nomic-embed-text-v1.5
    if model_name != "nomic-ai/nomic-embed-text-v1.5":
        logger.warning(
            f"Only nomic-embed-text-v1.5 is supported. Using nomic-ai/nomic-embed-text-v1.5 instead of {model_name}"
        )
        model_name = "nomic-ai/nomic-embed-text-v1.5"

    # Load model with trust_remote_code for nomic models
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)

    # Handle both list of objects with page_content and list of strings
    if documents and hasattr(documents[0], "page_content"):
        texts = [doc.page_content for doc in documents]
    else:
        texts = documents

    # Initialize variables
    embeddings = None

    # Try to get embeddings from cache first
    cached_embeddings = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    if use_cache and cache_backend:
        for i, text in enumerate(texts):
            cache_key = _get_embedding_cache_key(text, model_name)
            cached_data = cache_backend.get(cache_key)
            if cached_data and isinstance(cached_data, list):
                try:
                    cached_embeddings[i] = np.array(cached_data)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid cached embedding for text at index {i}")
                    cached_embeddings[i] = None
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts

    # Encode uncached documents
    if uncached_texts:
        embeddings = model.encode(
            uncached_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Cache the new embeddings
        if use_cache and cache_backend:
            for i, embedding in enumerate(embeddings):
                cache_key = _get_embedding_cache_key(uncached_texts[i], model_name)
                try:
                    cache_backend.set(cache_key, embedding.tolist())
                except Exception as e:
                    logger.warning(
                        f"Failed to cache embedding for index {uncached_indices[i]}: {e}"
                    )

    # Merge cached and new embeddings
    if embeddings is not None and any(cached_embeddings):
        final_embeddings = np.zeros((len(texts), embeddings.shape[1]))
        new_idx = 0
        for i in range(len(texts)):
            if cached_embeddings[i] is not None:
                final_embeddings[i] = cached_embeddings[i]
            else:
                final_embeddings[i] = embeddings[new_idx]
                new_idx += 1
        embeddings = final_embeddings
    elif embeddings is None and any(cached_embeddings):
        # All embeddings were cached
        embeddings = np.array([emb for emb in cached_embeddings if emb is not None])

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


def _get_embedding_cache_key(text: str, model_name: str) -> str:
    """
    Generate a cache key for embedding based on text content and model.

    Args:
        text: Text to be embedded
        model_name: Name of the embedding model

    Returns:
        str: Cache key
    """
    # Create hash of text + model combination
    content = f"{model_name}:{text}".encode("utf-8")
    return f"embedding:{hashlib.md5(content).hexdigest()}"


def get_available_models():
    """
    Get list of available sentence-transformers models that can be loaded.

    Returns:
        list: List containing only nomic-embed-text-v1.5
    """
    # Only support nomic-embed-text-v1.5 - assume it's available
    # Model will be downloaded on first use
    return ["nomic-ai/nomic-embed-text-v1.5"]
    """
    Generate a cache key for embedding based on text content and model.

    Args:
        text: Text to be embedded
        model_name: Name of the embedding model

    Returns:
        str: Cache key
    """
    # Create hash of text + model combination
    content = f"{model_name}:{text}".encode("utf-8")
    return f"embedding:{hashlib.md5(content).hexdigest()}"


class EmbeddingCache:
    """
    Cache manager for embeddings with multiple backend support.
    """

    def __init__(self, backend=None, ttl_hours: int = 168):  # 1 week default
        """
        Initialize embedding cache.

        Args:
            backend: Cache backend (RedisCache, etc.)
            ttl_hours: Time to live in hours for cached embeddings
        """
        self.backend = backend
        self.ttl_hours = ttl_hours

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for
            model_name: Model name used for embedding

        Returns:
            Cached embedding array or None
        """
        if not self.backend:
            return None

        cache_key = _get_embedding_cache_key(text, model_name)
        cached_data = self.backend.get(cache_key)

        if cached_data and isinstance(cached_data, list):
            try:
                return np.array(cached_data)
            except (ValueError, TypeError):
                logger.warning(f"Invalid cached embedding data for key {cache_key}")
                return None

        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray) -> bool:
        """
        Cache embedding for text.

        Args:
            text: Text that was embedded
            model_name: Model name used for embedding
            embedding: Embedding array to cache

        Returns:
            True if cached successfully
        """
        if not self.backend:
            return False

        cache_key = _get_embedding_cache_key(text, model_name)
        try:
            return self.backend.set(cache_key, embedding.tolist())
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
            return False

    def get_batch(
        self, texts: List[str], model_name: str
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts
            model_name: Model name

        Returns:
            Tuple of (cached_embeddings, uncached_indices)
        """
        if not self.backend:
            return [None] * len(texts), list(range(len(texts)))

        cached_embeddings = []
        uncached_indices = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model_name)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                cached_embeddings.append(None)
                uncached_indices.append(i)

        return cached_embeddings, uncached_indices

    def set_batch(self, texts: List[str], model_name: str, embeddings: np.ndarray) -> int:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            model_name: Model name
            embeddings: Array of embeddings

        Returns:
            Number of embeddings successfully cached
        """
        if not self.backend:
            return 0

        cached_count = 0
        for i, text in enumerate(texts):
            if self.set(text, model_name, embeddings[i]):
                cached_count += 1

        return cached_count
