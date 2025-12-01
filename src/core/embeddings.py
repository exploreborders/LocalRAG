import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Install with: pip install ollama")


def create_embeddings(
    documents,
    model_name="embeddinggemma:latest",
    backend="ollama",
    batch_size=32,
    show_progress=True,
    use_cache=True,
    cache_backend=None,
):
    """
    Create embeddings for document chunks using specified backend with optional caching.

    Args:
        documents (list): List of document objects with page_content attribute, or list of strings
        model_name (str): Name of the model to use
        backend (str): Backend to use ('ollama' or 'sentence-transformers')
        batch_size (int): Batch size for encoding
        show_progress (bool): Whether to show progress bar
        use_cache (bool): Whether to use embedding cache
        cache_backend: Cache backend instance (e.g., RedisCache)

    Returns:
        tuple: (embeddings_array, model_info) where embeddings is numpy array
    """
    # Validate backend
    if backend not in ["ollama", "sentence-transformers"]:
        logger.warning(f"Unsupported backend: {backend}. Defaulting to 'ollama'")
        backend = "ollama"

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
            cache_key = _get_embedding_cache_key(text, f"{backend}:{model_name}")
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

    # Encode uncached documents based on backend
    if uncached_texts:
        if backend == "ollama":
            embeddings = _encode_with_ollama(uncached_texts, model_name, show_progress)
        elif backend == "sentence-transformers":
            embeddings = _encode_with_sentence_transformers(
                uncached_texts, model_name, batch_size, show_progress
            )

        # Cache the new embeddings
        if use_cache and cache_backend and embeddings is not None:
            for i, embedding in enumerate(embeddings):
                cache_key = _get_embedding_cache_key(uncached_texts[i], f"{backend}:{model_name}")
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

    return embeddings, f"{backend}:{model_name}"


def _encode_with_ollama(
    texts: List[str], model_name: str, show_progress: bool = True
) -> np.ndarray:
    """Encode texts using Ollama embeddings API."""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama is not available. Install with: pip install ollama")

    embeddings = []
    for i, text in enumerate(texts):
        if show_progress:
            logger.info(f"Encoding text {i + 1}/{len(texts)} with Ollama")
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            embeddings.append(response["embedding"])
        except Exception as e:
            logger.error(f"Failed to encode text with Ollama: {e}")
            # Return zero embedding as fallback
            embeddings.append([0.0] * 768)  # Assuming 768 dimensions

    return np.array(embeddings)


def _encode_with_sentence_transformers(
    texts: List[str], model_name: str, batch_size: int = 32, show_progress: bool = True
) -> np.ndarray:
    """Encode texts using SentenceTransformers."""
    # Load model
    if "nomic" in model_name:
        model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name, device="cpu")

    # Encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def get_embedding_model(model_name, backend="sentence-transformers"):
    """
    Get an embedding model instance.

    Args:
        model_name (str): Name of the model to load
        backend (str): Backend to use ('ollama' or 'sentence-transformers')

    Returns:
        Model instance or model info
    """
    if backend == "ollama":
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not available")
        return f"ollama:{model_name}"
    elif backend == "sentence-transformers":
        if "nomic" in model_name:
            return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        else:
            return SentenceTransformer(model_name, device="cpu")
    else:
        logger.warning(f"Unsupported backend: {backend}. Defaulting to 'ollama'")
        if not OLLAMA_AVAILABLE:
            raise ValueError(f"Ollama not available and unsupported backend: {backend}")
        return f"ollama:{model_name}"


def _get_embedding_cache_key(text: str, model_key: str) -> str:
    """
    Generate a cache key for embedding based on text content and model.

    Args:
        text: Text to be embedded
        model_key: Model identifier (includes backend)

    Returns:
        str: Cache key
    """
    # Create hash of text + model combination for caching
    content = f"{model_key}:{text}".encode("utf-8")
    return f"embedding:{hashlib.sha256(content).hexdigest()}"


def get_available_models(backend="ollama"):
    """
    Get list of available models for the specified backend.

    Args:
        backend (str): Backend to get models for

    Returns:
        list: List of available model names
    """
    if backend == "ollama":
        if OLLAMA_AVAILABLE:
            try:
                # Get installed models from Ollama
                result = ollama.list()
                return [model["name"] for model in result["models"]]
            except Exception as e:
                logger.warning(f"Failed to get Ollama models: {e}")
                return ["embeddinggemma:latest"]  # fallback
        else:
            return ["embeddinggemma:latest"]  # fallback
    elif backend == "sentence-transformers":
        # Return some common sentence-transformers models
        return [
            "nomic-ai/nomic-embed-text-v1.5",
            "google/embeddinggemma-300m",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
    else:
        return []


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

    def get(self, text: str, model_key: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for
            model_key: Model key used for embedding (includes backend)

        Returns:
            Cached embedding array or None
        """
        if not self.backend:
            return None

        cache_key = _get_embedding_cache_key(text, model_key)
        cached_data = self.backend.get(cache_key)

        if cached_data and isinstance(cached_data, list):
            try:
                return np.array(cached_data)
            except (ValueError, TypeError):
                logger.warning(f"Invalid cached embedding data for key {cache_key}")
                return None

        return None

    def set(self, text: str, model_key: str, embedding: np.ndarray) -> bool:
        """
        Cache embedding for text.

        Args:
            text: Text that was embedded
            model_key: Model key used for embedding
            embedding: Embedding array to cache

        Returns:
            True if cached successfully
        """
        if not self.backend:
            return False

        cache_key = _get_embedding_cache_key(text, model_key)
        try:
            return self.backend.set(cache_key, embedding.tolist())
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
            return False

    def get_batch(
        self, texts: List[str], model_key: str
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts
            model_key: Model key

        Returns:
            Tuple of (cached_embeddings, uncached_indices)
        """
        if not self.backend:
            return [None] * len(texts), list(range(len(texts)))

        cached_embeddings = []
        uncached_indices = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model_key)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                cached_embeddings.append(None)
                uncached_indices.append(i)

        return cached_embeddings, uncached_indices

    def set_batch(self, texts: List[str], model_key: str, embeddings: np.ndarray) -> int:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            model_key: Model key
            embeddings: Array of embeddings

        Returns:
            Number of embeddings successfully cached
        """
        if not self.backend:
            return 0

        cached_count = 0
        for i, text in enumerate(texts):
            if self.set(text, model_key, embeddings[i]):
                cached_count += 1

        return cached_count
