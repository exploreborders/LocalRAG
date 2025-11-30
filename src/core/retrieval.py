#!/usr/bin/env python3
"""
Combined retrieval system with database-backed retrieval and RAG pipeline.
"""

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from elasticsearch import Elasticsearch
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langdetect import LangDetectException, detect, detect_langs
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from src.core.embeddings import get_embedding_model
from src.core.knowledge_graph import KnowledgeGraph
from src.database.models import Document, DocumentChunk, SessionLocal
from src.utils.error_handler import (
    DatabaseError,
    ErrorHandler,
    ProcessingError,
    ValidationError,
    error_context,
    handle_errors,
    validate_and_handle,
)

logger = logging.getLogger(__name__)


class DatabaseRetriever:
    """
    Handles document retrieval using vector similarity search in Elasticsearch
    and metadata queries in PostgreSQL.
    """

    def __init__(
        self,
        model_name: str = "embeddinggemma:latest",
        backend: str = "ollama",
        use_batch_processing: bool = False,
        hybrid_alpha: float = 0.7,
    ):
        """
        Initialize the retriever with specified embedding model and knowledge graph.

        Args:
            model_name (str): Name of the embedding model to use
            backend (str): Backend for embeddings ('ollama' or 'sentence-transformers')
            use_batch_processing (bool): Whether to use batch embedding for improved performance
            hybrid_alpha (float): Weight for BM25 in the hybrid search (0=vector only, 1=BM25 only)
        """
        # Input validation
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("model_name must be a non-empty string")

        if not isinstance(hybrid_alpha, (int, float)) or not (
            0.0 <= hybrid_alpha <= 1.0
        ):
            raise ValidationError("hybrid_alpha must be a float between 0.0 and 1.0")

        self.model_name = model_name
        self.backend = backend
        self.use_batch_processing = use_batch_processing
        self.hybrid_alpha = hybrid_alpha

        # Initialize error handler
        self.error_handler = ErrorHandler(__name__)

        try:
            self.model = get_embedding_model(model_name, backend)
        except Exception as e:
            raise ProcessingError(f"Failed to load embedding model '{model_name}': {e}")

        # Batch processing can be added later if needed

        # Initialize database session and knowledge graph
        self.db = SessionLocal()
        self.knowledge_graph = KnowledgeGraph(self.db)

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def retrieve_with_knowledge_graph(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search with knowledge graph expansion.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for tags, categories, etc.

        Returns:
            List of retrieved document chunks with metadata
        """
        # Get expanded context from knowledge graph
        expanded_context = self._build_enhanced_context(query, filters or {})

        # Perform hybrid search with expanded context
        results = self.hybrid_search(
            query=query,
            top_k=top_k,
            expanded_tags=expanded_context.get("expanded_tags", []),
            expanded_categories=expanded_context.get("expanded_categories", []),
            filters=filters,
        )

        return results

    def _build_enhanced_context(
        self, query: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build enhanced search context using knowledge graph relationships.

        Args:
            query: Original search query
            filters: User-provided filters

        Returns:
            Dict with expanded search context
        """
        # Extract tags and categories from filters
        tag_names = filters.get("tags", [])
        category_names = filters.get("categories", [])

        # If no explicit filters, try to infer from query
        if not tag_names and not category_names:
            # Simple keyword extraction (could be enhanced with NLP)
            query_lower = query.lower()
            # This is a simplified approach - in production, use NLP for better extraction
            potential_tags = []
            potential_categories = []

            # Look for common academic/technical terms
            academic_terms = [
                "research",
                "study",
                "analysis",
                "method",
                "theory",
                "model",
            ]
            technical_terms = [
                "algorithm",
                "implementation",
                "system",
                "framework",
                "architecture",
            ]

            for term in academic_terms + technical_terms:
                if term in query_lower:
                    potential_tags.append(term.title())

            if potential_tags:
                tag_names = potential_tags[:3]  # Limit to avoid over-expansion

        # Expand context using knowledge graph
        expanded_context = self.knowledge_graph.expand_query_context(
            tag_names=tag_names, category_names=category_names, context_depth=2
        )

        return expanded_context

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        expanded_tags: Optional[List[str]] = None,
        expanded_categories: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.

        Args:
            query: Search query
            top_k: Number of results to return
            expanded_tags: Additional tags from knowledge graph
            expanded_categories: Additional categories from knowledge graph
            filters: Additional filters

        Returns:
            List of retrieved document chunks with scores
        """
        # Generate query embedding using the appropriate backend
        from src.core.embeddings import create_embeddings

        embeddings_array, _ = create_embeddings(
            [query], model_name=self.model_name, backend=self.backend
        )
        if embeddings_array is None or len(embeddings_array) == 0:
            raise ProcessingError(f"Failed to generate embedding for query: {query}")
        query_embedding = embeddings_array[0]

        # Build Elasticsearch query
        es_query = self._build_es_query(
            query,
            query_embedding,
            expanded_tags or [],
            expanded_categories or [],
            filters or {},
        )

        # Execute search
        try:
            es_client = self._get_es_client()
            response = es_client.search(
                index="chunks",
                body=es_query,
                size=top_k * 2,  # Get more results for reranking
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                score = hit["_score"]

                # Get document metadata
                doc_id = source.get("document_id")
                document = self.db.query(Document).filter(Document.id == doc_id).first()

                if document:
                    result = {
                        "chunk_id": hit["_id"],
                        "content": source.get("content", ""),
                        "score": score,
                        "document_title": document.filename,
                        "document_id": doc_id,
                        "tags": [
                            tag_assignment.tag.name
                            for tag_assignment in document.tags
                            if tag_assignment.tag
                        ],
                        "categories": [
                            cat_assignment.category.name
                            for cat_assignment in document.categories
                            if cat_assignment.category
                        ],
                        "metadata": source.get("metadata", {}),
                        "chapter_title": source.get("metadata", {}).get(
                            "chapter_title"
                        ),
                        "chapter_path": source.get("metadata", {}).get("chapter_path"),
                    }
                    results.append(result)

            # Rerank and return top_k
            return self._rerank_results(results, query, top_k)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _build_es_query(
        self,
        query: str,
        query_embedding: np.ndarray,
        expanded_tags: List[str],
        expanded_categories: List[str],
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build Elasticsearch query with hybrid scoring."""
        # For now, use simple BM25 search until we fix the hybrid scoring
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title", "tags", "categories"],
                    "type": "best_fields",
                }
            }
        }

        # Add filters
        filter_clauses = []

        # Tag filters
        tags_to_search = set(filters.get("tags", []) + expanded_tags)
        if tags_to_search:
            filter_clauses.append({"terms": {"tags": list(tags_to_search)}})

        # Category filters
        categories_to_search = set(filters.get("categories", []) + expanded_categories)
        if categories_to_search:
            filter_clauses.append({"terms": {"categories": list(categories_to_search)}})

        if filter_clauses:
            es_query["query"] = {
                "bool": {"must": es_query["query"], "filter": filter_clauses}
            }

        return es_query

    def _get_es_client(self) -> Elasticsearch:
        """Get Elasticsearch client."""
        from src.database.opensearch_setup import get_elasticsearch_client

        return get_elasticsearch_client()

    def _rerank_results(
        self, results: List[Dict[str, Any]], query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using additional relevance signals.

        This post-processing step enhances the initial Elasticsearch scoring with:
        - Exact phrase matches (20% boost)
        - Tag/category keyword matches (10% per matching term)
        - Future: recency, document authority, user preferences, etc.
        """
        query_lower = query.lower()
        query_terms = set(query.split())

        for result in results:
            original_score = result["score"]

            # Boost for exact query matches in content (case-insensitive)
            if query_lower in result["content"].lower():
                result["score"] *= 1.2

            # Boost for matching tags/categories in query terms
            # This helps when users mention topics that are tagged
            result_tags = set(result.get("tags", []))
            result_categories = set(result.get("categories", []))
            matching_terms = len((result_tags | result_categories) & query_terms)
            if matching_terms > 0:
                result["score"] *= 1.0 + matching_terms * 0.1

            # Log significant score changes for debugging
            score_change = result["score"] / original_score
            if score_change > 1.5:
                logger.debug(
                    f"Reranking boosted result by {score_change:.2f}x: {result['document_title']}"
                )

        # Sort by final score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class RAGPipelineDB:
    """
    Retrieval-Augmented Generation pipeline using database-backed retrieval
    and Ollama LLM for answer generation.
    """

    def __init__(
        self,
        model_name: str = "embeddinggemma:latest",
        backend: str = "ollama",
        llm_model: str = "llama3.2:latest",
        cache_enabled: Optional[bool] = None,
        cache_settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name (str): Embedding model for retrieval
            backend (str): Backend for embeddings ('ollama' or 'sentence-transformers')
            llm_model (str): Ollama model for generation
            cache_enabled (bool): Whether to enable caching (overrides env var)
            cache_settings (dict): Cache configuration settings
        """
        self.retriever = DatabaseRetriever(
            model_name, backend, use_batch_processing=True
        )
        self.llm = OllamaLLM(model=llm_model)

        # Initialize cache if enabled
        if cache_enabled is None:
            cache_enabled = os.getenv("CACHE_ENABLED", "false").lower() == "true"

        if cache_enabled:
            try:
                from cache.redis_cache import RedisCache

                cache_config = cache_settings or {}
                self.cache = RedisCache(
                    host=cache_config.get("host", os.getenv("REDIS_HOST", "localhost")),
                    port=cache_config.get("port", int(os.getenv("REDIS_PORT", 6379))),
                    db=cache_config.get("db", int(os.getenv("REDIS_DB", 0))),
                    password=cache_config.get("password", os.getenv("REDIS_PASSWORD")),
                )
            except ImportError:
                logger.warning("Redis cache not available, disabling caching")
                self.cache = None
        else:
            self.cache = None

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question: Question to answer
            top_k: Number of documents to retrieve
            filters: Optional filters for search
            language: Target language for response
            hybrid_alpha: Weight for BM25 in hybrid search (0=vector only, 1=BM25 only)

        Returns:
            Dict with answer and sources
        """
        # Validate inputs
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("question must be a non-empty string")

        if hybrid_alpha is not None and (
            not isinstance(hybrid_alpha, (int, float))
            or not (0.0 <= hybrid_alpha <= 1.0)
        ):
            raise ValidationError("hybrid_alpha must be a float between 0.0 and 1.0")

        # Set hybrid_alpha if provided
        original_alpha = None
        if hybrid_alpha is not None:
            original_alpha = self.retriever.hybrid_alpha
            self.retriever.hybrid_alpha = hybrid_alpha

        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                question,
                top_k,
                filters,
                language,
                hybrid_alpha or self.retriever.hybrid_alpha,
            )
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info("Cache hit for query")
                    return cached_result

            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve_with_knowledge_graph(
                question, top_k=top_k, filters=filters
            )

            if not retrieved_docs:
                result = {
                    "answer": "I could not find relevant information to answer your question.",
                    "sources": [],
                    "language": language or "en",
                }
            else:
                # Generate answer
                answer = self._generate_answer(question, retrieved_docs, language)

                # Format sources
                sources = self._format_sources(retrieved_docs)

                result = {
                    "answer": answer,
                    "sources": sources,
                    "language": language or self._detect_language(answer),
                    "retrieved_count": len(retrieved_docs),
                }

            # Cache the result
            if self.cache:
                self.cache.set(cache_key, result)

            return result

        finally:
            # Restore original alpha if it was changed
            if original_alpha is not None:
                self.retriever.hybrid_alpha = original_alpha

        if not retrieved_docs:
            result = {
                "answer": "I could not find relevant information to answer your question.",
                "sources": [],
                "language": language or "en",
            }
        else:
            # Generate answer
            answer = self._generate_answer(question, retrieved_docs, language)

            # Format sources
            sources = self._format_sources(retrieved_docs)

            result = {
                "answer": answer,
                "sources": sources,
                "language": language or self._detect_language(answer),
                "retrieved_count": len(retrieved_docs),
            }

        # Cache result
        if self.cache:
            self.cache.set(cache_key, result)

        return result

    def _generate_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        language: Optional[str] = None,
    ) -> str:
        """Generate answer using retrieved documents."""
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents[:3]):  # Limit to top 3 for context length
            context_parts.append(
                f"Document {i + 1}: {doc['content'][:1000]}..."
            )  # Truncate for context length

        context = "\n\n".join(context_parts)

        # Create prompt
        prompt_template = """
        Based on the following documents, please answer the question. If the documents don't contain enough information to fully answer the question, say so clearly.

        Documents:
        {context}

        Question: {question}

        Answer:
        """

        if language:
            prompt_template += f"\n\nPlease provide your answer in {language}."

        prompt = prompt_template.format(context=context, question=question)

        try:
            response = self.llm.invoke(prompt)
            return response.strip() if response else "I could not generate an answer."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "An error occurred while generating the answer."

    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieved documents as sources."""
        sources = []
        for doc in documents:
            source = {
                "title": doc.get("document_title", "Unknown"),
                "document_id": doc.get("document_id"),
                "content_preview": (
                    doc.get("content", "")[:200] + "..."
                    if len(doc.get("content", "")) > 200
                    else doc.get("content", "")
                ),
                "tags": doc.get("tags", []),
                "categories": doc.get("categories", []),
                "score": doc.get("score", 0),
                "chapter_title": doc.get("chapter_title"),
                "chapter_path": doc.get("chapter_path"),
                "chunk_index": doc.get("chunk_index", 0),
            }
            sources.append(source)
        return sources

    def _detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def _generate_cache_key(
        self,
        question: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        language: Optional[str],
        hybrid_alpha: Optional[float] = None,
    ) -> str:
        """Generate a cache key for the query."""
        key_components = [question, str(top_k), str(language)]
        if filters:
            key_components.append(str(sorted(filters.items())))
        if hybrid_alpha is not None:
            key_components.append(str(hybrid_alpha))
        return hashlib.md5("|".join(key_components).encode()).hexdigest()


# Format functions for web interface
def format_results_db(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieval results for display in the web interface.

    Args:
        results: List of retrieval results

    Returns:
        Formatted string for display
    """
    if not results:
        return "No results found."

    lines = []
    for i, result in enumerate(results, 1):
        doc = result.get("document", {})
        score = result.get("score", 0)
        topic_boost = result.get("topic_boost", 0)

        # Get document info
        filename = doc.get("filename", "Unknown")
        content_preview = (
            doc.get("page_content", "")[:100] + "..."
            if len(doc.get("page_content", "")) > 100
            else doc.get("page_content", "")
        )

        lines.append(f"{i}. {filename} (Score: {score:.4f})")
        if topic_boost > 0:
            lines.append(f"   Topic Boost: {topic_boost:.2f}")
        lines.append(f"   Content: {content_preview}")
        lines.append("")

    return "\n".join(lines)


def format_answer_db(answer: str) -> str:
    """
    Format RAG answer for display in the web interface.

    Args:
        answer: Raw answer text from LLM

    Returns:
        Formatted markdown string
    """
    if not answer:
        return "*No answer generated.*"

    # Basic markdown formatting - ensure proper line breaks
    formatted = answer.strip()

    # Convert simple line breaks to proper markdown paragraphs
    # This is a basic formatter - could be enhanced
    paragraphs = [p.strip() for p in formatted.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        formatted = "\n\n".join(paragraphs)
    else:
        # Handle single paragraph with line breaks
        lines = [line.strip() for line in formatted.split("\n") if line.strip()]
        formatted = " ".join(lines)

    return formatted
