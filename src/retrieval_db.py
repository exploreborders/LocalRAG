#!/usr/bin/env python3
"""
Updated retrieval system using Elasticsearch for vector search and PostgreSQL for metadata.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from .database.models import SessionLocal, Document, DocumentChunk
from .embeddings import get_embedding_model

class DatabaseRetriever:
    """
    Handles document retrieval using vector similarity search in Elasticsearch
    and metadata queries in PostgreSQL.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", use_batch_processing: bool = False):
        """
        Initialize the retriever with specified embedding model.

        Args:
            model_name (str): Name of the sentence-transformers model to use
            use_batch_processing (bool): Whether to use batch embedding for improved performance
        """
        self.model_name = model_name
        self.use_batch_processing = use_batch_processing
        self.model = get_embedding_model(model_name)

        # Initialize batch embedding service if enabled
        self.batch_service = None
        if self.use_batch_processing:
            try:
                from .batch_embedding import BatchEmbeddingService
                self.batch_service = BatchEmbeddingService(model_name)
                # Start the batch processing
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task in existing loop
                        asyncio.create_task(self.batch_service.start_processing())
                    else:
                        # This will be a problem - we can't start async processing in sync context
                        # Let's defer this to when it's actually needed
                        pass
                except RuntimeError:
                    # No event loop, defer to when needed
                    pass
                print("✅ Batch embedding service enabled")
            except ImportError as e:
                print(f"⚠️ Batch embedding service not available: {e}")
                self.use_batch_processing = False
            except Exception as e:
                print(f"⚠️ Failed to initialize batch service: {e}")
                self.use_batch_processing = False

        self.es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False
        )
        self.db: Session = SessionLocal()

    def __del__(self):
        """Clean up database connections."""
        self.db.close()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed the query text using the configured model.

        Uses batch processing service if available for improved performance,
        otherwise falls back to direct embedding.

        Args:
            query (str): Query text to embed

        Returns:
            np.ndarray: Query embedding vector
        """
        if self.use_batch_processing and self.batch_service:
            try:
                # Ensure batch processing is started
                if not self.batch_service.is_running:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_running():
                            # We can start it in a new loop
                            asyncio.run(self.batch_service.start_processing())
                        # If loop is running, we can't start async processing
                        # Fall back to direct embedding
                        return self.model.encode([query], convert_to_numpy=True)[0]
                    except RuntimeError:
                        # No event loop, fall back
                        return self.model.encode([query], convert_to_numpy=True)[0]

                return self.batch_service.embed_query_sync(query)
            except Exception as e:
                print(f"⚠️ Batch embedding failed, falling back to direct: {e}")
                # Fall back to direct embedding

        # Direct embedding (original method)
        return self.model.encode([query], convert_to_numpy=True)[0]

    def start_batch_processing(self):
        """
        Start the background batch processing service.
        Should be called once when the retriever is ready to handle queries.
        """
        if self.batch_service and not self.batch_service.is_running:
            import asyncio
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task in existing loop
                    asyncio.create_task(self.batch_service.start_processing())
                else:
                    # Start in new loop
                    asyncio.run(self.batch_service.start_processing())
                print("🚀 Batch processing started")
            except Exception as e:
                print(f"⚠️ Failed to start batch processing: {e}")

    def stop_batch_processing(self):
        """
        Stop the background batch processing service.
        """
        if self.batch_service and self.batch_service.is_running:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.batch_service.stop_processing())
                else:
                    asyncio.run(self.batch_service.stop_processing())
                print("🛑 Batch processing stopped")
            except Exception as e:
                print(f"⚠️ Failed to stop batch processing: {e}")

    def get_batch_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get batch processing performance statistics.

        Returns:
            dict: Performance statistics or None if batch processing not available
        """
        if self.batch_service:
            return self.batch_service.get_stats()
        return None

    def search_vectors(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Elasticsearch using KNN.

        Args:
            query_embedding (np.ndarray): Query embedding vector
            top_k (int): Number of top results to return

        Returns:
            list: List of search results with content and metadata
        """
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": top_k,
                "num_candidates": top_k * 10
            }
        }

        response = self.es.search(index="rag_vectors", knn=query["knn"])  # type: ignore
        hits = response['hits']['hits']

        results = []
        for hit in hits:
            source = hit['_source']
            result = {
                'document_id': source['document_id'],
                'chunk_id': source['chunk_id'],
                'content': source['content'],
                'score': hit['_score'],
                'embedding_model': source['embedding_model'],
                'metadata': source.get('metadata', {})
            }

            # Include chapter-specific fields if present
            if 'content_type' in source and source['content_type'] == 'chapter':
                result.update({
                    'content_type': 'chapter',
                    'chapter_id': source.get('chapter_id'),
                    'chapter_title': source.get('chapter_title'),
                    'chapter_path': source.get('chapter_path'),
                    'section_type': source.get('section_type')
                })
            else:
                result['content_type'] = 'chunk'

            results.append(result)

        return results

    def get_document_info(self, document_id: int) -> Dict[str, Any]:
        """
        Get document metadata from PostgreSQL database.

        Args:
            document_id (int): ID of the document

        Returns:
            dict: Document metadata or empty dict if not found
        """
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if doc:
            return {
                'id': doc.id,
                'filename': doc.filename,
                'filepath': doc.filepath,
                'status': doc.status
            }
        return {}

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query with enriched metadata.
        Uses optimized single-query approach to avoid N+1 problems.

        Args:
            query (str): Search query text
            top_k (int): Number of top results to return
            filters (dict): Optional filters for advanced search
                - tags: List of tag names to filter by
                - categories: List of category names to filter by
                - author: Author name to filter by
                - detected_language: Language code to filter by
                - date_from: ISO date string for minimum upload date
                - date_to: ISO date string for maximum upload date

        Returns:
            list: List of search results with document metadata
        """
        # Embed the query
        query_embedding = self.embed_query(query)

        # Search for similar vectors
        vector_results = self.search_vectors(query_embedding, top_k)

        # Enrich with document metadata using optimized batch query
        enriched_results = self._enrich_with_batch_metadata(vector_results)

        # Apply filters if provided
        if filters:
            enriched_results = self._apply_filters(enriched_results, filters)

        return enriched_results

    def _enrich_with_batch_metadata(self, vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich vector results with document metadata using single batch query and Redis caching.
        Avoids N+1 query problem by fetching all document metadata in one query with caching.

        Args:
            vector_results: List of vector search results

        Returns:
            list: Enriched results with document metadata
        """
        if not vector_results:
            return []

        # Extract unique document IDs
        doc_ids = list(set(result['document_id'] for result in vector_results))

        # Try to get cached metadata first
        doc_map = {}
        uncached_doc_ids = doc_ids.copy()

        cache_available = False
        cache = None
        try:
            from .cache.redis_cache import RedisCache
            cache = RedisCache()
            cache_available = True
            cached_metadata = cache.get_document_metadata(doc_ids)
            if cached_metadata:
                doc_map.update(cached_metadata)
                # Remove cached docs from query list
                uncached_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in cached_metadata]
        except Exception:
            # Cache not available, proceed with database query
            cache_available = False

        # Query database for uncached documents
        if uncached_doc_ids:
            docs = self.db.query(Document).filter(Document.id.in_(uncached_doc_ids)).all()
            for doc in docs:
                # Convert datetime objects to strings for JSON serialization
                metadata = {
                    'id': doc.id,
                    'filename': doc.filename,
                    'filepath': doc.filepath,
                    'detected_language': doc.detected_language,
                    'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                    'last_modified': doc.last_modified.isoformat() if doc.last_modified else None,
                }
                doc_map[doc.id] = metadata

                # Cache the metadata if cache is available
                if cache_available and cache:
                    try:
                        cache.set_document_metadata(doc.id, metadata)
                    except Exception:
                        # Cache failure, continue
                        pass

        # Enrich results with pre-loaded metadata
        enriched_results = []
        for result in vector_results:
            doc = doc_map.get(result['document_id'])
            if doc:
                # Handle both cached dict and database Document object
                if isinstance(doc, dict):
                    # From cache
                    enriched_results.append({
                        **result,
                        'document': {
                            'id': doc['id'],
                            'filename': doc['filename'],
                            'filepath': doc['filepath'],
                            'detected_language': doc['detected_language'],
                            'upload_date': doc['upload_date'],
                            'tags': doc.get('tags', []),
                            'categories': doc.get('categories', [])
                        }
                    })
                else:
                    # From database
                    enriched_results.append({
                        **result,
                        'document': {
                            'id': doc.id,
                            'filename': doc.filename,
                            'filepath': doc.filepath,
                            'status': doc.status,
                            'detected_language': doc.detected_language,
                            'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                            'author': doc.author,
                            'reading_time': doc.reading_time,
                            'custom_fields': doc.custom_fields,
                            'tags': [{'id': tag.id, 'name': tag.name, 'color': tag.color} for tag in doc.tags],
                            'categories': [{'id': cat.id, 'name': cat.name, 'description': cat.description} for cat in doc.categories]
                        }
                    })
            else:
                # Fallback to empty document info if not found
                enriched_results.append({
                    **result,
                    'document': {}
                })

        return enriched_results

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply advanced filters to search results.

        Args:
            results: List of enriched search results
            filters: Dictionary of filter criteria

        Returns:
            Filtered list of results
        """
        filtered_results = []

        for result in results:
            doc = result.get('document', {})
            include_result = True

            # Filter by tags
            if 'tags' in filters and filters['tags']:
                doc_tags = [tag['name'] for tag in doc.get('tags', [])]
                if not any(tag in doc_tags for tag in filters['tags']):
                    include_result = False

            # Filter by categories
            if 'categories' in filters and filters['categories']:
                doc_categories = [cat['name'] for cat in doc.get('categories', [])]
                if not any(cat in doc_categories for cat in filters['categories']):
                    include_result = False

            # Filter by author
            if 'author' in filters and filters['author']:
                if doc.get('author') != filters['author']:
                    include_result = False

            # Filter by detected language
            if 'detected_language' in filters and filters['detected_language']:
                if doc.get('detected_language') != filters['detected_language']:
                    include_result = False

            # Filter by date range
            if 'date_from' in filters and filters['date_from']:
                upload_date = doc.get('upload_date')
                if upload_date and upload_date < filters['date_from']:
                    include_result = False

            if 'date_to' in filters and filters['date_to']:
                upload_date = doc.get('upload_date')
                if upload_date and upload_date > filters['date_to']:
                    include_result = False

            if include_result:
                filtered_results.append(result)

        return filtered_results

    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents by text content in Elasticsearch.

        Args:
            query (str): Text query to search for
            top_k (int): Number of top results to return

        Returns:
            list: List of search results with document content and metadata
        """
        es_query = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }

        response = self.es.search(index="rag_documents", body=es_query)  # type: ignore
        hits = response['hits']['hits']

        results = []
        for hit in hits:
            source = hit['_source']
            results.append({
                'document_id': source['document_id'],
                'filename': source['filename'],
                'content': source['content'],
                'score': hit['_score']
            })

        return results

    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = 0.7, text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text matching.

        Args:
            query (str): Search query text
            top_k (int): Number of top results to return
            vector_weight (float): Weight for vector similarity scores
            text_weight (float): Weight for text matching scores

        Returns:
            list: List of combined search results sorted by relevance
        """
        vector_results = self.retrieve(query, top_k * 2)
        text_results = self.search_text(query, top_k * 2)

        # Combine and rerank results
        combined = {}

        # Add vector results
        for result in vector_results:
            key = f"{result['document_id']}_{result['chunk_id']}"
            combined[key] = {
                **result,
                'combined_score': result['score'] * vector_weight
            }

        # Add text results
        for result in text_results:
            key = f"{result['document_id']}_text"
            if key in combined:
                combined[key]['combined_score'] += result['score'] * text_weight
            else:
                combined[key] = {
                    **result,
                    'combined_score': result['score'] * text_weight
                }

        # Sort by combined score and return top_k
        sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_results[:top_k]