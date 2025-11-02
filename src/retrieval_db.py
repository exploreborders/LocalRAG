#!/usr/bin/env python3
"""
Updated retrieval system using Elasticsearch for vector search and PostgreSQL for metadata.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
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

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize the retriever with specified embedding model.

        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = get_embedding_model(model_name)
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

        Args:
            query (str): Query text to embed

        Returns:
            np.ndarray: Query embedding vector
        """
        return self.model.encode([query], convert_to_numpy=True)[0]



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

        response = self.es.search(index="rag_vectors", body=query)
        hits = response['hits']['hits']

        results = []
        for hit in hits:
            source = hit['_source']
            results.append({
                'document_id': source['document_id'],
                'chunk_id': source['chunk_id'],
                'content': source['content'],
                'score': hit['_score'],
                'embedding_model': source['embedding_model'],
                'metadata': source.get('metadata', {})
            })

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

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query with enriched metadata.
        Uses optimized single-query approach to avoid N+1 problems.

        Args:
            query (str): Search query text
            top_k (int): Number of top results to return

        Returns:
            list: List of search results with document metadata
        """
        # Embed the query
        query_embedding = self.embed_query(query)

        # Search for similar vectors
        vector_results = self.search_vectors(query_embedding, top_k)

        # Enrich with document metadata using optimized batch query
        enriched_results = self._enrich_with_batch_metadata(vector_results)

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
                    'content_type': doc.content_type,
                    'detected_language': doc.detected_language,
                    'status': doc.status,
                    'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                    'last_modified': doc.last_modified.isoformat() if doc.last_modified else None
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
                            'status': doc['status'],
                            'detected_language': doc['detected_language'],
                            'upload_date': doc['upload_date']
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
                            'upload_date': doc.upload_date.isoformat() if doc.upload_date else None
                        }
                    })
            else:
                # Fallback to empty document info if not found
                enriched_results.append({
                    **result,
                    'document': {}
                })

        return enriched_results

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

        response = self.es.search(index="rag_documents", body=es_query)
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