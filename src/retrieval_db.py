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

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
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

        # Enrich with document metadata
        enriched_results = []
        for result in vector_results:
            doc_info = self.get_document_info(result['document_id'])
            enriched_results.append({
                **result,
                'document': doc_info
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