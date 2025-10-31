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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = get_embedding_model(model_name)
        self.es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False
        )
        self.db: Session = SessionLocal()

    def __del__(self):
        self.db.close()

    def embed_query(self, query: str) -> np.ndarray:
        """Embed the query using the sentence transformer model."""
        return self.model.encode([query], convert_to_numpy=True)[0]

    def search_vectors(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Elasticsearch."""
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
        """Get document metadata from PostgreSQL."""
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
        """Retrieve relevant documents based on query."""
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
        """Search documents by text content in Elasticsearch."""
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
        """Perform hybrid search combining vector and text search."""
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