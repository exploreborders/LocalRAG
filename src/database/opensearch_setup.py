#!/usr/bin/env python3
"""
OpenSearch setup script for RAG system.
Run this after starting OpenSearch.
"""

from elasticsearch import Elasticsearch
import json
import os

def setup_opensearch_indices():
    # Connect to Elasticsearch (using instead of OpenSearch due to Docker issues)
    client = Elasticsearch(
        hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
        basic_auth=(os.getenv("OPENSEARCH_USER", "elastic"), os.getenv("OPENSEARCH_PASSWORD", "changeme")),
        verify_certs=False
    )

    # Document index for full-text search
    document_index = {
        "mappings": {
            "properties": {
                "document_id": {"type": "integer"},
                "filename": {"type": "text"},
                "content": {"type": "text", "analyzer": "standard"},
                "embedding_model": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "created_at": {"type": "date"}
            }
        }
    }

    # Vector index for similarity search
    vector_index = {
        "mappings": {
            "properties": {
                "document_id": {"type": "integer"},
                "chunk_id": {"type": "integer"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768  # Adjust based on model
                },
                "embedding_model": {"type": "keyword"},
                "metadata": {"type": "object"}
            }
        }
    }

    # Create indices
    indices = {
        "rag_documents": document_index,
        "rag_vectors": vector_index
    }

    for index_name, index_config in indices.items():
        if not client.indices.exists(index=index_name):
            response = client.indices.create(index=index_name, body=index_config)
            print(f"Created index {index_name}: {response}")
        else:
            print(f"Index {index_name} already exists")

if __name__ == "__main__":
    setup_opensearch_indices()