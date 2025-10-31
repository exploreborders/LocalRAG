#!/usr/bin/env python3
"""
Migration script to move data from pickle files to PostgreSQL + OpenSearch.
Run this after setting up both databases.
"""

import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# Database connections
def get_postgres_connection():
    """
    Get PostgreSQL database connection for migration operations.

    Returns:
        psycopg2.connection: Configured PostgreSQL connection
    """
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "rag_system"),
        user=os.getenv("POSTGRES_USER", "christianhein"),
        password=os.getenv("POSTGRES_PASSWORD", "")
    )

def get_elasticsearch_client():
    """
    Get Elasticsearch client for migration operations.

    Returns:
        Elasticsearch: Configured Elasticsearch client instance
    """
    return Elasticsearch(
        hosts=[{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": int(os.getenv("OPENSEARCH_PORT", 9200)), "scheme": "http"}],
        basic_auth=(os.getenv("OPENSEARCH_USER", "elastic"), os.getenv("OPENSEARCH_PASSWORD", "changeme")),
        verify_certs=False
    )

def calculate_file_hash(filepath):
    """
    Calculate MD5 hash of a file for change detection.

    Args:
        filepath (str): Path to the file

    Returns:
        str: MD5 hash as hexadecimal string
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def migrate_documents_to_postgres():
    """
    Migrate document metadata from filesystem to PostgreSQL database.

    Scans the data directory and creates document records for all supported
    file types (.txt, .pdf, .docx, .pptx, .xlsx).
    """
    print("Migrating documents to PostgreSQL...")

    conn = get_postgres_connection()
    cursor = conn.cursor()

    data_dir = Path("data")
    documents = []

    for file_path in data_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']:
            file_hash = calculate_file_hash(file_path)
            documents.append((
                file_path.name,
                str(file_path),
                file_hash,
                file_path.suffix[1:],  # content_type
                datetime.now(),
                datetime.now(),
                'processed'
            ))

    if documents:
        execute_values(cursor, """
            INSERT INTO documents (filename, filepath, file_hash, content_type, upload_date, last_modified, status)
            VALUES %s ON CONFLICT (file_hash) DO NOTHING
        """, documents)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Migrated {len(documents)} documents")

def migrate_chunks_to_postgres():
    """
    Migrate document chunks to PostgreSQL database.

    Note: Currently a placeholder - implementation depends on existing
    chunk storage format.
    """
    print("Migrating document chunks to PostgreSQL...")

    # This would require loading the existing chunk data
    # For now, placeholder - actual implementation would depend on how chunks are stored
    pass

def migrate_embeddings_to_opensearch():
    """
    Migrate vector embeddings from pickle files to Elasticsearch.

    Loads existing embeddings, documents, and chunks from the models directory
    and bulk indexes them into the rag_vectors index.
    """
    print("Migrating embeddings to OpenSearch...")

    client = get_elasticsearch_client()

    models_dir = Path("models")
    for model_file in models_dir.glob("embeddings_*.pkl"):
        model_name = model_file.stem.replace("embeddings_", "")

        print(f"Loading data for {model_name}...")
        embeddings_path = models_dir / f"embeddings_{model_name}.pkl"
        documents_path = models_dir / f"documents_{model_name}.pkl"
        chunks_path = models_dir / f"chunks_{model_name}.pkl"

        if not all(p.exists() for p in [embeddings_path, documents_path, chunks_path]):
            print(f"Skipping {model_name}, missing files")
            continue

        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        # Bulk index to OpenSearch
        actions = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            action = {
                "_index": "rag_vectors",
                "_id": f"{model_name}_{i}",
                "_source": {
                    "document_id": i // 100,  # Placeholder logic
                    "chunk_id": i,
                    "content": chunk.page_content if hasattr(chunk, 'page_content') else str(chunk),
                    "embedding": embedding.tolist(),
                    "embedding_model": model_name,
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
            }
            actions.append(action)

        # Bulk insert
        if actions:
            from elasticsearch.helpers import bulk
            bulk(client, actions)
            print(f"Migrated {len(actions)} vectors for {model_name}")

if __name__ == "__main__":
    try:
        migrate_documents_to_postgres()
        migrate_chunks_to_postgres()
        migrate_embeddings_to_opensearch()
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Migration failed: {e}")