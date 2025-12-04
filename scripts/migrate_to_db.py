#!/usr/bin/env python3
"""
Migration script to move data from pickle files to PostgreSQL + OpenSearch.
Run this after setting up both databases.
"""

import hashlib
import os
import pickle
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from psycopg2.extras import execute_values

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
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def get_elasticsearch_client():
    """
    Get Elasticsearch client for migration operations.

    Returns:
        Elasticsearch: Configured Elasticsearch client instance
    """
    return Elasticsearch(
        hosts=[
            {
                "host": os.getenv("OPENSEARCH_HOST", "localhost"),
                "port": int(os.getenv("OPENSEARCH_PORT", 9200)),
                "scheme": "http",
            }
        ],
        basic_auth=(
            os.getenv("OPENSEARCH_USER", "elastic"),
            os.getenv("OPENSEARCH_PASSWORD", "changeme"),
        ),
        verify_certs=False,
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
    file types (.txt, .pdf, .docx, .pptx, .xlsx). Documents are initially
    marked as 'pending' and will be processed to create chunks.
    """
    print("Migrating documents to PostgreSQL...")

    conn = get_postgres_connection()
    cursor = conn.cursor()

    data_dir = Path("data")
    documents = []

    for file_path in data_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [
            ".txt",
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
        ]:
            file_hash = calculate_file_hash(file_path)
            documents.append(
                (
                    file_path.name,
                    str(file_path),
                    file_hash,
                    file_path.suffix[1:],  # content_type
                    datetime.now(),
                    datetime.now(),
                    "pending",  # Start as pending, will be processed later
                )
            )

    if documents:
        execute_values(
            cursor,
            """
            INSERT INTO documents (filename, filepath, file_hash, content_type, upload_date, last_modified, status)
            VALUES %s ON CONFLICT (file_hash) DO NOTHING
        """,
            documents,
        )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Migrated {len(documents)} documents (marked as pending)")


def migrate_chunks_to_postgres():
    """
    Process documents and create chunks in PostgreSQL database.

    Runs the document processor to handle all pending documents.
    """
    print("Processing documents and creating chunks...")

    import os
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, "..")

    try:
        # Run the document processor
        cmd = [
            "python",
            "-c",
            "from src.core.document_manager import DocumentProcessor; "
            "p = DocumentProcessor(); "
            "p.process_existing_documents()",
        ]

        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.path.join(project_dir, "src")},
        )

        if result.returncode == 0:
            print("Document processing completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"Error during document processing: {result.stderr}")
            raise Exception(f"Document processing failed: {result.stderr}")

    except Exception as e:
        print(f"Error during document processing: {e}")
        raise


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

        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        with open(chunks_path, "rb") as f:
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
                    "content": chunk.page_content
                    if hasattr(chunk, "page_content")
                    else str(chunk),
                    "embedding": embedding.tolist(),
                    "embedding_model": model_name,
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
                },
            }
            actions.append(action)

        # Bulk insert
        if actions:
            from elasticsearch.helpers import bulk

            bulk(client, actions)
            print(f"Migrated {len(actions)} vectors for {model_name}")


if __name__ == "__main__":
    print("Starting database migration...")
    print("This will:")
    print("1. Create document records for files in data/ directory")
    print("2. Process documents to create chunks and embeddings")
    print("3. Store everything in PostgreSQL and Elasticsearch")
    print()

    try:
        print("Step 1: Migrating document metadata...")
        migrate_documents_to_postgres()

        print("\nStep 2: Processing documents and creating chunks...")
        migrate_chunks_to_postgres()

        print("\nStep 3: Migrating legacy embeddings (if any)...")
        migrate_embeddings_to_opensearch()

        print("\n🎉 Migration completed successfully!")
        print("All documents have been processed and chunks/embeddings created.")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("You may need to check your database connections and try again.")
