#!/usr/bin/env python3
"""
New document processor with database storage and Elasticsearch integration.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from .database.models import Document, DocumentChunk, ProcessingJob, SessionLocal
from .data_loader import load_documents, split_documents
from .embeddings import get_embedding_model, create_embeddings
from .database.opensearch_setup import get_elasticsearch_client

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.db: Session = SessionLocal()
        self.es = get_elasticsearch_client()

    def __del__(self):
        self.db.close()

    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def document_exists(self, file_hash: str) -> bool:
        """Check if document already exists in database."""
        return self.db.query(Document).filter(Document.file_hash == file_hash).first() is not None

    def save_document(self, filename: str, filepath: str, file_hash: str, content_type: str) -> Document:
        """Save document metadata to database."""
        doc = Document(
            filename=filename,
            filepath=filepath,
            file_hash=file_hash,
            content_type=content_type,
            status='processing'
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def save_chunks(self, document_id: int, chunks: List, embedding_model: str, chunk_size: int, overlap: int):
        """Save document chunks to database."""
        for i, chunk in enumerate(chunks):
            chunk_obj = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=chunk.page_content,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap=overlap
            )
            self.db.add(chunk_obj)
        self.db.commit()

    def save_embeddings_to_es(self, chunks: List, embeddings: List, model_name: str, document_id: int):
        """Save embeddings to Elasticsearch."""
        actions = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            action = {
                "_index": "rag_vectors",
                "_id": f"{model_name}_{document_id}_{i}",
                "_source": {
                    "document_id": document_id,
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "embedding": embedding.tolist(),
                    "embedding_model": model_name,
                    "metadata": chunk.metadata
                }
            }
            actions.append(action)

        if actions:
            from elasticsearch.helpers import bulk
            bulk(self.es, actions)

    def process_document(self, filepath: str, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, overlap: int = 200):
        """Process a single document: load, chunk, embed, save to database and ES."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        file_hash = self.calculate_file_hash(str(filepath))
        if self.document_exists(file_hash):
            print(f"Document {filepath.name} already processed")
            return

        # Load and split document
        documents = load_documents(str(filepath))
        chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)

        # Save document metadata
        content_type = filepath.suffix[1:] if filepath.suffix else 'unknown'
        doc = self.save_document(filepath.name, str(filepath), file_hash, content_type)

        # Generate embeddings
        model = get_embedding_model(model_name)
        embeddings = create_embeddings(chunks, model)

        # Save chunks to database
        self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)

        # Save embeddings to Elasticsearch
        self.save_embeddings_to_es(chunks, embeddings, model_name, doc.id)

        # Update document status
        doc.status = 'processed'
        doc.last_modified = datetime.now()
        self.db.commit()

        print(f"Processed {filepath.name}: {len(chunks)} chunks")

    def process_directory(self, directory: str, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        """Process all documents in a directory."""
        data_dir = Path(directory)
        for file_path in data_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']:
                try:
                    self.process_document(str(file_path), model_name, **kwargs)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from database."""
        docs = self.db.query(Document).all()
        return [{
            'id': doc.id,
            'filename': doc.filename,
            'filepath': doc.filepath,
            'status': doc.status,
            'upload_date': doc.upload_date,
            'last_modified': doc.last_modified
        } for doc in docs]

    def get_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get chunks for a document."""
        chunks = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
        return [{
            'id': chunk.id,
            'chunk_index': chunk.chunk_index,
            'content': chunk.content,
            'embedding_model': chunk.embedding_model
        } for chunk in chunks]