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
    """
    Handles document processing, chunking, embedding generation, and storage
    in both PostgreSQL database and Elasticsearch.
    """

    def __init__(self):
        """
        Initialize the document processor with database and Elasticsearch connections.
        """
        self.db: Session = SessionLocal()
        self.es = get_elasticsearch_client()

    def __del__(self):
        """
        Clean up database connections when the processor is destroyed.
        """
        self.db.close()

    def calculate_file_hash(self, filepath: str) -> str:
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

    def document_exists(self, file_hash: str) -> bool:
        """
        Check if a document with the given hash already exists in the database.

        Args:
            file_hash (str): MD5 hash of the file

        Returns:
            bool: True if document exists, False otherwise
        """
        return self.db.query(Document).filter(Document.file_hash == file_hash).first() is not None

    def save_document(self, filename: str, filepath: str, file_hash: str, content_type: str) -> Document:
        """
        Save document metadata to the database.

        Args:
            filename (str): Name of the file
            filepath (str): Path to the file
            file_hash (str): MD5 hash of the file
            content_type (str): File extension/type

        Returns:
            Document: The created Document database object
        """
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
        """
        Save document chunks to the database.

        Args:
            document_id (int): ID of the parent document
            chunks (list): List of Document chunk objects
            embedding_model (str): Name of the embedding model used
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
        """
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
        """
        Save document chunks and their embeddings to Elasticsearch.

        Args:
            chunks (list): List of Document chunk objects
            embeddings (list): List of embedding vectors (numpy arrays)
            model_name (str): Name of the embedding model
            document_id (int): ID of the parent document
        """
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

    def process_document(self, filepath: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 1000, overlap: int = 200):
        """
        Process a single document: load, chunk, embed, and save to database and Elasticsearch.

        Args:
            filepath (str): Path to the document file
            model_name (str): Name of the embedding model to use
            chunk_size (int): Maximum size of each text chunk
            overlap (int): Number of characters to overlap between chunks

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_hash = self.calculate_file_hash(str(file_path))
        if self.document_exists(file_hash):
            print(f"Document {file_path.name} already processed")
            return

        # Load and split document
        documents = load_documents(str(file_path))
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

        # Save document metadata
        content_type = file_path.suffix[1:] if file_path.suffix else 'unknown'
        doc = self.save_document(file_path.name, str(file_path), file_hash, content_type)

        # Generate embeddings
        embeddings, _ = create_embeddings(chunks, model_name)

        # Save chunks to database
        self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)

        # Save embeddings to Elasticsearch
        self.save_embeddings_to_es(chunks, embeddings, model_name, doc.id)

        # Update document status
        doc.status = 'processed'
        doc.last_modified = datetime.now()
        self.db.commit()

        print(f"Processed {file_path.name}: {len(chunks)} chunks")

    def process_directory(self, directory: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5", **kwargs):
        """
        Process all supported documents in a directory.

        Args:
            directory (str): Path to the directory containing documents
            model_name (str): Name of the embedding model to use
            **kwargs: Additional arguments passed to process_document
        """
        data_dir = Path(directory)
        for file_path in data_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']:
                try:
                    self.process_document(str(file_path), model_name, **kwargs)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from the database.

        Returns:
            list: List of dictionaries containing document metadata
        """
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
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id (int): ID of the document

        Returns:
            list: List of dictionaries containing chunk information
        """
        chunks = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
        return [{
            'id': chunk.id,
            'chunk_index': chunk.chunk_index,
            'content': chunk.content,
            'embedding_model': chunk.embedding_model
        } for chunk in chunks]

    def process_existing_documents(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 1000, overlap: int = 200):
        """
        Process all existing documents in the database that haven't been processed yet.
        
        Args:
            model_name (str): Embedding model to use
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
        """
        from .embeddings import create_embeddings
        from .data_loader import load_pdf, load_docx, load_pptx, load_xlsx, split_documents
        
        # Get documents that are not processed
        docs = self.db.query(Document).filter(Document.status != 'processed').all()
        
        if not docs:
            print("No unprocessed documents found.")
            return
        
        print(f"Found {len(docs)} documents to process.")
        
        for doc in docs:
            if os.path.exists(doc.filepath) and os.path.isfile(doc.filepath):
                print(f"🔄 Processing {doc.filename}...")
                try:
                    # Load document content based on type
                    if doc.content_type == 'pdf':
                        documents = load_pdf(doc.filepath)
                    elif doc.content_type == 'docx':
                        documents = load_docx(doc.filepath)
                    elif doc.content_type == 'pptx':
                        documents = load_pptx(doc.filepath)
                    elif doc.content_type == 'xlsx':
                        documents = load_xlsx(doc.filepath)
                    elif doc.content_type == 'txt':
                        # For txt, create Document object
                        with open(doc.filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        from langchain_core.documents import Document as LangchainDocument
                        documents = [LangchainDocument(page_content=content, metadata={"source": doc.filepath})]
                    else:
                        print(f"⚠️ Unsupported content type: {doc.content_type}")
                        continue
                    
                    # Split into chunks
                    chunks = split_documents(documents, chunk_size, overlap)
                    
                    if not chunks:
                        print(f"⚠️ No chunks generated for {doc.filename}")
                        continue
                    
                    # Generate embeddings
                    embeddings_array, _ = create_embeddings(chunks, model_name)
                    
                    # Save chunks to database
                    self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)
                    
                    # Save embeddings to Elasticsearch
                    self.save_embeddings_to_es(chunks, embeddings_array, model_name, doc.id)
                    
                    # Update document status
                    doc.status = 'processed'
                    doc.last_modified = datetime.now()
                    self.db.commit()
                    
                    print(f"✅ Processed {doc.filename}: {len(chunks)} chunks created")
                    
                except Exception as e:
                    print(f"❌ Failed to process {doc.filename}: {e}")
                    self.db.rollback()
            else:
                print(f"⚠️ File not found, skipping: {doc.filepath}")
        
        print("🎉 Finished processing existing documents.")