#!/usr/bin/env python3
"""
New document processor with database storage and Elasticsearch integration.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
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

    def save_embeddings_to_es(self, chunks: List, embeddings: np.ndarray, model_name: str, document_id: int):
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
                    "embedding_model": model_name
                }
            }
            actions.append(action)

        if actions:
            from elasticsearch.helpers import bulk
            success, failed = bulk(self.es, actions, stats_only=False, raise_on_error=False)
            if failed:
                print(f"Failed to index {failed} embeddings")
                # Print first error
                for action in actions[:1]:
                    try:
                        self.es.index(index="rag_vectors", id=action["_id"], document=action["_source"])
                    except Exception as e:
                        print(f"Sample indexing error: {e}")
                        break

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

        # Load document using Docling
        content_type = file_path.suffix[1:] if file_path.suffix else 'unknown'

        if content_type == 'txt':
            # Use simple text loading for txt files
            with open(str(file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain_core.documents import Document as LangchainDocument
            documents = [LangchainDocument(page_content=content, metadata={"source": str(file_path)})]
        else:
            # For now, use basic text extraction as Docling has API compatibility issues
            # TODO: Re-enable Docling once API stabilizes
            print(f"Using basic text extraction for {file_path} (Docling temporarily disabled)")
            try:
                with open(str(file_path), 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain_core.documents import Document as LangchainDocument
                documents = [LangchainDocument(
                    page_content=content,
                    metadata={"source": str(file_path), "method": "basic_text"}
                )]
            except UnicodeDecodeError:
                # If UTF-8 fails, try with different encoding or skip binary files
                print(f"Skipping binary file {file_path} (not text-based)")
                documents = []
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                documents = []

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

        # Save document metadata
        doc = self.save_document(file_path.name, str(file_path), file_hash, content_type)

        # Generate embeddings
        embeddings, _ = create_embeddings(chunks, model_name)

        # Save chunks to database
        self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)  # type: ignore

        # Save embeddings to Elasticsearch
        self.save_embeddings_to_es(chunks, embeddings, model_name, doc.id)  # type: ignore

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
            'content_type': doc.content_type,
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

    def process_existing_documents(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 1000, overlap: int = 200, batch_size: int = 5, use_parallel: bool = True, max_workers: int = 4, memory_limit_mb: int = 500):
        """
        Process all existing documents in the database that haven't been processed yet.
        Optimized with batch processing, converter reuse, and parallel processing for maximum performance.

        Args:
            model_name (str): Embedding model to use
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            batch_size (int): Number of documents to process in each batch
            use_parallel (bool): Whether to use parallel processing for batches
            max_workers (int): Maximum number of worker processes (defaults to CPU count)
        """
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        # Get documents that are not processed
        docs = self.db.query(Document).filter(Document.status != 'processed').all()

        if not docs:
            print("No unprocessed documents found.")
            return

        print(f"Found {len(docs)} documents to process.")

        # Separate text files from other formats for different processing
        text_docs = [doc for doc in docs if doc.content_type == 'txt']
        other_docs = [doc for doc in docs if doc.content_type != 'txt']

        # Process text files (simple and fast)
        self._process_text_documents(text_docs, model_name, chunk_size, overlap, memory_limit_mb)

        # Process other formats with optimized batch processing
        if use_parallel and len(other_docs) > batch_size:
            workers = max_workers or min(4, len(other_docs) // batch_size + 1)
            self._process_batch_documents_parallel(other_docs, model_name, chunk_size, overlap, batch_size, workers, memory_limit_mb)
        else:
            self._process_batch_documents(other_docs, model_name, chunk_size, overlap, batch_size, memory_limit_mb)

        print("🎉 Finished processing existing documents.")

    def _process_text_documents(self, text_docs, model_name, chunk_size, overlap, memory_limit_mb=500):
        """Process text documents efficiently."""
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from langchain_core.documents import Document as LangchainDocument

        for doc in text_docs:
            if not (os.path.exists(doc.filepath) and os.path.isfile(doc.filepath)):
                print(f"⚠️ File not found, skipping: {doc.filepath}")
                continue

            print(f"🔄 Processing text file {doc.filename}...")
            try:
                # Simple text loading
                with open(doc.filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [LangchainDocument(page_content=content, metadata={"source": doc.filepath})]

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

    def _process_batch_documents(self, other_docs, model_name, chunk_size, overlap, batch_size, memory_limit_mb=500):
        """Process non-text documents using optimized batch processing."""
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from langchain_core.documents import Document as LangchainDocument

        if not other_docs:
            return

        # Create converter with default options due to docling 2.60.0 backend bug
        # TODO: Re-enable custom pipeline options when docling backend issue is fixed
        doc_converter = DocumentConverter()

        # Process documents in batches with memory monitoring
        for i in range(0, len(other_docs), batch_size):
            batch = other_docs[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(other_docs) + batch_size - 1)//batch_size} ({len(batch)} documents)...")

            # Memory check (basic implementation)
            try:
                import psutil
                memory_mb = psutil.virtual_memory().used / 1024 / 1024
                if memory_mb > memory_limit_mb:
                    print(f"⚠️ Memory usage high ({memory_mb:.1f}MB), processing smaller batch...")
                    # Reduce batch size if memory is high
                    current_batch_size = max(1, batch_size // 2)
                    batch = batch[:current_batch_size]
            except ImportError:
                # psutil not available, skip memory check
                pass

            # Filter out missing files
            valid_batch = [doc for doc in batch if os.path.exists(doc.filepath) and os.path.isfile(doc.filepath)]
            if len(valid_batch) != len(batch):
                missing = len(batch) - len(valid_batch)
                print(f"⚠️ Skipped {missing} missing files in this batch")

            if not valid_batch:
                continue

            # Batch convert documents (significant performance boost)
            try:
                file_paths = [doc.filepath for doc in valid_batch]
                results = doc_converter.convert_all(file_paths)

                # Process each result
                for doc, result in zip(valid_batch, results):
                    try:
                        text_content = result.document.export_to_markdown()
                        documents = [LangchainDocument(
                            page_content=text_content,
                            metadata={
                                "source": doc.filepath,
                                "docling_metadata": result.document.origin.model_dump() if result.document.origin else {}
                            }
                        )]

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

                        print(f"✅ Processed {doc.filename}: {len(chunks)} chunks created")

                    except Exception as e:
                        print(f"❌ Failed to process {doc.filename}: {e}")

                # Commit batch
                self.db.commit()

            except Exception as e:
                print(f"❌ Failed to process batch: {e}")
                self.db.rollback()

    def _process_batch_documents_parallel(self, other_docs, model_name, chunk_size, overlap, batch_size, max_workers, memory_limit_mb=500):
        """Process documents using parallel batch processing for maximum performance."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if not other_docs:
            return

        print(f"🔄 Processing {len(other_docs)} documents with {max_workers} parallel workers...")

        # Convert documents to serializable data before multiprocessing
        serializable_docs = []
        for doc in other_docs:
            serializable_docs.append({
                'id': doc.id,
                'filepath': doc.filepath,
                'filename': doc.filename
            })

        # Split serializable documents into worker batches
        worker_batches = []
        docs_per_worker = max(1, len(serializable_docs) // max_workers)
        for i in range(0, len(serializable_docs), docs_per_worker):
            worker_batches.append(serializable_docs[i:i + docs_per_worker])

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing (pass only serializable data)
            future_to_batch = {
                executor.submit(self._process_document_batch_worker, batch, model_name, chunk_size, overlap): batch
                for batch in worker_batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    results = future.result()
                    # Save results to database (main process handles DB operations)
                    for doc_data in results:
                        doc_id, chunks, embeddings_array, filename = doc_data
                        self.save_chunks(doc_id, chunks, model_name, chunk_size, overlap)
                        self.save_embeddings_to_es(chunks, embeddings_array, model_name, doc_id)

                        # Update document status
                        doc = self.db.query(Document).filter(Document.id == doc_id).first()
                        if doc:
                            doc.status = 'processed'
                            doc.last_modified = datetime.now()
                            print(f"✅ Processed {filename}: {len(chunks)} chunks created")
                except Exception as e:
                    print(f"❌ Failed to process batch: {e}")

        # Commit all changes
        self.db.commit()

    @staticmethod
    def _process_document_batch_worker(batch, model_name, chunk_size, overlap):
        """Worker function for parallel document processing. Returns serializable data only."""
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from langchain_core.documents import Document as LangchainDocument

        results = []

        # Create converter with default options due to docling 2.60.0 backend bug
        # TODO: Re-enable custom pipeline options when docling backend issue is fixed
        doc_converter = DocumentConverter()

        for doc_data in batch:
            filepath = doc_data['filepath']
            filename = doc_data['filename']
            doc_id = doc_data['id']

            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                continue

            try:
                # Convert document
                result = doc_converter.convert(filepath)
                text_content = result.document.export_to_markdown()

                documents = [LangchainDocument(
                    page_content=text_content,
                    metadata={
                        "source": filepath,
                        "docling_metadata": result.document.origin.model_dump() if result.document.origin else {}
                    }
                )]

                # Split into chunks
                chunks = split_documents(documents, chunk_size, overlap)

                if not chunks:
                    continue

                # Generate embeddings
                embeddings_array, _ = create_embeddings(chunks, model_name)

                # Return serializable data: (doc_id, chunks, embeddings_array, filename)
                results.append((doc_id, chunks, embeddings_array, filename))

            except Exception as e:
                print(f"❌ Worker failed to process {filename}: {e}")

        return results