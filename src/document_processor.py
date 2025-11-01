#!/usr/bin/env python3
"""
New document processor with database storage and Elasticsearch integration.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from langdetect import detect, LangDetectException as LangDetectError
import spacy
from langchain_core.documents import Document as LangchainDocument

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

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text (str): Text content to analyze

        Returns:
            str: ISO 639-1 language code (e.g., 'en', 'de'), or 'unknown' if detection fails
        """
        if not text or len(text.strip()) < 10:
            return 'unknown'

        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'

    def preprocess_german_text(self, text: str) -> str:
        """
        Preprocess German text using spaCy for better chunking and embedding.

        Args:
            text (str): Raw German text

        Returns:
            str: Preprocessed text with improved tokenization
        """
        try:
            nlp = spacy.load('de_core_news_sm')
            doc = nlp(text)

            # Extract sentences and clean them
            sentences = []
            for sent in doc.sents:
                # Remove extra whitespace and normalize
                clean_sent = ' '.join(sent.text.split())
                if clean_sent:
                    sentences.append(clean_sent)

            return ' '.join(sentences)
        except Exception as e:
            print(f"German preprocessing failed: {e}, using original text")
            return text

    def document_exists(self, file_hash: str) -> bool:
        """
        Check if a document with the given hash already exists in the database.

        Args:
            file_hash (str): MD5 hash of the file

        Returns:
            bool: True if document exists, False otherwise
        """
        return self.db.query(Document).filter(Document.file_hash == file_hash).first() is not None

    def save_document(self, filename: str, filepath: str, file_hash: str, content_type: str, detected_language: Optional[str] = None) -> Document:
        """
        Save document metadata to the database.

        Args:
            filename (str): Name of the file
            filepath (str): Path to the file
            file_hash (str): MD5 hash of the file
            content_type (str): File extension/type
            detected_language (str): Detected language code (ISO 639-1)

        Returns:
            Document: The created Document database object
        """
        doc = Document(
            filename=filename,
            filepath=filepath,
            file_hash=file_hash,
            content_type=content_type,
            detected_language=detected_language,
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

    def process_document(self, filepath: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 1000, overlap: int = 200, force_reprocess: bool = False):
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
        existing_doc = self.db.query(Document).filter(Document.file_hash == file_hash).first()

        if existing_doc and not force_reprocess:
            print(f"Document {file_path.name} already processed")
            return
        elif existing_doc and force_reprocess:
            print(f"Reprocessing existing document {file_path.name}")
            # Delete existing chunks from database
            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == existing_doc.id).delete()
            # Delete existing embeddings from Elasticsearch
            try:
                self.es.delete_by_query(
                    index="rag_vectors",
                    body={"query": {"term": {"document_id": existing_doc.id}}}
                )
            except Exception as e:
                print(f"Warning: Could not delete existing embeddings from Elasticsearch: {e}")
            # Keep the document record but update it
            doc = existing_doc
        else:
            doc = None

        # Load document using Docling
        content_type = file_path.suffix[1:] if file_path.suffix else 'unknown'

        if content_type == 'txt':
            # Use simple text loading for txt files
            with open(str(file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [LangchainDocument(page_content=content, metadata={"source": str(file_path)})]
        else:
            # Use Docling for all document processing (better quality than PyPDF2)
            print(f"Processing {file_path} with Docling")
            try:
                from docling.document_converter import DocumentConverter

                # Use default Docling configuration for best quality
                doc_converter = DocumentConverter()
                result = doc_converter.convert(str(file_path))
                text_content = result.document.export_to_markdown()
                documents = [LangchainDocument(
                    page_content=text_content,
                    metadata={"source": str(file_path), "method": "docling"}
                )]
            except Exception as e:
                print(f"Docling processing failed for {file_path}: {e}")
                # Fallback to PyPDF2 for PDF files if Docling fails
                if content_type == 'pdf':
                    print(f"Trying PyPDF2 fallback for {file_path}")
                    try:
                        import PyPDF2
                        with open(str(file_path), 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            text_content = ""
                            for page in pdf_reader.pages:
                                text_content += page.extract_text() + "\n"
                            documents = [LangchainDocument(
                                page_content=text_content,
                                metadata={"source": str(file_path), "method": "pypdf2_fallback"}
                            )]
                    except Exception as e2:
                        print(f"PyPDF2 fallback also failed: {e2}")
                        documents = []
                else:
                    documents = []

        # Detect language from content
        full_content = ' '.join([doc.page_content for doc in documents])
        detected_language = self.detect_language(full_content)

        # Preprocess text based on detected language
        if detected_language == 'de':
            # Apply German-specific preprocessing
            processed_content = self.preprocess_german_text(full_content)
            # Update documents with processed content
            if processed_content != full_content:
                documents = [LangchainDocument(page_content=processed_content, metadata={"source": str(file_path), "language": detected_language})]

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

        # Save or update document metadata
        if doc is None:
            doc = self.save_document(file_path.name, str(file_path), file_hash, content_type, detected_language)
        else:
            # Update existing document
            doc.filename = file_path.name
            doc.filepath = str(file_path)
            doc.content_type = content_type
            doc.detected_language = detected_language
            doc.last_modified = datetime.now()
            doc.status = 'processing'
            self.db.commit()

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

    def reprocess_all_documents(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 1000, overlap: int = 200, batch_size: int = 5, use_parallel: bool = True, max_workers: int = 4, memory_limit_mb: int = 500):
        """
        Reprocess all documents in the data directory with performance optimizations.
        Uses the same optimized processing as process_existing_documents but forces reprocessing of existing documents.

        Args:
            model_name (str): Embedding model to use
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            batch_size (int): Number of documents to process in each batch
            use_parallel (bool): Whether to use parallel processing for batches
            max_workers (int): Maximum number of worker processes
            memory_limit_mb (int): Memory limit for processing
        """
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from docling.document_converter import DocumentConverter

        # Get all documents from the data directory
        data_dir = Path("data")
        file_paths = []
        for file_path in data_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']:
                file_paths.append(file_path)

        if not file_paths:
            print("No documents found in data directory.")
            return

        print(f"Found {len(file_paths)} documents to reprocess with performance optimizations.")

        # Separate text files from other formats
        text_files = [fp for fp in file_paths if fp.suffix.lower() == '.txt']
        other_files = [fp for fp in file_paths if fp.suffix.lower() != '.txt']

        # Process text files efficiently
        for file_path in text_files:
            try:
                self.process_document(str(file_path), model_name, chunk_size=chunk_size, overlap=overlap, force_reprocess=True)
            except Exception as e:
                print(f"Error reprocessing {file_path}: {e}")

        # Process other formats with optimized batch processing
        if other_files:
            if use_parallel and len(other_files) > batch_size:
                workers = max_workers or min(4, len(other_files) // batch_size + 1)
                self._reprocess_batch_documents_parallel(other_files, model_name, chunk_size, overlap, batch_size, workers, memory_limit_mb)
            else:
                self._reprocess_batch_documents(other_files, model_name, chunk_size, overlap, batch_size, memory_limit_mb)

        print("🎉 Finished reprocessing all documents with performance optimizations.")

    def _reprocess_batch_documents(self, file_paths, model_name, chunk_size, overlap, batch_size, memory_limit_mb):
        """Reprocess documents in batches sequentially."""
        from .embeddings import create_embeddings
        from .data_loader import split_documents
        from docling.document_converter import DocumentConverter

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size} ({len(batch)} documents)")

            # Create converter (reuse for batch)
            doc_converter = DocumentConverter()

            for file_path in batch:
                try:
                    # Check if document exists and delete old data
                    file_hash = self.calculate_file_hash(str(file_path))
                    existing_doc = self.db.query(Document).filter(Document.file_hash == file_hash).first()

                    if existing_doc:
                        # Delete existing chunks and embeddings
                        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == existing_doc.id).delete()
                        try:
                            self.es.delete_by_query(
                                index="rag_vectors",
                                body={"query": {"term": {"document_id": existing_doc.id}}}
                            )
                        except Exception as e:
                            print(f"Warning: Could not delete existing embeddings: {e}")

                    # Process document with Docling
                    result = doc_converter.convert(str(file_path))
                    text_content = result.document.export_to_markdown()

                    from langchain_core.documents import Document as LangchainDocument
                    documents = [LangchainDocument(page_content=text_content, metadata={"source": str(file_path)})]

                    # Detect language
                    full_content = ' '.join([doc.page_content for doc in documents])
                    detected_language = self.detect_language(full_content)

                    # Apply German preprocessing if needed
                    if detected_language == 'de':
                        processed_content = self.preprocess_german_text(full_content)
                        if processed_content != full_content:
                            documents = [LangchainDocument(page_content=processed_content, metadata={"source": str(file_path), "language": detected_language})]

                    # Split documents
                    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

                    # Create/update document record
                    if existing_doc:
                        doc = existing_doc
                        doc.detected_language = detected_language
                        doc.last_modified = datetime.now()
                        doc.status = 'processing'
                    else:
                        doc = self.save_document(file_path.name, str(file_path), file_hash, file_path.suffix[1:], detected_language)

                    # Generate embeddings
                    embeddings, _ = create_embeddings(chunks, model_name)

                    # Save chunks and embeddings
                    self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)
                    self.save_embeddings_to_es(chunks, embeddings, model_name, doc.id)

                    # Update status
                    doc.status = 'processed'
                    self.db.commit()

                    print(f"✅ Reprocessed {file_path.name}")

                except Exception as e:
                    print(f"❌ Error reprocessing {file_path}: {e}")
                    self.db.rollback()

    def _reprocess_batch_documents_parallel(self, file_paths, model_name, chunk_size, overlap, batch_size, max_workers, memory_limit_mb):
        """Reprocess documents in batches using parallel processing."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Convert file paths to serializable data
        serializable_files = []
        for file_path in file_paths:
            # Calculate hash for each file
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            file_hash = hash_md5.hexdigest()

            serializable_files.append({
                'filepath': str(file_path),
                'filename': file_path.name,
                'file_hash': file_hash
            })

        # Split into worker batches
        worker_batches = []
        files_per_worker = max(1, len(serializable_files) // max_workers)
        for i in range(0, len(serializable_files), files_per_worker):
            worker_batches.append(serializable_files[i:i + files_per_worker])

        print(f"Processing {len(worker_batches)} worker batches with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing (pass only serializable data)
            future_to_batch = {
                executor.submit(self._process_reprocess_batch_worker, batch, model_name, chunk_size, overlap): batch
                for batch in worker_batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    results = future.result()
                    # Save results to database (main process handles DB operations)
                    for file_data in results:
                        filepath, file_hash, chunks, embeddings_array, detected_language, filename = file_data

                        # Find or create document record
                        existing_doc = self.db.query(Document).filter(Document.file_hash == file_hash).first()
                        if existing_doc:
                            # Update existing document
                            doc = existing_doc
                            doc.detected_language = detected_language
                            doc.last_modified = datetime.now()
                            doc.status = 'processing'
                            # Delete old chunks
                            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                        else:
                            # Create new document
                            doc = Document(
                                filename=filename,
                                filepath=filepath,
                                file_hash=file_hash,
                                content_type=filepath.split('.')[-1] if '.' in filepath else 'unknown',
                                detected_language=detected_language,
                                status='processing'
                            )
                            self.db.add(doc)
                            self.db.flush()

                        # Save chunks and embeddings
                        self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)
                        self.save_embeddings_to_es(chunks, embeddings_array, model_name, doc.id)

                        # Update status
                        doc.status = 'processed'
                        doc.last_modified = datetime.now()
                        print(f"✅ Reprocessed {filename}: {len(chunks)} chunks created")

                except Exception as e:
                    print(f"❌ Batch failed: {e}")

        # Commit all changes
        self.db.commit()

    @staticmethod
    def _process_reprocess_batch_worker(batch, model_name, chunk_size, overlap):
        """Worker function for parallel reprocessing. Returns serializable data only."""
        from docling.document_converter import DocumentConverter
        from langchain_core.documents import Document as LangchainDocument
        from .embeddings import create_embeddings
        from .data_loader import split_documents

        results = []

        # Create converter for this worker
        doc_converter = DocumentConverter()

        for file_data in batch:
            filepath = file_data['filepath']
            filename = file_data['filename']
            file_hash = file_data['file_hash']

            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                continue

            try:
                # Process document with Docling
                result = doc_converter.convert(filepath)
                text_content = result.document.export_to_markdown()
                documents = [LangchainDocument(page_content=text_content, metadata={"source": filepath})]

                # Detect language (simplified version for worker)
                full_content = ' '.join([doc.page_content for doc in documents])
                try:
                    from langdetect import detect
                    detected_language = detect(full_content[:1000])  # Use first 1000 chars for speed
                except:
                    detected_language = 'unknown'

                # Apply German preprocessing if detected
                if detected_language == 'de':
                    try:
                        import spacy
                        nlp = spacy.load('de_core_news_sm')
                        doc = nlp(full_content[:5000])  # Process first 5000 chars for speed
                        sentences = [sent.text for sent in doc.sents]
                        processed_content = ' '.join(sentences)
                        if processed_content:
                            documents = [LangchainDocument(page_content=processed_content, metadata={"source": filepath, "language": detected_language})]
                    except:
                        pass  # Skip preprocessing if spaCy fails

                # Split and embed
                chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
                if not chunks:
                    continue

                embeddings_array, _ = create_embeddings(chunks, model_name)

                # Return serializable data: (filepath, file_hash, chunks, embeddings_array, detected_language, filename)
                results.append((filepath, file_hash, chunks, embeddings_array, detected_language, filename))

            except Exception as e:
                print(f"❌ Worker failed to process {filename}: {e}")

        return results

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
            'detected_language': doc.detected_language,
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