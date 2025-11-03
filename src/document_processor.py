#!/usr/bin/env python3
"""
New document processor with database storage and Elasticsearch integration.
"""

import os
import hashlib
import re
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

from .database.models import Document, DocumentChunk, SessionLocal
from .data_loader import load_documents, split_documents
from .embeddings import get_embedding_model, create_embeddings
from .database.opensearch_setup import get_elasticsearch_client

load_dotenv()

class DocumentProcessor:
    """
    Legacy document processor for basic document operations.

    This class provides basic document processing functionality and database queries.
    For new uploads with enhanced structure extraction, use UploadProcessor instead.

    Maintained for backward compatibility and basic operations.
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

    def extract_author_from_docling(self, docling_document) -> Optional[str]:
        """
        Extract author information from a Docling document.

        Tries multiple strategies to find author information:
        1. Document metadata/properties
        2. Text patterns in the document content
        3. Header/footer analysis

        Args:
            docling_document: Docling document object

        Returns:
            str: Author name if found, None otherwise
        """
        # Strategy 1: Check document metadata/properties
        if hasattr(docling_document, 'origin') and docling_document.origin:
            origin_data = docling_document.origin.model_dump()
            # Check for author in document properties
            if 'author' in origin_data and origin_data['author']:
                return origin_data['author'].strip()
            if 'creator' in origin_data and origin_data['creator']:
                return origin_data['creator'].strip()

        # Strategy 2: Search for author patterns in text content
        markdown_content = docling_document.export_to_markdown()
        author = self._extract_author_from_text(markdown_content)

        if author:
            return author

        # Strategy 3: Check document text items for author information
        if hasattr(docling_document, 'texts'):
            for text_item in docling_document.texts:
                if hasattr(text_item, 'text') and text_item.text:
                    author = self._extract_author_from_text(text_item.text)
                    if author:
                        return author

        return None

    def _extract_author_from_text(self, text: str) -> Optional[str]:
        """
        Extract author information from text using pattern matching.

        Args:
            text: Text content to search

        Returns:
            str: Author name if found, None otherwise
        """
        # Common author patterns
        author_patterns = [
            r'Author[:\-]?\s*([^\n\r]{1,100})',
            r'By[:\-]?\s*([^\n\r]{1,100})',
            r'Written by[:\-]?\s*([^\n\r]{1,100})',
            r'Created by[:\-]?\s*([^\n\r]{1,100})',
            r'^([^\n\r]{1,50})$',  # Single line that might be author (check first few lines)
        ]

        lines = text.split('\n')[:10]  # Check first 10 lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in author_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    author_candidate = match.group(1).strip()
                    # Validate author candidate
                    if self._is_valid_author(author_candidate):
                        return author_candidate

        return None

    def _is_valid_author(self, author_candidate: str) -> bool:
        """
        Validate if a string looks like a valid author name.

        Args:
            author_candidate: Potential author name

        Returns:
            bool: True if it looks like a valid author name
        """
        if not author_candidate or len(author_candidate) < 2 or len(author_candidate) > 100:
            return False

        # Exclude common non-author text
        exclude_patterns = [
            r'^\d+$',  # Just numbers
            r'^[^\w\s]+$',  # Just punctuation
            r'(?i)(table|figure|page|chapter|section|volume)',  # Document structure words
            r'(?i)(http|www|\.com|\.org|\.edu)',  # URLs
            r'^.{1,3}$',  # Very short strings
        ]

        for pattern in exclude_patterns:
            if re.search(pattern, author_candidate):
                return False

        # Check for reasonable name-like structure (contains letters, not too many numbers)
        has_letters = bool(re.search(r'[a-zA-Z]', author_candidate))
        number_ratio = len(re.findall(r'\d', author_candidate)) / len(author_candidate)

        return has_letters and number_ratio < 0.5

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
        except LangDetectError:
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

    def preprocess_french_text(self, text: str) -> str:
        """
        Preprocess French text using spaCy for better chunking and embedding.

        Args:
            text (str): Raw French text

        Returns:
            str: Preprocessed text with improved tokenization
        """
        try:
            nlp = spacy.load('fr_core_news_sm')
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
            print(f"French preprocessing failed: {e}, using original text")
            return text

    def preprocess_spanish_text(self, text: str) -> str:
        """
        Preprocess Spanish text using spaCy for better chunking and embedding.

        Args:
            text (str): Raw Spanish text

        Returns:
            str: Preprocessed text with improved tokenization
        """
        try:
            nlp = spacy.load('es_core_news_sm')
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
            print(f"Spanish preprocessing failed: {e}, using original text")
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

    def save_document(self, filename: str, filepath: str, file_hash: str, content_type: Optional[str] = None, detected_language: Optional[str] = None) -> Document:
        """
        Save document metadata to database.

        Args:
            filename (str): Name of the document file
            filepath (str): Full path to the document file
            file_hash (str): MD5 hash of the file for change detection
            content_type (str): File content type/extension
            detected_language (str): Detected language code (ISO 639-1)

        Returns:
            Document: The created or updated document object
        """
        # Check if document already exists
        existing_doc = self.db.query(Document).filter(Document.file_hash == file_hash).first()

        if existing_doc:
            # Update existing document
            existing_doc.filename = filename
            existing_doc.filepath = filepath
            existing_doc.last_modified = datetime.now()
            if content_type:
                existing_doc.content_type = content_type
            if detected_language:
                existing_doc.detected_language = detected_language
            self.db.commit()
            return existing_doc

        # Create new document
        doc = Document(
            filename=filename,
            filepath=filepath,
            file_hash=file_hash,
            content_type=content_type,
            detected_language=detected_language,
        )
        self.db.add(doc)
        self.db.commit()
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
            # Extract caption metadata from chunk metadata
            metadata = chunk.metadata or {}
            chunk_type = metadata.get('chunk_type', 'standard')
            has_captions = metadata.get('has_captions', False)
            caption_text = metadata.get('caption_text')
            caption_line = metadata.get('caption_line')
            context_lines = metadata.get('context_lines')

            chunk_obj = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=chunk.page_content,
                embedding_model=embedding_model
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
                # Note: This uses the older ES API syntax for compatibility
                self.es.delete_by_query(  # type: ignore
                    index="rag_vectors",
                    body={"query": {"term": {"document_id": existing_doc.id}}}  # type: ignore
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
        elif detected_language == 'fr':
            # Apply French-specific preprocessing
            processed_content = self.preprocess_french_text(full_content)
            # Update documents with processed content
            if processed_content != full_content:
                documents = [LangchainDocument(page_content=processed_content, metadata={"source": str(file_path), "language": detected_language})]
        elif detected_language == 'es':
            # Apply Spanish-specific preprocessing
            processed_content = self.preprocess_spanish_text(full_content)
            # Update documents with processed content
            if processed_content != full_content:
                documents = [LangchainDocument(page_content=processed_content, metadata={"source": str(file_path), "language": detected_language})]

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

        # Combine all document content
        full_content = ' '.join([doc.page_content for doc in documents])

        # Save or update document metadata
        if doc is None:
            content_type = file_path.suffix[1:] if file_path.suffix else 'unknown'
            doc = self.save_document(file_path.name, str(file_path), file_hash, content_type, detected_language)
        else:
            # Update existing document
            doc.filename = file_path.name
            doc.filepath = str(file_path)
            doc.detected_language = detected_language
            # No author, content_type, or status fields to set
            doc.last_modified = datetime.now()
            self.db.commit()

        # Store full content
        doc.full_content = full_content
        self.db.commit()

        # Generate embeddings
        embeddings, _ = create_embeddings(chunks, model_name)

        # Save chunks to database
        self.save_chunks(doc.id, chunks, model_name, chunk_size, overlap)  # type: ignore

        # Save embeddings to Elasticsearch
        self.save_embeddings_to_es(chunks, embeddings, model_name, doc.id)  # type: ignore

        # AI enrichment (optional, only if Ollama is available)
        try:
            from .ai_enrichment import AIEnrichmentService
            enrichment_service = AIEnrichmentService()
            enrichment_result = enrichment_service.enrich_document(doc.id)
            if enrichment_result['success']:
                print(f"AI enrichment completed for {file_path.name}")
            else:
                print(f"AI enrichment skipped: {enrichment_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"AI enrichment failed: {e}")

        # Update document last modified time
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

    def process_existing_documents(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", force_reprocess: bool = True, **kwargs):
        """
        Process all existing documents in the database.

        Args:
            model_name (str): Name of the embedding model to use
            force_reprocess (bool): Whether to force reprocessing of already processed documents
            **kwargs: Additional arguments passed to process_document
        """
        # Get all documents
        if force_reprocess:
            docs = self.db.query(Document).all()
        else:
            docs = self.db.query(Document).filter(Document.status == 'pending').all()

        if not docs:
            print("No documents to process")
            return

        print(f"Processing {len(docs)} documents...")

        for doc in docs:
            try:
                print(f"Processing {doc.filename}...")
                self.process_document(doc.filepath, model_name, force_reprocess=force_reprocess, **kwargs)
                # Update status to processed
                doc.status = 'processed'
                self.db.commit()
            except Exception as e:
                print(f"Error processing {doc.filename}: {e}")
                # Mark as failed
                doc.status = 'failed'
                self.db.commit()

        print("Finished processing existing documents")

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

    def get_documents_with_chunk_counts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents with their chunk counts using a single optimized query.
        Eliminates N+1 query problem by joining and counting chunks in one database call.

        Returns:
            list: List of dictionaries containing document metadata with chunk_count
        """
        from sqlalchemy import func

        # Single query with join and count to get documents with chunk counts
        results = self.db.query(
            Document,
            func.count(DocumentChunk.id).label('chunk_count')
        ).outerjoin(DocumentChunk).group_by(Document.id).all()

        return [{
            'id': doc.id,
            'filename': doc.filename,
            'filepath': doc.filepath,
            'content_type': doc.content_type,
            'detected_language': doc.detected_language,
            'status': doc.status,
            'upload_date': doc.upload_date,
            'last_modified': doc.last_modified,
            'chunk_count': chunk_count
        } for doc, chunk_count in results]

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

