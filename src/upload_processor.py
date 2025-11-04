#!/usr/bin/env python3
"""
Enhanced document upload processor with integrated structure extraction and chapter-aware processing.

This module provides the UploadProcessor class which handles:
- Integrated Docling document structure extraction during upload
- Chapter-aware chunking with hierarchical metadata
- Parallel processing for multiple files
- Progress tracking and error handling
- Automatic language detection and preprocessing

The UploadProcessor replaces the older DocumentProcessor for upload operations,
providing better structure awareness and performance.
"""

import os
import hashlib
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time

from sqlalchemy.orm import Session
from langdetect import detect, LangDetectException
import spacy

from database.models import Document, DocumentChunk, DocumentChapter, SessionLocal
from data_loader import split_documents
from embeddings import get_embedding_model, create_embeddings
from database.opensearch_setup import get_elasticsearch_client

class UploadProcessor:
    """
    Enhanced document upload processor with integrated Docling processing,
    chapter extraction, and parallel processing capabilities.

    This class extends the basic document processing with:
    - Immediate structure extraction during upload
    - Chapter-aware chunking and metadata
    - Parallel processing for multiple files
    - Progress tracking and error handling
    - Memory management for large uploads
    """

    def __init__(self, progress_callback: Optional[Callable[[str, float, str], None]] = None):
        """
        Initialize the upload processor.

        Args:
            progress_callback: Optional callback function for progress updates
                Signature: (file_name, progress_percent, status_message)
        """
        self.progress_callback = progress_callback
        self.db: Session = SessionLocal()
        self.es = get_elasticsearch_client()

        # Initialize embedding model
        self.embedding_model = get_embedding_model("nomic-ai/nomic-embed-text-v1.5")

        # Load spaCy models for language processing
        self.nlp_models = {}
        self._load_spacy_models()

    def __del__(self):
        """Clean up database connections."""
        self.db.close()

    def _load_spacy_models(self):
        """Load spaCy language models for text preprocessing."""
        language_models = {
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'sv': 'sv_core_news_sm',
            'pl': 'pl_core_news_sm',
            'zh': 'zh_core_web_sm',
            'ja': 'ja_core_news_sm',
            'ko': 'ko_core_news_sm'
        }

        for lang, model_name in language_models.items():
            try:
                self.nlp_models[lang] = spacy.load(model_name)
            except OSError:
                # Model not available, skip
                continue

    def calculate_file_hash(self, filepath: str) -> str:
        """
        Calculate MD5 hash of a file for change detection.

        Args:
            filepath: Path to the file

        Returns:
            MD5 hash as hexadecimal string
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
            text: Text content to analyze

        Returns:
            ISO 639-1 language code or 'unknown' if detection fails
        """
        if not text or len(text.strip()) < 10:
            return 'unknown'

        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'

    def preprocess_text(self, text: str, language: str) -> str:
        """
        Preprocess text based on detected language using spaCy.

        Args:
            text: Raw text content
            language: Detected language code

        Returns:
            Preprocessed text with improved tokenization
        """
        if language not in self.nlp_models:
            return text

        try:
            nlp = self.nlp_models[language]
            doc = nlp(text)

            # Extract sentences and clean them
            sentences = []
            for sent in doc.sents:
                clean_sent = ' '.join(sent.text.split())
                if clean_sent:
                    sentences.append(clean_sent)

            return ' '.join(sentences)
        except Exception:
            return text

    def extract_document_structure(self, docling_document) -> Dict[str, Any]:
        """
        Extract document structure including chapters, sections, and metadata using Docling.

        Args:
            docling_document: Docling document object

        Returns:
            Dictionary containing extracted structure information
        """
        structure = {
            'chapters': [],
            'toc': [],
            'full_content': '',
            'metadata': {}
        }

        try:
            # Extract full content
            structure['full_content'] = docling_document.export_to_markdown()

            # Extract table of contents if available
            if hasattr(docling_document, 'structure') and docling_document.structure:
                toc_items = []
                for item in docling_document.structure:
                    if hasattr(item, 'level') and hasattr(item, 'text'):
                        toc_items.append({
                            'level': item.level,
                            'text': item.text,
                            'page': getattr(item, 'page', None)
                        })
                structure['toc'] = toc_items

            # Extract chapters/sections from document body
            chapters = []
            current_chapter = None
            current_content = []

            # Process document items to identify chapters
            if hasattr(docling_document, 'body') and docling_document.body:
                for item in docling_document.body:
                    if hasattr(item, 'label') and item.label in ['title', 'section_header']:
                        # Save previous chapter if exists
                        if current_chapter and current_content:
                            current_chapter['content'] = '\n'.join(current_content)
                            chapters.append(current_chapter)

                        # Start new chapter
                        raw_title = item.text if hasattr(item, 'text') else str(item)
                        clean_title = self._clean_chapter_title(raw_title)
                        current_chapter = {
                            'title': clean_title,
                            'level': getattr(item, 'level', 1),
                            'path': f"{len(chapters) + 1}"
                        }
                        current_content = []
                    elif current_chapter and hasattr(item, 'text'):
                        current_content.append(item.text)

                # Save final chapter
                if current_chapter and current_content:
                    current_chapter['content'] = '\n'.join(current_content)
                    chapters.append(current_chapter)

            # Fallback: Extract chapters from markdown content using regex
            if not chapters:
                chapters = self._extract_chapters_from_markdown(structure['full_content'])

            structure['chapters'] = chapters

            # Extract metadata
            if hasattr(docling_document, 'origin') and docling_document.origin:
                origin_data = docling_document.origin.model_dump()
                structure['metadata'] = origin_data

        except Exception as e:
            print(f"Warning: Structure extraction failed: {e}")
            # Fallback to basic content extraction
            try:
                structure['full_content'] = docling_document.export_to_markdown()
                # Try to extract chapters from markdown even in fallback
                structure['chapters'] = self._extract_chapters_from_markdown(structure['full_content'])
            except Exception:
                structure['full_content'] = str(docling_document)

        return structure

    def _extract_chapters_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract chapters from markdown content using regex patterns.

        Args:
            markdown_content: Markdown content as string

        Returns:
            List of chapter dictionaries
        """
        chapters = []
        lines = markdown_content.split('\n')

        current_chapter = None
        current_content = []
        chapter_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for headers (# ## ###)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous chapter if exists
                if current_chapter and current_content:
                    current_chapter['content'] = '\n'.join(current_content)
                    chapters.append(current_chapter)

                # Start new chapter
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Clean up the title - remove any code blocks, images, or excessive content
                # Limit title to 200 characters to fit in VARCHAR(255) with safety margin
                title = self._clean_chapter_title(title)

                chapter_index += 1

                current_chapter = {
                    'title': title,
                    'level': level,
                    'path': str(chapter_index)
                }
                current_content = []
            elif current_chapter:
                # Add content to current chapter
                current_content.append(line)

        # Save final chapter
        if current_chapter and current_content:
            current_chapter['content'] = '\n'.join(current_content)
            chapters.append(current_chapter)

        return chapters

    def _clean_chapter_title(self, title: str) -> str:
        """
        Clean and truncate chapter titles to fit database constraints.

        Args:
            title: Raw chapter title

        Returns:
            Cleaned and truncated title
        """
        # Remove markdown formatting
        title = re.sub(r'\*\*([^*]+)\*\*', r'\1', title)  # Bold
        title = re.sub(r'\*([^*]+)\*', r'\1', title)      # Italic
        title = re.sub(r'`([^`]+)`', r'\1', title)        # Code

        # Remove image and link references
        title = re.sub(r'!\[.*?\]\(.*?\)', '', title)
        title = re.sub(r'\[.*?\]\(.*?\)', '', title)

        # Remove excessive whitespace
        title = ' '.join(title.split())

        # Truncate to 200 characters to fit VARCHAR(255) safely
        if len(title) > 200:
            title = title[:197] + '...'

        return title.strip()

    def extract_author(self, docling_document) -> Optional[str]:
        """
        Extract author information from Docling document.

        Args:
            docling_document: Docling document object

        Returns:
            Author name if found, None otherwise
        """
        # Try metadata extraction
        if hasattr(docling_document, 'origin') and docling_document.origin:
            origin_data = docling_document.origin.model_dump()
            if 'author' in origin_data and origin_data['author']:
                return origin_data['author'].strip()
            if 'creator' in origin_data and origin_data['creator']:
                return origin_data['creator'].strip()

        # Try text pattern matching
        markdown_content = docling_document.export_to_markdown()
        return self._extract_author_from_text(markdown_content)

    def _extract_author_from_text(self, text: str) -> Optional[str]:
        """
        Extract author from text using pattern matching.

        Args:
            text: Text content to search

        Returns:
            Author name if found, None otherwise
        """
        import re

        author_patterns = [
            r'Author[:\-]?\s*([^\n\r]{1,100})',
            r'By[:\-]?\s*([^\n\r]{1,100})',
            r'Written by[:\-]?\s*([^\n\r]{1,100})',
            r'Created by[:\-]?\s*([^\n\r]{1,100})'
        ]

        lines = text.split('\n')[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in author_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    author_candidate = match.group(1).strip()
                    if self._is_valid_author(author_candidate):
                        return author_candidate

        return None

    def _is_valid_author(self, author_candidate: str) -> bool:
        """
        Validate if a string looks like a valid author name.

        Args:
            author_candidate: Potential author name

        Returns:
            True if it looks valid, False otherwise
        """
        if not author_candidate or len(author_candidate) < 2 or len(author_candidate) > 100:
            return False

        exclude_patterns = [
            r'^\d+$',  # Just numbers
            r'^[^\w\s]+$',  # Just punctuation
            r'(?i)(table|figure|page|chapter|section|volume)',  # Document structure
            r'(?i)(http|www|\.com|\.org|\.edu)',  # URLs
            r'^.{1,3}$',  # Very short strings
        ]

        for pattern in exclude_patterns:
            if re.search(pattern, author_candidate):
                return False

        # Check for reasonable name-like structure
        has_letters = bool(re.search(r'[a-zA-Z]', author_candidate))
        number_ratio = len(re.findall(r'\d', author_candidate)) / len(author_candidate)

        return has_letters and number_ratio < 0.5

    def process_single_file(self, file_path: str, filename: str, file_hash: str, force_enrichment: bool = False) -> Dict[str, Any]:
        """
        Process a single file with full structure extraction and chapter creation.

        Args:
            file_path: Path to the file
            filename: Original filename
            file_hash: MD5 hash of the file
            force_enrichment: Whether to force re-enrichment of already enriched documents

        Returns:
            Dictionary with processing results and metadata
        """
        result = {
            'success': False,
            'filename': filename,
            'file_hash': file_hash,
            'document_id': None,
            'chunks_created': 0,
            'chapters_created': 0,
            'error': None
        }

        try:
            # Update progress
            if self.progress_callback:
                self.progress_callback(filename, 10, "Extracting document structure...")

            # Process with Docling
            from docling.document_converter import DocumentConverter
            doc_converter = DocumentConverter()
            docling_result = doc_converter.convert(file_path)
            docling_document = docling_result.document

            # Extract structure
            structure = self.extract_document_structure(docling_document)

            # Detect language
            detected_language = self.detect_language(structure['full_content'])

            # Preprocess content
            if detected_language in self.nlp_models:
                processed_content = self.preprocess_text(structure['full_content'], detected_language)
            else:
                processed_content = structure['full_content']

            # Extract author
            author = self.extract_author(docling_document)

            # Update progress
            if self.progress_callback:
                self.progress_callback(filename, 30, "Creating document record...")

            # Create or update document record
            existing_doc = self.db.query(Document).filter(Document.file_hash == file_hash).first()

            if existing_doc:
                # Update existing document
                doc = existing_doc
                doc.filename = filename
                doc.filepath = file_path
                doc.detected_language = detected_language
                doc.full_content = structure['full_content']
                doc.chapter_content = structure['chapters']
                doc.toc_content = structure['toc']
                doc.content_structure = structure['metadata']
                doc.last_modified = datetime.now()
            else:
                # Create new document
                doc = Document(
                    filename=filename,
                    filepath=file_path,
                    file_hash=file_hash,
                    content_type=Path(file_path).suffix[1:] or 'unknown',
                    detected_language=detected_language,
                    status='processing',
                    full_content=structure['full_content'],
                    chapter_content=structure['chapters'],
                    toc_content=structure['toc'],
                    content_structure=structure['metadata']
                )
                self.db.add(doc)

            self.db.flush()  # Get document ID

            # Update progress
            if self.progress_callback:
                self.progress_callback(filename, 50, "Creating chapters and chunks...")

            # Create chapters
            chapter_objects = []  # Keep references to chapter objects
            chapters_created = 0
            for chapter_data in structure['chapters']:
                try:
                    # Validate chapter data
                    if not isinstance(chapter_data, dict):
                        print(f"Warning: Invalid chapter data type: {type(chapter_data)}, skipping")
                        continue

                    if 'title' not in chapter_data or 'content' not in chapter_data:
                        print(f"Warning: Chapter missing required fields: {chapter_data.keys()}, skipping")
                        continue

                    # Validate and clean chapter title (VARCHAR(255))
                    chapter_title = str(chapter_data['title']).strip()
                    if len(chapter_title) > 240:  # Conservative limit with margin
                        chapter_title = chapter_title[:237] + '...'
                    if not chapter_title:  # Ensure non-empty title
                        chapter_title = f"Chapter {chapters_created + 1}"

                    # Validate and clean chapter path (VARCHAR(100))
                    chapter_path = str(chapter_data.get('path', str(chapters_created + 1))).strip()
                    if len(chapter_path) > 90:  # Conservative limit with margin
                        chapter_path = chapter_path[:87] + '...'
                    if not chapter_path:  # Ensure non-empty path
                        chapter_path = str(chapters_created + 1)

                    # Validate and clean content
                    chapter_content = str(chapter_data['content']).strip()
                    # Remove excessive whitespace but preserve paragraph structure
                    chapter_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', chapter_content)
                    word_count = len(chapter_content.split()) if chapter_content else 0

                    # Skip empty chapters
                    if not chapter_content or word_count < 5:
                        print(f"Warning: Skipping empty chapter '{chapter_title}' with {word_count} words")
                        continue

                    # Final length checks before database insertion
                    if len(chapter_title.encode('utf-8')) > 255:
                        print(f"Warning: Chapter title too long after encoding, truncating: {chapter_title[:50]}...")
                        chapter_title = chapter_title.encode('utf-8')[:252].decode('utf-8', errors='ignore') + '...'

                    if len(chapter_path.encode('utf-8')) > 100:
                        print(f"Warning: Chapter path too long after encoding, truncating: {chapter_path[:30]}...")
                        chapter_path = chapter_path.encode('utf-8')[:97].decode('utf-8', errors='ignore') + '...'

                    chapter = DocumentChapter(
                        document_id=doc.id,
                        chapter_title=chapter_title,
                        chapter_path=chapter_path,
                        content=chapter_content,
                        section_type='chapter',
                        level=int(chapter_data.get('level', 1)),
                        word_count=word_count
                    )
                    self.db.add(chapter)
                    chapter_objects.append(chapter)
                    chapters_created += 1

                except Exception as e:
                    print(f"Warning: Failed to create chapter {chapters_created + 1}: {e}")
                    print(f"Chapter data: title='{chapter_data.get('title', 'N/A')[:50]}...', path='{chapter_data.get('path', 'N/A')}'")
                    continue

            # Create chunks from processed content
            from langchain_core.documents import Document as LangchainDocument
            langchain_doc = LangchainDocument(
                page_content=processed_content,
                metadata={"source": file_path, "language": detected_language}
            )

            chunks = split_documents([langchain_doc], chunk_size=1000, chunk_overlap=200)

            # Create embeddings for chunks
            chunks_created = 0
            if chunks:
                embeddings, _ = create_embeddings(chunks, "nomic-ai/nomic-embed-text-v1.5")

                # Save chunks to database
                for i, chunk in enumerate(chunks):
                    chunk_obj = DocumentChunk(
                        document_id=doc.id,
                        chunk_index=i,
                        content=chunk.page_content,
                        embedding_model="nomic-ai/nomic-embed-text-v1.5"
                    )
                    self.db.add(chunk_obj)
                    chunks_created += 1

                # Save embeddings to Elasticsearch
                self._save_embeddings_to_es(doc.id, chunks, embeddings)

            # Create chapter embeddings if chapters exist
            if chapter_objects:
                chapter_texts = [ch.content for ch in chapter_objects]
                if chapter_texts:
                    chapter_embeddings, _ = create_embeddings(
                        [LangchainDocument(page_content=text) for text in chapter_texts],
                        "nomic-ai/nomic-embed-text-v1.5"
                    )

                    # Update chapters with embeddings using the objects we just created
                    for chapter_obj, embedding in zip(chapter_objects, chapter_embeddings):
                        chapter_obj.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                        chapter_obj.embedding_model = "nomic-ai/nomic-embed-text-v1.5"

                    # Flush to get chapter IDs before indexing to ES
                    self.db.flush()

                    # Save chapter embeddings to Elasticsearch
                    self._save_chapter_embeddings_to_es(doc.id, chapter_objects, chapter_embeddings)

            # Update progress
            if self.progress_callback:
                self.progress_callback(filename, 85, "Finalizing...")

            # Update document status
            doc.status = 'processed'
            doc.last_modified = datetime.now()

            self.db.commit()

            # AI enrichment (after commit so chunks are available)
            if self.progress_callback:
                self.progress_callback(filename, 90, "AI enrichment...")

            try:
                from .ai_enrichment import AIEnrichmentService
                enrichment_service = AIEnrichmentService()
                enrichment_result = enrichment_service.enrich_document(doc.id, force=force_enrichment)

                if enrichment_result.get('success'):
                    print(f"AI enrichment completed for {filename}")
                else:
                    print(f"AI enrichment skipped: {enrichment_result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"AI enrichment failed: {e}")

            # Update progress
            if self.progress_callback:
                self.progress_callback(filename, 100, "Complete!")

            result.update({
                'success': True,
                'document_id': doc.id,
                'chunks_created': chunks_created,
                'chapters_created': chapters_created
            })

        except Exception as e:
            self.db.rollback()
            result['error'] = str(e)
            if self.progress_callback:
                self.progress_callback(filename, 0, f"Failed: {e}")

        return result

    def _save_embeddings_to_es(self, document_id: int, chunks: List, embeddings):
        """
        Save document chunks and embeddings to Elasticsearch.

        Args:
            document_id: ID of the parent document
            chunks: List of chunk objects
            embeddings: List of embedding vectors
        """
        try:
            actions = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                action = {
                    "_index": "rag_vectors",
                    "_id": f"nomic-embed_{document_id}_{i}",
                    "_source": {
                        "document_id": document_id,
                        "chunk_id": i,
                        "content": chunk.page_content,
                        "embedding": embedding.tolist(),
                        "embedding_model": "nomic-ai/nomic-embed-text-v1.5"
                    }
                }
                actions.append(action)

            if actions:
                from elasticsearch.helpers import bulk
                success, failed = bulk(self.es, actions, stats_only=False, raise_on_error=False)
                if failed:
                    print(f"Warning: Failed to index {failed} embeddings for document {document_id}")
        except Exception as e:
            print(f"Warning: Elasticsearch indexing failed for document {document_id}: {e}")

    def _save_chapter_embeddings_to_es(self, document_id: int, chapters: List, embeddings):
        """
        Save document chapters and embeddings to Elasticsearch.

        Args:
            document_id: ID of the parent document
            chapters: List of chapter objects
            embeddings: List of embedding vectors
        """
        try:
            actions = []
            for i, (chapter, embedding) in enumerate(zip(chapters, embeddings)):
                action = {
                    "_index": "rag_vectors",
                    "_id": f"chapter_{document_id}_{chapter.id}",
                    "_source": {
                        "document_id": document_id,
                        "chapter_id": chapter.id,
                        "chunk_id": -1,  # Special marker for chapters
                        "content": chapter.content,
                        "chapter_title": chapter.chapter_title,
                        "chapter_path": chapter.chapter_path,
                        "section_type": chapter.section_type,
                        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
                        "content_type": "chapter"
                    }
                }
                actions.append(action)

            if actions:
                from elasticsearch.helpers import bulk
                success, failed = bulk(self.es, actions, stats_only=False, raise_on_error=False)
                if failed:
                    print(f"Warning: Failed to index {failed} chapter embeddings for document {document_id}")
        except Exception as e:
            print(f"Warning: Chapter Elasticsearch indexing failed for document {document_id}: {e}")

    def upload_files(self, uploaded_files: List, data_dir: str = "data",
                    use_parallel: bool = True, max_workers: int = 4) -> Dict[str, Any]:
        """
        Upload and process multiple files with progress tracking and parallel processing.

        Args:
            uploaded_files: List of uploaded file objects (from Streamlit)
            data_dir: Directory to save processed files
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary with upload results and statistics
        """
        results = {
            'total_files': len(uploaded_files),
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_chunks': 0,
            'total_chapters': 0,
            'file_results': [],
            'errors': []
        }

        if not uploaded_files:
            return results

        # Create data directory
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)

        # Use temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded files to temp directory
            file_paths = []
            for uploaded_file in uploaded_files:
                # Handle duplicate names
                file_path = temp_path / uploaded_file.name
                counter = 1
                while file_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    file_path = temp_path / f"{stem}_{counter}{suffix}"
                    counter += 1

                # Save file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_paths.append((file_path, uploaded_file.name))

            # Process files
            if use_parallel and len(file_paths) > 1:
                # Parallel processing
                processed_files = self._process_files_parallel(
                    file_paths, max_workers, data_path
                )
            else:
                # Sequential processing
                processed_files = []
                for file_path, filename in file_paths:
                    # Move to permanent location
                    permanent_path = data_path / filename
                    counter = 1
                    while permanent_path.exists():
                        stem = permanent_path.stem
                        suffix = permanent_path.suffix
                        permanent_path = data_path / f"{stem}_{counter}{suffix}"
                        counter += 1

                    shutil.move(str(file_path), str(permanent_path))

                    # Calculate hash
                    file_hash = self.calculate_file_hash(str(permanent_path))

                    # Check for duplicates
                    existing_doc = self.db.query(Document).filter(
                        Document.file_hash == file_hash
                    ).first()

                    if existing_doc:
                        results['file_results'].append({
                            'filename': filename,
                            'success': False,
                            'error': 'Document already exists in database'
                        })
                        results['failed_uploads'] += 1
                        continue

                    # Process file
                    result = self.process_single_file(str(permanent_path), filename, file_hash)
                    processed_files.append(result)

            # Aggregate results
            for result in processed_files:
                results['file_results'].append(result)
                if result['success']:
                    results['successful_uploads'] += 1
                    results['total_chunks'] += result['chunks_created']
                    results['total_chapters'] += result['chapters_created']
                else:
                    results['failed_uploads'] += 1
                    results['errors'].append(f"{result['filename']}: {result['error']}")

        return results

    def _process_files_parallel(self, file_paths: List[Tuple[Path, str]], max_workers: int,
                               data_path: Path) -> List[Dict[str, Any]]:
        """
        Process files in parallel using ProcessPoolExecutor.

        Args:
            file_paths: List of (file_path, filename) tuples
            max_workers: Maximum number of workers
            data_path: Permanent data directory

        Returns:
            List of processing results
        """
        results = []

        # Prepare serializable data for workers
        worker_data = []
        for file_path, filename in file_paths:
            # Move to permanent location first
            permanent_path = data_path / filename
            counter = 1
            while permanent_path.exists():
                stem = permanent_path.stem
                suffix = permanent_path.suffix
                permanent_path = data_path / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(file_path), str(permanent_path))

            # Calculate hash
            file_hash = self.calculate_file_hash(str(permanent_path))

            worker_data.append({
                'filepath': str(permanent_path),
                'filename': filename,
                'file_hash': file_hash
            })

        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_file_worker, data): data['filename']
                for data in worker_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'filename': filename,
                        'error': str(e)
                    })

        return results

    @staticmethod
    def _process_file_worker(file_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Worker function for parallel file processing.

        Args:
            file_data: Dictionary with filepath, filename, and file_hash

        Returns:
            Processing result dictionary
        """
        # Create processor instance in worker
        processor = UploadProcessor()

        try:
            return processor.process_single_file(
                file_data['filepath'],
                file_data['filename'],
                file_data['file_hash']
            )
        finally:
            processor.db.close()