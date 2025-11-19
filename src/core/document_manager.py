#!/usr/bin/env python3
"""
Combined document management system for processing, tagging, and categorization.
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
from .base_processor import BaseProcessor
import spacy

from database.models import (
    Document,
    DocumentChunk,
    DocumentChapter,
    DocumentEmbedding,
    DocumentTag,
    DocumentCategory,
    DocumentTagAssignment,
    DocumentCategoryAssignment,
    SessionLocal,
)
from data.loader import split_documents
from core.embeddings import get_embedding_model, create_embeddings
from database.opensearch_setup import get_elasticsearch_client
from ai.tag_suggester import AITagSuggester
from utils.tag_colors import TagColorManager
import logging

logger = logging.getLogger(__name__)


class TagManager:
    """
    Manager class for document tags.

    Handles CRUD operations for tags and tag-document associations,
    with AI-powered suggestions and intelligent color management.
    """

    def __init__(self, db: Session):
        self.db = db
        self.ai_suggester = AITagSuggester()
        self.color_manager = TagColorManager()

    def get_tag_by_name(self, name: str) -> Optional[DocumentTag]:
        """Get a tag by name."""
        return self.db.query(DocumentTag).filter(DocumentTag.name == name).first()

    def create_tag(self, name: str, color: Optional[str] = None) -> DocumentTag:
        """Create a new tag with optional color."""
        if color is None:
            color = self.color_manager.generate_unique_color()

        tag = DocumentTag(name=name, color=color)
        self.db.add(tag)
        self.db.commit()
        return tag

    def create_tag_with_ai_color(self, name: str) -> DocumentTag:
        """Create a tag with AI-suggested color."""
        color = self.color_manager.generate_color(name)
        return self.create_tag(name, color)

    def add_tag_to_document(self, document_id: int, tag_id: int) -> bool:
        """Add a tag to a document."""
        # Check if assignment already exists
        existing = (
            self.db.query(DocumentTagAssignment)
            .filter(
                DocumentTagAssignment.document_id == document_id,
                DocumentTagAssignment.tag_id == tag_id,
            )
            .first()
        )

        if existing:
            return True

        assignment = DocumentTagAssignment(document_id=document_id, tag_id=tag_id)
        self.db.add(assignment)
        self.db.commit()
        return True

    def remove_tag_from_document(self, document_id: int, tag_id: int) -> bool:
        """Remove a tag from a document."""
        assignment = (
            self.db.query(DocumentTagAssignment)
            .filter(
                DocumentTagAssignment.document_id == document_id,
                DocumentTagAssignment.tag_id == tag_id,
            )
            .first()
        )

        if assignment:
            self.db.delete(assignment)
            self.db.commit()
            return True
        return False

    def get_document_tags(self, document_id: int) -> List[DocumentTag]:
        """Get all tags for a document."""
        assignments = (
            self.db.query(DocumentTagAssignment)
            .filter(DocumentTagAssignment.document_id == document_id)
            .all()
        )
        return [assignment.tag for assignment in assignments if assignment.tag]

    def suggest_tags_for_document(
        self, document_id: int, max_suggestions: int = 5
    ) -> List[str]:
        """Suggest tags for a document using AI."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return []

        # Get document content
        chunks = (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .limit(3)
            .all()
        )

        content = " ".join([chunk.content for chunk in chunks])

        # Use AI suggester to get tag suggestions
        if content.strip():
            return self.ai_suggester.suggest_tags(content, max_suggestions)
        return []

    def get_tag_usage_stats(self) -> List[Dict[str, Any]]:
        """Get usage statistics for all tags."""
        # Get all tags with their usage counts
        from sqlalchemy import func

        tag_stats = (
            self.db.query(
                DocumentTag.name,
                DocumentTag.color,
                DocumentTag.description,
                DocumentTag.usage_count,
                DocumentTag.created_at,
                func.count(DocumentTagAssignment.document_id).label("document_count"),
            )
            .outerjoin(DocumentTagAssignment)
            .group_by(DocumentTag.id)
            .all()
        )

        return [
            {
                "name": stat.name,
                "color": stat.color,
                "description": stat.description,
                "usage_count": stat.usage_count,
                "created_at": stat.created_at,
                "document_count": stat.document_count,
            }
            for stat in tag_stats
        ]

    def get_all_tags(self) -> List[DocumentTag]:
        """Get all tags."""
        return self.db.query(DocumentTag).all()

    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most popular tags by usage count."""
        # Get tags sorted by document count (descending)
        from sqlalchemy import func, desc

        popular_tags = (
            self.db.query(
                DocumentTag.name,
                DocumentTag.color,
                func.count(DocumentTagAssignment.document_id).label("document_count"),
            )
            .outerjoin(DocumentTagAssignment)
            .group_by(DocumentTag.id)
            .order_by(desc(func.count(DocumentTagAssignment.document_id)))
            .limit(limit)
            .all()
        )

        return [
            {"name": tag.name, "color": tag.color, "document_count": tag.document_count}
            for tag in popular_tags
            if tag.document_count > 0  # Only return tags that are actually used
        ]

    def get_related_tags(self, tag_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get tags that are often used together with the given tag."""
        # Find the tag by name
        tag = self.get_tag_by_name(tag_name)
        if not tag:
            return []

        # Simple approach: get all documents with this tag, then find other popular tags on those documents
        # This is a simplified version that works reliably
        try:
            # Get document IDs that have this tag
            doc_ids = [
                assignment.document_id
                for assignment in self.db.query(DocumentTagAssignment)
                .filter(DocumentTagAssignment.tag_id == tag.id)
                .all()
            ]

            if not doc_ids:
                return []

            # Get all other tags on those documents
            from sqlalchemy import func

            related_tags = (
                self.db.query(
                    DocumentTag.name,
                    DocumentTag.color,
                    func.count(DocumentTagAssignment.tag_id).label("co_occurrence"),
                )
                .join(DocumentTagAssignment)
                .filter(
                    DocumentTagAssignment.document_id.in_(doc_ids),
                    DocumentTag.id != tag.id,
                )
                .group_by(DocumentTag.id, DocumentTag.name, DocumentTag.color)
                .order_by(func.count(DocumentTagAssignment.tag_id).desc())
                .limit(limit)
                .all()
            )

            return [
                {"name": rt.name, "color": rt.color, "co_occurrence": rt.co_occurrence}
                for rt in related_tags
            ]
        except Exception:
            # Fallback: return empty list if query fails
            return []


class CategoryManager:
    """
    Manager class for document categories.

    Handles hierarchical category management with parent-child relationships.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_category_by_name(
        self, name: str, parent_id: Optional[int] = None
    ) -> Optional[DocumentCategory]:
        """Get a category by name and optional parent."""
        query = self.db.query(DocumentCategory).filter(DocumentCategory.name == name)
        if parent_id is not None:
            query = query.filter(DocumentCategory.parent_category_id == parent_id)
        return query.first()

    def create_category(
        self,
        name: str,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> DocumentCategory:
        """Create a new category."""
        category = DocumentCategory(
            name=name, description=description, parent_category_id=parent_id
        )
        self.db.add(category)
        self.db.commit()
        return category

    def add_category_to_document(self, document_id: int, category_id: int) -> bool:
        """Add a category to a document."""
        # Check if assignment already exists
        existing = (
            self.db.query(DocumentCategoryAssignment)
            .filter(
                DocumentCategoryAssignment.document_id == document_id,
                DocumentCategoryAssignment.category_id == category_id,
            )
            .first()
        )

        if existing:
            return True

        assignment = DocumentCategoryAssignment(
            document_id=document_id, category_id=category_id
        )
        self.db.add(assignment)
        self.db.commit()
        return True

    def remove_category_from_document(self, document_id: int, category_id: int) -> bool:
        """Remove a category from a document."""
        assignment = (
            self.db.query(DocumentCategoryAssignment)
            .filter(
                DocumentCategoryAssignment.document_id == document_id,
                DocumentCategoryAssignment.category_id == category_id,
            )
            .first()
        )

        if assignment:
            self.db.delete(assignment)
            self.db.commit()
            return True
        return False

    def get_document_categories(self, document_id: int) -> List[DocumentCategory]:
        """Get all categories for a document."""
        assignments = (
            self.db.query(DocumentCategoryAssignment)
            .filter(DocumentCategoryAssignment.document_id == document_id)
            .all()
        )
        return [
            assignment.category for assignment in assignments if assignment.category
        ]

    def get_category_hierarchy(self, category_id: int) -> List[DocumentCategory]:
        """Get the full hierarchy path for a category."""
        hierarchy = []
        current = (
            self.db.query(DocumentCategory)
            .filter(DocumentCategory.id == category_id)
            .first()
        )

        while current:
            hierarchy.insert(0, current)
            if current.parent_category_id:
                current = (
                    self.db.query(DocumentCategory)
                    .filter(DocumentCategory.id == current.parent_category_id)
                    .first()
                )
            else:
                current = None

        return hierarchy

    def get_category_hierarchy_path(self, category_id: int) -> List[str]:
        """Get the full hierarchy path for a category as a list of names."""
        hierarchy = self.get_category_hierarchy(category_id)
        return [cat.name for cat in hierarchy]

    def get_root_categories(self) -> List[DocumentCategory]:
        """Get all root categories (categories with no parent)."""
        return (
            self.db.query(DocumentCategory)
            .filter(DocumentCategory.parent_category_id.is_(None))
            .order_by(DocumentCategory.name)
            .all()
        )

    def get_category_usage_stats(self) -> List[Dict[str, Any]]:
        """Get usage statistics for all categories."""
        # Get all categories with their usage counts
        from sqlalchemy import func

        cat_stats = (
            self.db.query(
                DocumentCategory.name,
                DocumentCategory.description,
                func.count(DocumentCategoryAssignment.document_id).label(
                    "document_count"
                ),
            )
            .outerjoin(DocumentCategoryAssignment)
            .group_by(DocumentCategory.id)
            .all()
        )

        return [
            {
                "name": stat.name,
                "description": stat.description or "",
                "document_count": stat.document_count,
            }
            for stat in cat_stats
        ]


class DocumentProcessor(BaseProcessor):
    """
    Enhanced document processor with integrated structure extraction and chapter-aware processing.

    This module provides the DocumentProcessor class which handles:
    - Integrated Docling document structure extraction during upload
    - Chapter-aware chunking with hierarchical metadata
    - Parallel processing for multiple files
    - Progress tracking and error handling
    - Automatic language detection and preprocessing
    """

    def __init__(self, db: Session = None):
        super().__init__(db or SessionLocal())
        self.tag_manager = TagManager(self.db)
        self.category_manager = CategoryManager(self.db)

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def process_document(
        self, file_path: str, filename: str = None, progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Process a single document with enhanced structure extraction.

        Args:
            file_path: Path to the document file
            filename: Optional custom filename
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with processing results
        """
        if filename is None:
            filename = Path(file_path).name

        if progress_callback:
            progress_callback(0, f"Starting processing of {filename}")

        try:
            # Detect language
            language = self._detect_language_from_file(file_path)
            if progress_callback:
                progress_callback(10, f"Detected language: {language}")

            # Load and split document
            documents = split_documents([file_path])
            if not documents:
                return {"success": False, "error": "Failed to load document"}

            doc_content = documents[0]

            # Create document record
            import hashlib

            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            document = Document(
                filename=filename,
                filepath=file_path,
                file_hash=file_hash,
                detected_language=language,
                status="processing",
            )
            self.db.add(document)
            self.db.commit()

            if progress_callback:
                progress_callback(20, "Document record created")

            # Process content into chunks
            chunks = self._create_chunks(doc_content, document.id)

            if progress_callback:
                progress_callback(60, f"Created {len(chunks)} chunks")

            # Generate embeddings
            embeddings_array, model = create_embeddings(
                [chunk["content"] for chunk in chunks]
            )

            # Store chunks with embeddings
            for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings_array)):
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    chapter_title=chunk_data.get("chapter_title"),
                    chapter_path=chunk_data.get("chapter_path"),
                )
                self.db.add(chunk)
                self.db.flush()  # Get chunk ID

                # Create embedding record
                embedding_record = DocumentEmbedding(
                    chunk_id=chunk.id,
                    embedding=embedding.tolist(),
                    embedding_model="nomic-ai/nomic-embed-text-v1.5",
                )
                self.db.add(embedding_record)

            self.db.commit()

            if progress_callback:
                progress_callback(90, "Embeddings generated and stored")

            # Index in Elasticsearch
            self._index_document(document, chunks, embeddings_array.tolist())

            if progress_callback:
                progress_callback(100, "Document processing complete")

            return {
                "success": True,
                "document_id": document.id,
                "chunks_count": len(chunks),
                "language": language,
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self.db.rollback()
            return {"success": False, "error": str(e)}

    def _detect_language_from_file(self, file_path: str) -> str:
        """Detect language from file content."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(1000)
                return detect(sample)
        except (LangDetectException, FileNotFoundError):
            return "en"

    def _create_chunks(self, content: str, document_id: int) -> List[Dict[str, Any]]:
        """Create chunks from document content."""
        # Simple chunking for now - can be enhanced with hierarchical chunking
        chunk_size = 1000
        overlap = 200

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunk_content = content[start:end]

            # Find sentence boundary for better chunks
            if end < len(content):
                # Look for sentence endings
                sentence_endings = [". ", "! ", "? ", "\n\n"]
                best_end = end
                for ending in sentence_endings:
                    pos = chunk_content.rfind(ending)
                    if pos > chunk_size * 0.7:  # Don't break too early
                        best_end = start + pos + len(ending)
                        break
                end = best_end

            chunk = {
                "content": content[start:end],
                "metadata": {
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                },
            }
            chunks.append(chunk)

            start = end - overlap
            chunk_index += 1

        return chunks

    def _index_document(
        self,
        document: Document,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        """Index document in Elasticsearch."""
        try:
            es_client = get_elasticsearch_client()

            # Index each chunk
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "document_id": document.id,
                    "content": chunk["content"],
                    "embedding": embedding,
                    "chunk_index": i,
                    "filename": document.filename,
                    "language": document.detected_language,
                    "metadata": chunk.get("metadata", {}),
                }
                es_client.index(index="documents", id=f"{document.id}_{i}", body=doc)

        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {e}")

    def _process_single_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a single document for the base class method."""
        return self.process_document(file_path, filename)


class UploadProcessor(BaseProcessor):
    """
    Enhanced document upload processor with batch processing capabilities.
    """

    def __init__(self, max_workers: int = None):
        super().__init__(SessionLocal())
        self.max_workers = max_workers or min(mp.cpu_count(), 4)

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def process_files(
        self, file_paths: List[str], progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for overall progress

        Returns:
            Dict with batch processing results
        """
        results = []
        total_files = len(file_paths)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in file_paths
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress = (i + 1) / total_files * 100
                        progress_callback(
                            progress, f"Processed {i + 1}/{total_files} files"
                        )

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append(
                        {"file_path": file_path, "success": False, "error": str(e)}
                    )

        # Summarize results
        successful = sum(1 for r in results if r.get("success", False))
        failed = total_files - successful

        return {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def process_single_file(
        self,
        file_path: str,
        filename: str = None,
        file_hash: str = None,
        force_enrichment: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single file with optional reprocessing capabilities.

        Args:
            file_path: Path to the file to process
            filename: Optional filename (used for database lookup)
            file_hash: Optional file hash (used for database lookup)
            force_enrichment: Whether to force AI enrichment even if document exists

        Returns:
            Dict with processing results
        """
        try:
            # Calculate file hash if not provided
            if not file_hash:
                import hashlib

                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

            # Use filename from path if not provided
            if not filename:
                filename = Path(file_path).name

            # Check if document already exists in database
            existing_doc = None
            if filename and file_hash:
                existing_doc = (
                    self.db.query(Document)
                    .filter(
                        Document.filename == filename, Document.file_hash == file_hash
                    )
                    .first()
                )

            if existing_doc and not force_enrichment:
                # Document exists and we're not forcing reprocessing
                return {
                    "success": True,
                    "filename": filename,
                    "message": "Document already exists",
                    "document_id": existing_doc.id,
                    "chunks_created": 0,
                    "chapters_created": 0,
                }

            # Process the document
            processor = DocumentProcessor()
            result = processor.process_document(file_path, filename)

            # If this is a reprocessing operation, update existing document
            if existing_doc and force_enrichment:
                result = self.reprocess_existing_document(
                    existing_doc, result, file_path
                )
            elif result.get("success"):
                # DocumentProcessor already stored the document, just format the result
                result["chunks_created"] = result.get("chunks_count", 0)
                result["chapters_created"] = (
                    0  # DocumentProcessor doesn't create chapters yet
                )
                result["filename"] = filename

            return result

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                "success": False,
                "filename": filename,
                "error": str(e),
                "chunks_created": 0,
                "chapters_created": 0,
            }

    def reprocess_existing_document(
        self, existing_doc: Document, processing_result: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document with new processing results.

        Args:
            existing_doc: The existing document record
            processing_result: Results from document processing
            file_path: Path to the file being reprocessed

        Returns:
            Dict with reprocessing results
        """
        try:
            # Update document metadata
            existing_doc.last_modified = datetime.now()
            existing_doc.status = "processed"

            # Update AI-enriched fields if available
            if "document_summary" in processing_result:
                existing_doc.document_summary = processing_result["document_summary"]
            if "key_topics" in processing_result:
                existing_doc.key_topics = processing_result["key_topics"]
            if "reading_time_minutes" in processing_result:
                existing_doc.reading_time_minutes = processing_result[
                    "reading_time_minutes"
                ]
            if "author" in processing_result:
                existing_doc.author = processing_result["author"]
            if "publication_date" in processing_result:
                existing_doc.publication_date = processing_result["publication_date"]
            if "detected_language" in processing_result:
                existing_doc.detected_language = processing_result["detected_language"]

            # Clear existing chunks and chapters for this document
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing_doc.id
            ).delete()
            self.db.query(DocumentChapter).filter(
                DocumentChapter.document_id == existing_doc.id
            ).delete()

            # Add new chunks and chapters
            chunks_created = 0
            chapters_created = 0

            if "chunks" in processing_result:
                for chunk_data in processing_result["chunks"]:
                    chunk = DocumentChunk(
                        document_id=existing_doc.id,
                        content=chunk_data["content"],
                        chunk_index=chunk_data["chunk_index"],
                        word_count=chunk_data.get("word_count", 0),
                        chapter_path=chunk_data.get("chapter_path"),
                        chapter_title=chunk_data.get("chapter_title"),
                    )
                    self.db.add(chunk)
                    chunks_created += 1

            if "chapters" in processing_result:
                for chapter_data in processing_result["chapters"]:
                    chapter = DocumentChapter(
                        document_id=existing_doc.id,
                        chapter_title=chapter_data["title"],
                        chapter_path=chapter_data["path"],
                        level=chapter_data["level"],
                        word_count=chapter_data.get("word_count", 0),
                        content=chapter_data.get("content"),
                    )
                    self.db.add(chapter)
                    chapters_created += 1

            # Create embeddings for new chunks
            if chunks_created > 0:
                chunks = (
                    self.db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == existing_doc.id)
                    .all()
                )
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings_array, _ = create_embeddings(chunk_texts)

                # Clear existing embeddings
                self.db.query(DocumentEmbedding).filter(
                    DocumentEmbedding.chunk_id.in_([chunk.id for chunk in chunks])
                ).delete()

                # Store new embeddings
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_array)):
                    doc_embedding = DocumentEmbedding(
                        chunk_id=chunk.id,
                        embedding=embedding.tolist(),
                        embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    )
                    self.db.add(doc_embedding)

                # Update search index
                processor = DocumentProcessor()
                processor._index_document(
                    existing_doc,
                    [{"content": chunk.content, "id": chunk.id} for chunk in chunks],
                    embeddings_array.tolist(),
                )

            self.db.commit()

            return {
                "success": True,
                "filename": existing_doc.filename,
                "message": f"Reprocessed document with {chunks_created} chunks and {chapters_created} chapters",
                "document_id": existing_doc.id,
                "chunks_created": chunks_created,
                "chapters_created": chapters_created,
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to reprocess document {existing_doc.id}: {e}")
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": str(e),
                "chunks_created": 0,
                "chapters_created": 0,
            }

    def _store_processed_document(
        self, processing_result: Dict[str, Any], file_path: str, filename: str
    ) -> Dict[str, Any]:
        """Store a newly processed document in the database."""
        try:
            # Create document record
            doc = Document(
                filename=filename,
                filepath=file_path,
                file_hash=processing_result.get("file_hash", ""),
                status="processed",
                detected_language=processing_result.get("detected_language"),
                document_summary=processing_result.get("document_summary"),
                key_topics=processing_result.get("key_topics"),
                reading_time_minutes=processing_result.get("reading_time_minutes"),
                author=processing_result.get("author"),
                publication_date=processing_result.get("publication_date"),
            )
            self.db.add(doc)
            self.db.flush()  # Get the document ID

            # Add chunks and chapters
            chunks_created = 0
            chapters_created = 0

            if "chunks" in processing_result:
                for chunk_data in processing_result["chunks"]:
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        content=chunk_data["content"],
                        chunk_index=chunk_data["chunk_index"],
                        word_count=chunk_data.get("word_count", 0),
                        chapter_path=chunk_data.get("chapter_path"),
                        chapter_title=chunk_data.get("chapter_title"),
                    )
                    self.db.add(chunk)
                    chunks_created += 1

            if "chapters" in processing_result:
                for chapter_data in processing_result["chapters"]:
                    chapter = DocumentChapter(
                        document_id=doc.id,
                        chapter_title=chapter_data["title"],
                        chapter_path=chapter_data["path"],
                        level=chapter_data["level"],
                        word_count=chapter_data.get("word_count", 0),
                        content=chapter_data.get("content"),
                    )
                    self.db.add(chapter)
                    chapters_created += 1

            # Create embeddings
            if chunks_created > 0:
                chunks = (
                    self.db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == doc.id)
                    .all()
                )
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings_array, _ = create_embeddings(chunk_texts)

                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_array)):
                    doc_embedding = DocumentEmbedding(
                        chunk_id=chunk.id,
                        embedding=embedding.tolist(),
                        embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    )
                    self.db.add(doc_embedding)

                # Index in search
                processor = DocumentProcessor()
                processor._index_document(
                    doc,
                    [{"content": chunk.content, "id": chunk.id} for chunk in chunks],
                    embeddings_array.tolist(),
                )

            self.db.commit()

            processing_result.update(
                {
                    "document_id": doc.id,
                    "chunks_created": chunks_created,
                    "chapters_created": chapters_created,
                }
            )

            return processing_result

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to store processed document: {e}")
            processing_result["success"] = False
            processing_result["error"] = str(e)
            return processing_result

    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file (runs in separate process)."""
        try:
            processor = DocumentProcessor()
            result = processor.process_document(file_path)
            result["file_path"] = file_path
            return result
        except Exception as e:
            return {"file_path": file_path, "success": False, "error": str(e)}

    def upload_files(
        self,
        uploaded_files,
        data_dir: str = "data",
        use_parallel: bool = True,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Upload and process multiple files from Streamlit file uploader.

        Args:
            uploaded_files: List of Streamlit uploaded file objects
            data_dir: Directory to save uploaded files
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers

        Returns:
            Dict with upload results
        """
        import os
        from pathlib import Path

        # Ensure data directory exists
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)

        saved_files = []
        successful_uploads = 0
        failed_uploads = 0
        total_chunks = 0
        total_chapters = 0
        errors = []

        try:
            # Save uploaded files to disk
            for uploaded_file in uploaded_files:
                try:
                    # Create safe filename
                    safe_filename = "".join(
                        c for c in uploaded_file.name if c.isalnum() or c in "._- "
                    )
                    file_path = data_path / safe_filename

                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    saved_files.append(str(file_path))
                    print(f"✅ Saved file: {safe_filename}")

                except Exception as e:
                    print(f"❌ Failed to save {uploaded_file.name}: {e}")
                    failed_uploads += 1

            if not saved_files:
                return {
                    "success": False,
                    "error": "No files were successfully saved",
                    "successful_uploads": 0,
                    "failed_uploads": failed_uploads,
                    "total_chunks": 0,
                    "total_chapters": 0,
                }

            # Process the saved files
            if use_parallel and len(saved_files) > 1:
                # Use parallel processing
                results = self.process_files(saved_files)
            else:
                # Process sequentially
                results = []
                for file_path in saved_files:
                    try:
                        result = self.process_single_file(file_path)
                        results.append(result)
                    except Exception as e:
                        results.append(
                            {"file_path": file_path, "success": False, "error": str(e)}
                        )

            # Aggregate results
            for result in results:
                if result.get("success", False):
                    successful_uploads += 1
                    total_chunks += result.get("chunks_created", 0)
                    total_chapters += result.get("chapters_created", 0)
                else:
                    failed_uploads += 1
                    if result.get("error"):
                        errors.append(
                            f"{result.get('filename', 'Unknown file')}: {result['error']}"
                        )

            return {
                "success": successful_uploads > 0,
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "total_chunks": total_chunks,
                "total_chapters": total_chapters,
                "file_results": results,
                "errors": errors,
                "message": f"Processed {successful_uploads} files successfully",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "total_chunks": total_chunks,
                "total_chapters": total_chapters,
                "file_results": [],
                "errors": [str(e)],
            }

    def _process_single_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a single document for the base class method."""
        # Create a temporary DocumentProcessor to handle the actual processing
        processor = DocumentProcessor(self.db)
        return processor.process_document(file_path, filename)
