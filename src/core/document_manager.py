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
from src.core.base_processor import BaseProcessor
from src.data.loader import AdvancedDocumentProcessor
import spacy

from src.database.models import (
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
from src.data.loader import split_documents
from src.core.embeddings import get_embedding_model, create_embeddings
from src.database.opensearch_setup import get_elasticsearch_client
from src.ai.tag_suggester import AITagSuggester
from src.utils.tag_colors import TagColorManager
from src.utils.content_validator import ContentValidator
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
            color = self.color_manager.generate_color(name)

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

    def delete_tag(self, tag_name: str) -> bool:
        """Delete a tag and all its assignments."""
        try:
            # Find the tag
            tag = self.get_tag_by_name(tag_name)
            if not tag:
                return False

            # Remove all tag assignments
            self.db.query(DocumentTagAssignment).filter(
                DocumentTagAssignment.tag_id == tag.id
            ).delete(synchronize_session=False)

            # Delete the tag itself
            self.db.delete(tag)
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete tag {tag_name}: {e}")
            return False

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

    def get_category_tree(self) -> List[Dict[str, Any]]:
        """Get the complete category tree with hierarchy."""
        from sqlalchemy import func

        def build_tree(parent_id=None, level=0):
            categories = (
                self.db.query(DocumentCategory)
                .filter(DocumentCategory.parent_category_id == parent_id)
                .order_by(DocumentCategory.name)
                .all()
            )

            tree = []
            for category in categories:
                # Get document count for this category
                doc_count = (
                    self.db.query(func.count(DocumentCategoryAssignment.document_id))
                    .filter(DocumentCategoryAssignment.category_id == category.id)
                    .scalar()
                )

                node = {
                    "id": category.id,
                    "name": category.name,
                    "description": category.description,
                    "level": level,
                    "document_count": doc_count or 0,
                    "children": build_tree(category.id, level + 1),
                }
                tree.append(node)

            return tree

        return build_tree()

    def delete_category(self, category_id: int) -> bool:
        """Delete a category and all its subcategories."""
        from sqlalchemy import func

        try:
            # First, get all descendant categories (recursive deletion)
            def get_descendants(cat_id):
                descendants = [cat_id]
                children = (
                    self.db.query(DocumentCategory.id)
                    .filter(DocumentCategory.parent_category_id == cat_id)
                    .all()
                )
                for child in children:
                    descendants.extend(get_descendants(child.id))
                return descendants

            all_category_ids = get_descendants(category_id)

            # Remove all category assignments for these categories
            self.db.query(DocumentCategoryAssignment).filter(
                DocumentCategoryAssignment.category_id.in_(all_category_ids)
            ).delete(synchronize_session=False)

            # Delete the categories themselves (in reverse order to handle foreign keys)
            for cat_id in reversed(all_category_ids):
                category = (
                    self.db.query(DocumentCategory)
                    .filter(DocumentCategory.id == cat_id)
                    .first()
                )
                if category:
                    self.db.delete(category)

            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete category {category_id}: {e}")
            return False


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
        self.tag_suggester = AITagSuggester()

    def _suggest_categories_ai(
        self, content: str, filename: str, tags: List[str]
    ) -> List[str]:
        """
        Suggest categories for a document using AI-based classification.

        Args:
            content: Document content
            filename: Document filename
            tags: Generated tags

        Returns:
            List of suggested category names
        """
        try:
            # Use the tag suggester's LLM to classify categories
            category_prompt = f"""
            Analyze this document and suggest 1-3 relevant categories.
            Choose from: Academic, Technical, Business, Scientific, Educational, Legal, Medical, Creative, Reference, General

            Document: {filename}
            Content preview: {content[:500]}
            Tags: {", ".join(tags)}

            Return only category names separated by commas (no explanations):
            """

            response = self.tag_suggester._call_llm(
                category_prompt, max_tokens=50
            ).strip()

            # Parse the response
            categories = [cat.strip() for cat in response.split(",") if cat.strip()]

            # Validate categories
            valid_categories = [
                "Academic",
                "Technical",
                "Business",
                "Scientific",
                "Educational",
                "Legal",
                "Medical",
                "Creative",
                "Reference",
                "General",
            ]

            # Filter to valid categories and limit to 3
            validated_categories = [
                cat for cat in categories if cat in valid_categories
            ][:3]

            return validated_categories

        except Exception as e:
            # Fallback to simple keyword-based categorization
            content_lower = content.lower()
            categories = []

            if any(
                word in content_lower
                for word in ["deep learning", "neural", "machine learning", "ai"]
            ):
                categories.append("Technical")
            if any(
                word in content_lower
                for word in ["academic", "research", "paper", "study"]
            ):
                categories.append("Academic")
            if any(
                word in content_lower
                for word in ["tutorial", "guide", "course", "education"]
            ):
                categories.append("Educational")

            return categories[:3] if categories else ["General"]

    def _generate_document_summary(
        self, content: str, filename: str, tags: List[str], chapters_count: int
    ) -> str:
        """
        Generate an AI-powered document summary.

        Args:
            content: Document content sample
            filename: Document filename
            tags: Generated tags
            chapters_count: Number of chapters detected

        Returns:
            AI-generated document summary
        """
        try:
            # Create a comprehensive summary prompt
            summary_prompt = f"""
            Create a concise but informative summary of this document in 2-3 sentences.

            Content preview: {content[:800]}
            Tags: {", ".join(tags[:5])}  # Limit tags for prompt

            Focus on:
            - Main topics and subject matter
            - Key concepts covered
            - Document type and purpose
            - Target audience (if apparent)

            Summary should be professional and informative. Do not mention the filename or chapter/section counts.
            """

            summary = self.tag_suggester._call_llm(
                summary_prompt, max_tokens=200
            ).strip()

            # Clean up the summary - remove unwanted prefixes
            if summary:
                # Remove common AI prefixes
                prefixes_to_remove = [
                    "Here is a concise summary of the document in 2-3 sentences:",
                    "Here is a concise summary of the document:",
                    "Here is a summary of the document:",
                    "Summary:",
                    "Document summary:",
                ]

                for prefix in prefixes_to_remove:
                    if summary.lower().startswith(prefix.lower()):
                        summary = summary[len(prefix) :].strip()
                        break

                # Remove any leading/trailing artifacts
                summary = summary.strip()
                if summary.startswith('"') and summary.endswith('"'):
                    summary = summary[1:-1]
                if summary.startswith("'") and summary.endswith("'"):
                    summary = summary[1:-1]

                # Return clean summary without structure info
                return summary
            else:
                # Fallback summary
                return f"Document about {', '.join(tags[:3]) if tags else 'various topics'}. {chapters_count} chapters detected."

        except Exception as e:
            # Fallback to basic summary
            return f"Document processed with advanced AI pipeline. Covers {', '.join(tags[:3]) if tags else 'various topics'}. {chapters_count} chapters detected."

    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, "db"):
            self.db.close()

    def process_document(
        self,
        file_path: str,
        filename: Optional[str] = None,
        use_advanced_processing: bool = False,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document with optional advanced AI-powered pipeline.

        Args:
            file_path: Path to the document file
            filename: Optional filename override
            use_advanced_processing: Whether to use comprehensive AI pipeline
            progress_callback: Optional progress callback

        Returns:
            Processing results
        """
        if use_advanced_processing:
            logger.info(
                f"Using ADVANCED processing for {filename or os.path.basename(file_path)}"
            )
            return self._process_document_advanced(
                file_path, filename, progress_callback
            )
        else:
            logger.info(
                f"Using STANDARD processing for {filename or os.path.basename(file_path)}"
            )
            return self._process_document_standard(
                file_path, filename, progress_callback
            )

    def _process_document_advanced(
        self,
        file_path: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process document using comprehensive AI-powered pipeline.
        """
        try:
            # Initialize advanced processor
            advanced_processor = AdvancedDocumentProcessor()

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    5,
                    "Starting advanced AI processing...",
                )

            # Run comprehensive processing
            results = advanced_processor.process_document_comprehensive(file_path)

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    50,
                    "AI processing complete, storing results...",
                )

            # Store results in database (simplified for now)
            # This would need to be expanded to store all the advanced metadata
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Generate tags using AI
            content_for_analysis = results.get("extracted_content", "")[
                :2000
            ]  # Sample for analysis
            suggested_tags_data = self.tag_suggester.suggest_tags(
                content_for_analysis, filename or os.path.basename(file_path)
            )
            # Extract tag names from the suggestion data
            suggested_tags = [
                tag.get("tag", "") for tag in suggested_tags_data if tag.get("tag")
            ]

            # Generate categories using AI-based classification
            suggested_categories = self._suggest_categories_ai(
                content_for_analysis,
                filename or os.path.basename(file_path),
                suggested_tags,
            )

            # Generate AI-powered document summary
            document_summary = self._generate_document_summary(
                content_for_analysis,
                filename or os.path.basename(file_path),
                suggested_tags,
                results.get("chapters_detected", 0),
            )

            processing_result = {
                "file_hash": file_hash,
                "detected_language": results.get(
                    "language", "de"
                ),  # Default to German for technical docs
                "document_summary": document_summary,
                "key_topics": results.get("topics", []),
                "reading_time_minutes": max(
                    1, len(results.get("extracted_content", "")) // 2000
                ),  # Rough estimate
                "suggested_tags": suggested_tags,
                "suggested_categories": suggested_categories,
            }

            document = Document(
                filename=filename or os.path.basename(file_path),
                filepath=file_path,
                file_hash=processing_result["file_hash"],
                status="processed",
                detected_language=processing_result["detected_language"],
                document_summary=processing_result["document_summary"],
                key_topics=processing_result["key_topics"],
                reading_time_minutes=processing_result["reading_time_minutes"],
            )
            self.db.add(document)
            self.db.flush()

            # Store suggested tags
            for tag_name in processing_result.get("suggested_tags", []):
                if tag_name:
                    tag = self.tag_manager.get_tag_by_name(tag_name)
                    if not tag:
                        tag = self.tag_manager.create_tag(tag_name)
                    self.tag_manager.add_tag_to_document(document.id, tag.id)

            # Store suggested categories
            for category_name in processing_result.get("suggested_categories", []):
                if category_name:
                    category = self.category_manager.get_category_by_name(category_name)
                    if not category:
                        category = self.category_manager.create_category(category_name)
                    self.category_manager.add_category_to_document(
                        document.id, category.id
                    )

            # Store chapters from advanced processing
            structure_info = results.get("structure_analysis", {})
            hierarchy = structure_info.get("hierarchy", [])
            for chapter_data in hierarchy:
                chapter_record = DocumentChapter(
                    document_id=document.id,
                    chapter_title=chapter_data.get("title", "")[
                        :255
                    ],  # Limit to 255 chars
                    chapter_path=chapter_data.get("path", ""),
                    level=chapter_data.get("level", 1),
                    word_count=chapter_data.get("word_count", 0),
                    content=chapter_data.get(
                        "content_preview", chapter_data.get("title", "")
                    ),
                    section_type="chapter"
                    if chapter_data.get("level", 1) == 1
                    else "section",
                )
                self.db.add(chapter_record)

            # Store all chunks from advanced processing
            for i, chunk_data in enumerate(results.get("chunks", [])):
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=i,
                    chapter_title=chunk_data.get("chapter", "auto"),
                    embedding_model="sentence-transformers",  # Default embedding model
                )
                self.db.add(chunk)

            self.db.commit()

            if progress_callback:
                progress_callback(
                    filename or os.path.basename(file_path),
                    100,
                    "Advanced processing complete!",
                )

            return {
                "success": True,
                "document_id": document.id,
                "advanced_processing": True,
                "processing_stages": results.get("processing_stages", []),
                "chapters_detected": len(results.get("chapters", [])),
                "chunks_created": len(results.get("chunks", [])),
                "topics_identified": len(results.get("topics", [])),
            }

        except Exception as e:
            logger.error(f"Advanced document processing failed for {file_path}: {e}")
            # Fallback to standard processing
            return self._process_document_standard(
                file_path, filename, progress_callback
            )

    def _process_document_standard(
        self,
        file_path: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
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

            # Load document content
            doc_content = split_documents([file_path])
            if not doc_content:
                return {"success": False, "error": "Failed to load document"}

            # Detect language from extracted content
            language = self._detect_language_from_content(doc_content)

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
                full_content=doc_content,
            )
            self.db.add(document)
            self.db.commit()

            if progress_callback:
                progress_callback(20, "Document record created")

            # Detect chapters from full content first
            all_chapters = self._detect_all_chapters(doc_content)

            # Process content into chunks
            chunks = self._create_chunks(doc_content, document.id, all_chapters)

            if progress_callback:
                progress_callback(
                    60,
                    f"Created {len(chunks)} chunks, detected {len(all_chapters)} chapters",
                )

            # Store chapters in database
            for chapter in all_chapters:
                chapter_record = DocumentChapter(
                    document_id=document.id,
                    chapter_title=chapter["title"],  # Short title (max 255 chars)
                    chapter_path=chapter["path"],
                    level=chapter.get("level", 1),
                    word_count=len(chapter["title"].split()),
                    content=chapter.get("content", chapter["title"]),  # Full content
                )
                self.db.add(chapter_record)
            self.db.commit()

            # Generate embeddings
            embeddings_array, model = create_embeddings(
                [chunk["content"] for chunk in chunks]
            )

            # Store chunks with embeddings and enhanced metadata
            for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings_array)):
                metadata = chunk_data.get("metadata", {})

                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    embedding_model="nomic-ai/nomic-embed-text-v1.5",
                    chapter_title=metadata.get("chapter_title"),
                    chapter_path=metadata.get("chapter_path"),
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

            # AI enrichment
            if progress_callback:
                progress_callback(95, "Running AI enrichment...")

            try:
                from ai.enrichment import AIEnrichmentService

                enrichment_service = AIEnrichmentService()
                enrichment_result = enrichment_service.enrich_document(document.id)

                # Update document with enrichment data
                if "summary" in enrichment_result:
                    document.document_summary = enrichment_result["summary"]
                if "topics" in enrichment_result:
                    document.key_topics = enrichment_result["topics"]

                self.db.commit()

                if progress_callback:
                    progress_callback(98, "AI enrichment completed")

            except Exception as e:
                print(f"⚠️ AI enrichment failed: {e}")
                # Continue without enrichment

            # Update document status to processed
            document.status = "processed"
            self.db.commit()

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

    def _detect_language_from_content(self, content: str) -> str:
        """Detect language from extracted content."""
        try:
            # Use multiple samples for better detection
            samples = [
                content[:2000],  # Beginning
                content[len(content) // 3 : len(content) // 3 + 2000]
                if len(content) > 6000
                else content[len(content) // 2 : len(content) // 2 + 1000],  # Middle
                content[-2000:]
                if len(content) > 2000
                else content[len(content) // 2 :],  # End
            ]

            # Try to detect language from each sample
            detections = []
            for sample in samples:
                if len(sample.strip()) > 100:  # Only use substantial samples
                    try:
                        lang = detect(sample)
                        detections.append(lang)
                    except LangDetectException:
                        continue

            # Return the most common detection, or 'en' as fallback
            if detections:
                return max(set(detections), key=detections.count)
            else:
                return "en"
        except (LangDetectException, ValueError):
            return "en"

    def _detect_all_chapters(self, content: str) -> List[Dict[str, Any]]:
        """Detect all chapters from the full document content."""
        chapters = []

        # Look for ## headers (markdown)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                header_text = line.strip()[2:].strip()  # Remove ##
                if header_text and len(header_text) > 3:
                    # Extract just the title (first line, truncated to 255 chars)
                    title_lines = header_text.split("\n")
                    short_title = title_lines[0].strip()[:255]  # Limit to 255 chars

                    chapters.append(
                        {
                            "title": short_title,
                            "content": header_text,  # Store full content separately
                            "path": f"section_{len(chapters) + 1}",
                            "start_line": i,
                            "level": 2,
                        }
                    )

        # Look for table format | number | title | (markdown tables)
        import re

        table_matches = re.findall(
            r"\|\s*(\d+(?:\.\d+)*)\s*\|\s*([^\|]+?)\s*\|", content
        )
        for number, title in table_matches:
            title = title.strip()
            if title and len(title) > 3:
                # Truncate title to 255 characters for database
                short_title = title[:255]

                chapters.append(
                    {
                        "title": short_title,
                        "content": title,  # Store full content separately
                        "path": number,
                        "start_line": -1,  # Table entries don't have line numbers
                        "level": 1 if "." not in number else len(number.split(".")) + 1,
                    }
                )

        # Additional patterns for scanned/OCR text (plain text patterns)
        if not chapters:  # Only try these if no structured chapters found
            # Look for numbered chapters (1. Chapter Title, 2. Chapter Title, etc.)
            chapter_patterns = [
                r"^\s*(\d+)\.?\s+(.+)$",  # 1. Chapter Title or 1 Chapter Title
                r"^\s*Kapitel\s*(\d+):?\s*(.+)$",  # Kapitel 1: Title (German)
                r"^\s*Chapter\s*(\d+):?\s*(.+)$",  # Chapter 1: Title (English)
                r"^\s*(\d+)\s+(.+)$",  # 1 Title (simple numbered)
            ]

            for pattern in chapter_patterns:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        number = match.group(1)
                        title = match.group(2).strip()

                        # Skip very short titles or common false positives
                        if len(title) < 3 or title.lower() in [
                            "page",
                            "seiten",
                            "chapter",
                            "kapitel",
                        ]:
                            continue

                        # Avoid duplicates
                        if not any(ch["title"] == title[:255] for ch in chapters):
                            chapters.append(
                                {
                                    "title": title[:255],
                                    "content": title,
                                    "path": number,
                                    "start_line": i,
                                    "level": 1,
                                }
                            )

            # Look for lines that look like chapter titles (capitalized, standalone)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Skip lines that are too short or too long
                if len(line) < 5 or len(line) > 100:
                    continue

                # Look for patterns that suggest chapter titles
                if (
                    line[0].isupper()  # Starts with capital
                    and not line.endswith(".")  # Not a sentence
                    and not any(
                        char.isdigit() for char in line[:10]
                    )  # No early numbers
                    and sum(1 for c in line if c.isupper()) / len(line) < 0.5
                ):  # Not ALL CAPS
                    # Additional heuristics for chapter-like content
                    chapter_keywords = [
                        "introduction",
                        "grundlagen",
                        "einführung",
                        "overview",
                        "grundlagen",
                        "theorie",
                        "praxis",
                        "anwendung",
                        "methoden",
                        "algorithmen",
                    ]
                    if any(keyword in line.lower() for keyword in chapter_keywords):
                        if not any(ch["title"] == line[:255] for ch in chapters):
                            chapters.append(
                                {
                                    "title": line[:255],
                                    "content": line,
                                    "path": f"section_{len(chapters) + 1}",
                                    "start_line": i,
                                    "level": 1,
                                }
                            )

        # For scanned PDFs with no clear chapter structure, create synthetic chapters
        # based on document length and common technical content
        if (
            not chapters and len(content) > 1000
        ):  # Any reasonable content but no chapters found
            logger.info(
                "No chapters found in scanned PDF, creating synthetic chapters for technical content"
            )

            # Try to use advanced structure analysis if available
            try:
                from src.data.loader import AdvancedDocumentProcessor

                processor = AdvancedDocumentProcessor()
                structure_analysis = processor._analyze_document_structure(content)

                if structure_analysis.get("sections"):
                    # Use AI-analyzed structure
                    for section in structure_analysis["sections"]:
                        chapters.append(
                            {
                                "title": section.get(
                                    "title", f"Chapter {len(chapters) + 1}"
                                )[:255],
                                "content": section.get("title", ""),
                                "path": str(len(chapters) + 1),
                                "start_line": section.get("start_line", 0),
                                "level": section.get("level", 1),
                            }
                        )
                    logger.info(
                        f"Created {len(chapters)} AI-analyzed chapters for scanned PDF"
                    )
                else:
                    # Fallback to synthetic chapters
                    synthetic_chapters = [
                        "Einführung und Grundlagen",
                        "Mathematische Grundlagen",
                        "Kernkonzepte und Architekturen",
                        "Training und Optimierung",
                        "Erweiterte Techniken",
                        "Praktische Anwendungen",
                        "Fallstudien und Beispiele",
                        "Best Practices und Tipps",
                        "Troubleshooting und Debugging",
                        "Performance und Optimierung",
                        "Deployment und Produktion",
                        "Zukunftsaussichten",
                    ]

                    # Create chapters distributed throughout the document
                    content_length = len(content)
                    chapter_count = min(
                        len(synthetic_chapters), max(6, content_length // 8000)
                    )  # 1 chapter per ~8k chars

                    for i in range(chapter_count):
                        chapters.append(
                            {
                                "title": synthetic_chapters[i][:255],
                                "content": synthetic_chapters[i],
                                "path": str(i + 1),
                                "start_line": (i * content_length) // chapter_count,
                                "level": 1,
                            }
                        )

                    logger.info(
                        f"Created {len(chapters)} synthetic chapters for scanned PDF"
                    )

            except Exception as e:
                logger.warning(
                    f"Advanced structure analysis failed, using basic synthetic chapters: {e}"
                )

                # Basic fallback
                synthetic_chapters = [
                    "Einführung und Grundlagen",
                    "Mathematische Grundlagen",
                    "Kernkonzepte und Architekturen",
                    "Training und Optimierung",
                    "Erweiterte Techniken",
                    "Praktische Anwendungen",
                ]

                content_length = len(content)
                chapter_count = min(
                    len(synthetic_chapters), max(3, content_length // 10000)
                )

                for i in range(chapter_count):
                    chapters.append(
                        {
                            "title": synthetic_chapters[i][:255],
                            "content": synthetic_chapters[i],
                            "path": str(i + 1),
                            "start_line": (i * content_length) // chapter_count,
                            "level": 1,
                        }
                    )

                logger.info(
                    f"Created {len(chapters)} basic synthetic chapters for scanned PDF"
                )

        return chapters

    def _check_memory_usage(self, memory_limit_mb: int = 512) -> bool:
        """Check if current memory usage is within limits."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < memory_limit_mb
        except ImportError:
            # psutil not available, skip memory check
            return True

    def _process_document_streaming(
        self,
        file_path: str,
        document_id: int,
        memory_limit_mb: int = 512,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process document with streaming/chunked approach for large files.

        Args:
            file_path: Path to the document file
            document_id: Database ID of the document
            memory_limit_mb: Memory limit in MB
            progress_callback: Optional progress callback

        Returns:
            Processing results dictionary
        """
        import gc
        from src.data.loader import split_documents

        if progress_callback:
            progress_callback(os.path.basename(file_path), 10, "Extracting content...")

        # Extract content with memory monitoring
        try:
            content = split_documents([file_path])
            content_length = len(content)

            if not self._check_memory_usage(memory_limit_mb):
                logger.warning(
                    f"High memory usage detected during content extraction for {file_path}"
                )
                gc.collect()

        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return {"error": f"Content extraction failed: {e}"}

        if progress_callback:
            progress_callback(os.path.basename(file_path), 20, "Detecting language...")

        # Language detection
        language = self._detect_language_from_content(content)

        if progress_callback:
            progress_callback(os.path.basename(file_path), 30, "Detecting chapters...")

        # Chapter detection
        all_chapters = self._detect_all_chapters(content)

        if progress_callback:
            progress_callback(os.path.basename(file_path), 40, "Creating chunks...")

        # Create chunks with memory monitoring
        try:
            chunks = self._create_chunks(content, document_id, all_chapters)

            if not self._check_memory_usage(memory_limit_mb):
                logger.warning(
                    f"High memory usage detected during chunking for {file_path}"
                )
                gc.collect()

        except Exception as e:
            logger.error(f"Failed to create chunks for {file_path}: {e}")
            return {"error": f"Chunking failed: {e}"}

        # Store chapters in database (batched)
        if all_chapters:
            if progress_callback:
                progress_callback(
                    os.path.basename(file_path),
                    50,
                    f"Storing {len(all_chapters)} chapters...",
                )

            try:
                batch_size = 50  # Process chapters in batches
                for i in range(0, len(all_chapters), batch_size):
                    batch = all_chapters[i : i + batch_size]
                    for chapter in batch:
                        chapter_record = DocumentChapter(
                            document_id=document_id,
                            chapter_title=chapter[
                                "title"
                            ],  # Short title (max 255 chars)
                            chapter_path=chapter["path"],
                            level=chapter.get("level", 1),
                            word_count=len(chapter["title"].split()),
                            content=chapter.get(
                                "content", chapter["title"]
                            ),  # Full content
                        )
                        self.db.add(chapter_record)
                    self.db.commit()  # Commit batch

                    if not self._check_memory_usage(memory_limit_mb):
                        logger.warning(
                            f"High memory usage during chapter storage for {file_path}"
                        )
                        gc.collect()

            except Exception as e:
                logger.error(f"Failed to store chapters for {file_path}: {e}")
                self.db.rollback()
                return {"error": f"Chapter storage failed: {e}"}

        # Process chunks and embeddings in batches
        total_embeddings_created = 0
        if chunks:
            if progress_callback:
                progress_callback(
                    os.path.basename(file_path),
                    70,
                    f"Processing {len(chunks)} chunks...",
                )

            try:
                # Process embeddings in smaller batches to manage memory
                batch_size = 20

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    chunk_texts = [chunk["content"] for chunk in batch]

                    if progress_callback:
                        progress_callback(
                            os.path.basename(file_path),
                            75 + int((i / len(chunks)) * 20),
                            f"Creating embeddings for batch {i // batch_size + 1}...",
                        )

                    # Create embeddings for this batch
                    embeddings_array, _ = create_embeddings(chunk_texts)

                    # Store chunks and embeddings
                    for j, (chunk_data, embedding) in enumerate(
                        zip(batch, embeddings_array)
                    ):
                        # Create chunk record
                        chunk = DocumentChunk(
                            document_id=document_id,
                            content=chunk_data["content"],
                            chunk_index=chunk_data["metadata"]["chunk_index"],
                            chapter_path=chunk_data["metadata"].get("chapter_path"),
                            chapter_title=chunk_data["metadata"].get("chapter_title"),
                        )
                        self.db.add(chunk)
                        self.db.flush()  # Get chunk ID

                        # Create embedding record
                        doc_embedding = DocumentEmbedding(
                            chunk_id=chunk.id,
                            embedding=embedding.tolist(),
                            embedding_model="nomic-ai/nomic-embed-text-v1.5",
                        )
                        self.db.add(doc_embedding)
                        total_embeddings_created += 1

                    # Commit batch
                    self.db.commit()

                    # Memory check and cleanup
                    if not self._check_memory_usage(memory_limit_mb):
                        logger.warning(
                            f"High memory usage during embedding creation for {file_path}"
                        )
                        gc.collect()

            except Exception as e:
                logger.error(
                    f"Failed to process chunks/embeddings for {file_path}: {e}"
                )
                self.db.rollback()
                return {"error": f"Chunk/embedding processing failed: {e}"}

        # Index in search systems
        if progress_callback:
            progress_callback(os.path.basename(file_path), 95, "Indexing for search...")

        try:
            self._index_document_chunks(document_id)
        except Exception as e:
            logger.warning(f"Search indexing failed for {file_path}: {e}")
            # Don't fail the whole process for indexing issues

        if progress_callback:
            progress_callback(os.path.basename(file_path), 100, "Processing complete!")

        return {
            "success": True,
            "chunks_created": len(chunks),
            "chapters_created": len(all_chapters),
            "embeddings_created": total_embeddings_created,
            "language": language,
            "content_length": content_length,
        }

    def _index_document_chunks(self, document_id: int) -> None:
        """
        Index document chunks in search systems.

        Args:
            document_id: Database ID of the document
        """
        try:
            # Get document and chunks
            document = (
                self.db.query(Document).filter(Document.id == document_id).first()
            )
            if not document:
                logger.warning(f"Document {document_id} not found for indexing")
                return

            chunks = (
                self.db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .all()
            )

            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return

            # Get embeddings for chunks
            chunk_ids = [chunk.id for chunk in chunks]
            embeddings = (
                self.db.query(DocumentEmbedding)
                .filter(DocumentEmbedding.chunk_id.in_(chunk_ids))
                .all()
            )

            # Create embedding lookup
            embedding_lookup = {emb.chunk_id: emb.embedding for emb in embeddings}

            # Prepare data for indexing
            chunk_data = []
            embeddings_list = []

            for chunk in chunks:
                chunk_dict = {
                    "content": chunk.content,
                    "metadata": {
                        "word_count": len(chunk.content.split()),
                        "char_count": len(chunk.content),
                        "chapter_title": chunk.chapter_title,
                        "chapter_path": chunk.chapter_path,
                    },
                }
                chunk_data.append(chunk_dict)
                embeddings_list.append(embedding_lookup.get(chunk.id, []))

            # Index in search systems
            self._index_document(document, chunk_data, embeddings_list)

        except Exception as e:
            logger.error(f"Failed to index document chunks for {document_id}: {e}")
            raise

    def _create_chunks(
        self,
        content: str,
        document_id: int,
        chapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Create intelligent chunks from document content with semantic awareness."""
        # Enhanced chunking with semantic awareness
        chunk_size = 1000
        overlap = 200

        chunks = []
        start = 0
        chunk_index = 0

        # Pre-process content for better chunking
        content = self._preprocess_content(content)

        while start < len(content):
            end = start + chunk_size
            chunk_content = content[start:end]

            # Find semantic boundary for better chunks
            if end < len(content):
                end = self._find_semantic_boundary(content, start, end, chunk_size)

            # Extract chunk with metadata
            chunk_text = content[start:end]
            chunk_metadata = self._extract_chunk_metadata(
                chunk_text, chunk_index, start, end, chapters
            )

            chunk = {
                "content": chunk_text,
                "metadata": chunk_metadata,
            }
            chunks.append(chunk)

            # Adaptive overlap based on content type
            adaptive_overlap = self._calculate_adaptive_overlap(chunk_text, overlap)
            start = end - adaptive_overlap
            chunk_index += 1

        return chunks

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for better chunking."""
        # Normalize whitespace
        content = re.sub(r"\s+", " ", content.strip())

        # Handle code blocks and special formatting
        # This is a basic implementation - could be enhanced with more sophisticated preprocessing

        return content

    def _find_semantic_boundary(
        self, content: str, start: int, end: int, chunk_size: int
    ) -> int:
        """Find the best semantic boundary for chunking."""
        chunk_content = content[start:end]

        # Priority order for boundaries
        boundary_markers = [
            "\n\n",  # Paragraph breaks
            ". ",  # Sentence endings
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "\n",  # Line breaks
            "; ",  # Semicolon
            ": ",  # Colon
        ]

        best_end = end
        for marker in boundary_markers:
            pos = chunk_content.rfind(marker)
            if pos > chunk_size * 0.6:  # Don't break too early (60% of chunk size)
                best_end = start + pos + len(marker)
                break

        return min(best_end, len(content))  # Don't exceed content length

    def _extract_chunk_metadata(
        self,
        chunk_text: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        chapters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Extract metadata for a chunk."""
        metadata = {
            "chunk_index": chunk_index,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
        }

        # Detect if chunk contains code-like content
        code_indicators = [
            "```",
            "def ",
            "class ",
            "import ",
            "function",
            "var ",
            "const ",
        ]
        metadata["contains_code"] = any(
            indicator in chunk_text for indicator in code_indicators
        )

        # Detect if chunk contains lists or structured content
        list_indicators = ["• ", "- ", "* ", "1. ", "2. ", "3. "]
        metadata["contains_list"] = any(
            indicator in chunk_text for indicator in list_indicators
        )

        # Assign chapter/section information from pre-detected chapters
        if chapters:
            # Find the chapter that this chunk belongs to based on position
            chunk_center = (start_pos + end_pos) // 2
            for chapter in chapters:
                # Simple assignment: if chunk contains chapter title or is near chapter position
                if chapter.get("title") and chapter["title"] in chunk_text:
                    metadata["chapter_title"] = chapter["title"]
                    metadata["chapter_path"] = chapter["path"]
                    break

        return metadata

    def _detect_chapter_info(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Detect chapter/section information in chunk text."""
        lines = chunk_text.split("\n")

        # Look for chapter/section headers - simplified approach
        # Find ## headers
        if "##" in chunk_text:
            # Extract text after ## until next ## or end
            parts = chunk_text.split("##")
            for i, part in enumerate(parts[1:], 1):  # Skip first part (before first ##)
                # Get the header text (first line or reasonable chunk)
                lines = part.strip().split("\n", 1)
                header_text = lines[0].strip()

                # Clean up the header
                header_text = re.sub(r"[^\w\s\-.,()]", "", header_text).strip()
                header_text = re.sub(r"\s+", " ", header_text).strip()

                if 5 <= len(header_text) <= 100:
                    # Determine path based on header content
                    if header_text.startswith(("1.", "2.", "3.", "4.", "5.")):
                        # Numbered section
                        path = header_text.split()[0] if header_text.split() else str(i)
                    else:
                        path = str(i)

                    return header_text, path

        # Look for table format | number | title |
        table_matches = re.findall(
            r"\\|\\s*(\\d+(?:\\.\\d+)*)\\s*\\|\\s*([^\\|]+?)\\s*\\|", chunk_text
        )
        if table_matches:
            for number, title in table_matches:
                title = title.strip()
                title = re.sub(r"[^\w\s\-.,()]", "", title).strip()
                if 5 <= len(title) <= 100:
                    return title, number

        # Simplified chapter detection
        # Find ## headers
        if "##" in chunk_text:
            # Extract text after ## until next ## or end
            parts = chunk_text.split("##")
            for i, part in enumerate(parts[1:], 1):  # Skip first part (before first ##)
                # Get the header text (first meaningful chunk)
                header_part = part.strip().split("\n", 1)[0]  # Take first line
                # Take reasonable amount of text from the beginning
                header_text = header_part[:60]  # Shorter limit

                # Clean up the header first
                header_text = re.sub(r"[^\w\s\-.,()]", "", header_text).strip()
                header_text = re.sub(r"\s+", " ", header_text).strip()

                # Take first few words that form a reasonable title
                words = header_text.split()
                if words:
                    # If it starts with a number, take number + next few meaningful words
                    if words[0].replace(".", "").isdigit():
                        # For numbered sections, take the number + up to 4 more words, but avoid duplicates
                        title_words = []
                        seen = set()
                        for word in words[:5]:  # Take up to 5 words
                            if word not in seen:
                                title_words.append(word)
                                seen.add(word)
                            if len(title_words) >= 4:  # Limit to 4 unique words
                                break
                    else:
                        title_words = words[
                            : min(3, len(words))
                        ]  # Take up to 3 words for other headers

                    header_text = " ".join(title_words)

                if 3 <= len(header_text) <= 60:  # Reasonable title length
                    # Determine path based on header content
                    if header_text.startswith(("1.", "2.", "3.", "4.", "5.")):
                        # Numbered section
                        path = header_text.split()[0] if header_text.split() else str(i)
                    else:
                        path = str(i)

                    return header_text, path

        # Look for table format | number | title |
        table_matches = re.findall(
            r"\\|\\s*(\\d+(?:\\.\\d+)*)\\s*\\|\\s*([^\\|]+?)\\s*\\|", chunk_text
        )
        if table_matches:
            for number, title in table_matches:
                title = title.strip()
                title = re.sub(r"[^\w\s\-.,()]", "", title).strip()
                if 5 <= len(title) <= 100:
                    return title, number

        return None, None

    def _calculate_adaptive_overlap(self, chunk_text: str, base_overlap: int) -> int:
        """Calculate adaptive overlap based on chunk content characteristics."""
        overlap = base_overlap

        # Increase overlap for code content to maintain context
        if any(
            indicator in chunk_text
            for indicator in ["```", "def ", "class ", "function"]
        ):
            overlap = int(base_overlap * 1.5)

        # Increase overlap for complex technical content
        technical_terms = [
            "algorithm",
            "implementation",
            "architecture",
            "framework",
            "system",
        ]
        if any(term in chunk_text.lower() for term in technical_terms):
            overlap = int(base_overlap * 1.3)

        # Decrease overlap for simple content
        if len(chunk_text.split()) < 50:  # Short chunks
            overlap = int(base_overlap * 0.7)

        return max(50, min(overlap, base_overlap * 2))  # Keep within reasonable bounds

    def _index_document(
        self,
        document: Document,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        """Index document in Elasticsearch with enhanced metadata."""
        try:
            es_client = get_elasticsearch_client()

            # Index each chunk with enhanced metadata
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = chunk.get("metadata", {})

                doc = {
                    "document_id": document.id,
                    "content": chunk["content"],
                    "embedding": embedding,
                    "chunk_index": i,
                    "filename": document.filename,
                    "language": document.detected_language,
                    "metadata": metadata,
                    # Enhanced fields for better search
                    "word_count": metadata.get("word_count", 0),
                    "char_count": metadata.get("char_count", 0),
                    "contains_code": metadata.get("contains_code", False),
                    "contains_list": metadata.get("contains_list", False),
                    "avg_words_per_sentence": metadata.get("avg_words_per_sentence", 0),
                    "chapter_title": chunk.get("chapter_title"),
                    "chapter_path": chunk.get("chapter_path"),
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
        use_advanced_processing: bool = None,
    ) -> Dict[str, Any]:
        """
        Process a single file with optional reprocessing capabilities.

        Args:
            file_path: Path to the file to process
            filename: Optional filename (used for database lookup)
            file_hash: Optional file hash (used for database lookup)
            force_enrichment: Whether to force AI enrichment even if document exists
            use_advanced_processing: Whether to use advanced AI processing (auto-detected for scanned PDFs)

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

            # Handle reprocessing of existing documents differently
            if existing_doc and force_enrichment:
                # For reprocessing, run advanced processing directly without trying to create a new document
                from src.data.loader import AdvancedDocumentProcessor

                advanced_processor = AdvancedDocumentProcessor()
                processing_result = advanced_processor.process_document_comprehensive(
                    file_path
                )

                # Check if processing was successful
                if processing_result is None:
                    logger.error(
                        f"Document processing failed for {file_path} - received None result"
                    )
                    return {
                        "success": False,
                        "filename": filename,
                        "error": "Document processing returned no results",
                        "document_id": existing_doc.id,
                    }

                # Update the existing document with fresh processing results
                result = self.reprocess_existing_document(
                    existing_doc, processing_result, file_path
                )
                return result

            # For new documents, proceed with normal processing
            # Determine if we should use advanced processing
            if use_advanced_processing is None:
                # Auto-detect scanned PDFs and enable advanced processing for them
                if file_path.lower().endswith(".pdf"):
                    from src.data.loader import is_scanned_pdf

                    is_scanned = is_scanned_pdf(file_path)
                    use_advanced_processing = is_scanned
                    logger.info(
                        f"Auto-detected scanned PDF: {is_scanned}, using advanced processing: {use_advanced_processing}"
                    )
                else:
                    use_advanced_processing = False

            # Process the document
            processor = DocumentProcessor()
            result = processor.process_document(
                file_path, filename, use_advanced_processing=use_advanced_processing
            )

            if result.get("success"):
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
            file_path: Path to the document file

        Returns:
            Dict with reprocessing results
        """
        # Validate processing_result
        if processing_result is None:
            logger.error(
                f"Cannot reprocess document {existing_doc.id} - processing_result is None"
            )
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": "Processing result is None",
                "document_id": existing_doc.id,
                "chunks_created": 0,
                "chapters_created": 0,
            }

        if not isinstance(processing_result, dict):
            logger.error(
                f"Cannot reprocess document {existing_doc.id} - processing_result is not a dict: {type(processing_result)}"
            )
            return {
                "success": False,
                "filename": existing_doc.filename,
                "error": f"Invalid processing result type: {type(processing_result)}",
                "document_id": existing_doc.id,
                "chunks_created": 0,
                "chapters_created": 0,
            }

        try:
            # Validate content quality before processing
            extracted_content = processing_result.get("extracted_content", "")
            chunks = processing_result.get("chunks", [])

            # Debug: Check processing result structure
            logger.debug(f"Processing result keys: {list(processing_result.keys())}")
            logger.debug(
                f"extracted_content type: {type(extracted_content)}, value preview: {str(extracted_content)[:100]}"
            )
            logger.debug(
                f"chunks type: {type(chunks)}, length: {len(chunks) if isinstance(chunks, list) else 'N/A'}"
            )

            if isinstance(extracted_content, dict):
                logger.error(f"extracted_content is a dict: {extracted_content}")
                extracted_content = str(
                    extracted_content
                )  # Convert to string as fallback

            # Ensure chunks is a list of strings
            if isinstance(chunks, list):
                logger.debug(f"Processing {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:3]):  # Debug first 3 chunks
                    logger.debug(
                        f"Chunk {i}: type={type(chunk)}, keys={chunk.keys() if isinstance(chunk, dict) else 'N/A'}"
                    )
                chunks = [
                    str(chunk) if not isinstance(chunk, str) else chunk
                    for chunk in chunks
                ]
            else:
                logger.warning(f"Chunks is not a list: {type(chunks)}")
                chunks = []

            validation_result = ContentValidator.validate_content_quality(
                extracted_content, chunks
            )

            # Log validation results and handle quality issues
            if not validation_result["is_valid"]:
                logger.warning(
                    f"Content quality issues detected for document {existing_doc.id}: "
                    f"{validation_result['issues']}"
                )
                logger.info(f"Quality score: {validation_result['quality_score']:.2f}")

                # Add quality issues to processing result for user feedback
                processing_result["quality_issues"] = validation_result["issues"]
                processing_result["quality_score"] = validation_result["quality_score"]
                processing_result["quality_recommendations"] = validation_result[
                    "recommendations"
                ]

                # Log recommendations for debugging
                if validation_result["recommendations"]:
                    logger.info(
                        f"Quality improvement recommendations: {validation_result['recommendations']}"
                    )

            # Update document metadata
            existing_doc.last_modified = datetime.now()

            # Set status based on content quality
            if validation_result["is_valid"]:
                existing_doc.status = "processed"
            else:
                existing_doc.status = "needs_review"  # New status for quality issues

            # Regenerate AI content for reprocessing
            extracted_content = processing_result.get("extracted_content", "")
            if extracted_content and isinstance(extracted_content, str):
                # Use DocumentProcessor for AI content regeneration
                doc_processor = DocumentProcessor()

                # Generate fresh AI content
                content_for_analysis = extracted_content[:2000]

                # Generate new tags
                suggested_tags_data = doc_processor.tag_suggester.suggest_tags(
                    content_for_analysis, existing_doc.filename
                )
                suggested_tags = [
                    tag.get("tag", "") for tag in suggested_tags_data if tag.get("tag")
                ]

                # Generate new categories
                suggested_categories = doc_processor._suggest_categories_ai(
                    content_for_analysis, existing_doc.filename, suggested_tags
                )

                # Generate new summary
                document_summary = doc_processor._generate_document_summary(
                    content_for_analysis,
                    existing_doc.filename,
                    suggested_tags,
                    processing_result.get("chapters_detected", 0),
                )

                # Update AI-enriched fields
                existing_doc.document_summary = document_summary
                existing_doc.key_topics = suggested_tags[:5]  # Store as key topics

                # Clear existing tags and categories
                self.db.query(DocumentTagAssignment).filter(
                    DocumentTagAssignment.document_id == existing_doc.id
                ).delete()
                self.db.query(DocumentCategoryAssignment).filter(
                    DocumentCategoryAssignment.document_id == existing_doc.id
                ).delete()

                # Add new tags
                for tag_name in suggested_tags:
                    if tag_name:
                        tag = doc_processor.tag_manager.get_tag_by_name(tag_name)
                        if not tag:
                            tag = doc_processor.tag_manager.create_tag(tag_name)
                        doc_processor.tag_manager.add_tag_to_document(
                            existing_doc.id, tag.id
                        )

                # Add new categories
                for category_name in suggested_categories:
                    if category_name:
                        category = doc_processor.category_manager.get_category_by_name(
                            category_name
                        )
                        if not category:
                            category = doc_processor.category_manager.create_category(
                                category_name
                            )
                        doc_processor.category_manager.add_category_to_document(
                            existing_doc.id, category.id
                        )

            # Update other fields if available
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

            # Clear existing chunks, chapters, and embeddings for this document
            # First, get existing chunk IDs to clear embeddings
            existing_chunk_ids = [
                chunk.id
                for chunk in self.db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == existing_doc.id)
                .all()
            ]

            # Clear embeddings first (before deleting chunks due to foreign key constraints)
            if existing_chunk_ids:
                self.db.query(DocumentEmbedding).filter(
                    DocumentEmbedding.chunk_id.in_(existing_chunk_ids)
                ).delete()

            # Clear existing chunks and chapters
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing_doc.id
            ).delete()
            self.db.query(DocumentChapter).filter(
                DocumentChapter.document_id == existing_doc.id
            ).delete()

            # Add new chunks and chapters
            chunks_created = 0
            chapters_created = 0

            if (
                "chunks" in processing_result
                and processing_result["chunks"] is not None
            ):
                chunks_list = processing_result["chunks"]
                if isinstance(chunks_list, list):
                    for i, chunk_data in enumerate(chunks_list):
                        if isinstance(chunk_data, dict) and "content" in chunk_data:
                            # Ensure content is a string
                            content = chunk_data["content"]
                            if isinstance(content, dict):
                                logger.warning(
                                    f"Chunk content is a dict, converting to string: {content}"
                                )
                                content = str(content)
                            elif not isinstance(content, str):
                                content = str(content)

                            chunk = DocumentChunk(
                                document_id=existing_doc.id,
                                content=content,
                                chunk_index=i,  # Use sequential index starting from 0
                                embedding_model="nomic-ai/nomic-embed-text-v1.5",
                                chapter_path=chunk_data.get("chapter_path"),
                                chapter_title=chunk_data.get("chapter_title"),
                            )
                            self.db.add(chunk)
                            chunks_created += 1

            if (
                "chapters" in processing_result
                and processing_result["chapters"] is not None
            ):
                chapters_list = processing_result["chapters"]
                if isinstance(chapters_list, list):
                    for i, chapter_data in enumerate(chapters_list):
                        if isinstance(chapter_data, dict) and "title" in chapter_data:
                            chapter = DocumentChapter(
                                document_id=existing_doc.id,
                                chapter_title=chapter_data["title"],
                                chapter_path=chapter_data.get(
                                    "path", f"chapter_{i + 1}"
                                ),
                                level=chapter_data.get("level", 1),
                                word_count=chapter_data.get(
                                    "word_count", len(chapter_data["title"].split())
                                ),
                                content=chapter_data.get(
                                    "content",
                                    chapter_data.get(
                                        "content_preview", chapter_data["title"]
                                    ),
                                ),
                            )
                    self.db.add(chapter)
                    chapters_created += 1

            # Commit chunks and chapters before creating embeddings
            self.db.commit()

            # Create embeddings for new chunks in batches to avoid memory issues
            if chunks_created > 0:
                # Get the newly created chunks
                chunks = (
                    self.db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == existing_doc.id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )

                # Create embeddings in batches to avoid memory issues
                embedding_batch_size = 100  # Process 100 chunks at a time
                total_chunks = len(chunks)

                logger.info(
                    f"Creating embeddings for {total_chunks} chunks in batches of {embedding_batch_size}"
                )

                for batch_start in range(0, total_chunks, embedding_batch_size):
                    batch_end = min(batch_start + embedding_batch_size, total_chunks)
                    batch_chunks = chunks[batch_start:batch_end]
                    batch_texts = [chunk.content for chunk in batch_chunks]

                    logger.debug(
                        f"Processing embedding batch {batch_start // embedding_batch_size + 1}/{(total_chunks + embedding_batch_size - 1) // embedding_batch_size}"
                    )

                    try:
                        # Create embeddings for this batch
                        embeddings_array, _ = create_embeddings(batch_texts)

                        # Store embeddings if creation was successful
                        if embeddings_array is not None and len(embeddings_array) > 0:
                            for chunk, embedding in zip(batch_chunks, embeddings_array):
                                # Check if embedding is valid (not None and not all zeros)
                                if (
                                    embedding is not None
                                    and hasattr(embedding, "shape")
                                    and embedding.shape[0] > 0
                                ):
                                    try:
                                        doc_embedding = DocumentEmbedding(
                                            chunk_id=chunk.id,
                                            embedding=embedding.tolist(),
                                            embedding_model="nomic-ai/nomic-embed-text-v1.5",
                                        )
                                        self.db.add(doc_embedding)
                                    except Exception as embed_error:
                                        logger.warning(
                                            f"Failed to store embedding for chunk {chunk.id}: {embed_error}"
                                        )
                                        continue
                        else:
                            logger.warning(
                                f"Failed to create embeddings for batch {batch_start // embedding_batch_size + 1}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Embedding creation failed for batch {batch_start // embedding_batch_size + 1}: {e}"
                        )
                        # Continue with next batch

                logger.info(
                    f"Completed embedding creation for document {existing_doc.id}"
                )

                # Commit embeddings to database before indexing
                self.db.commit()

                # Update search index with enhanced metadata
                # Get all embeddings for this document
                all_embeddings = []
                for chunk in chunks:
                    embedding_record = (
                        self.db.query(DocumentEmbedding)
                        .filter(DocumentEmbedding.chunk_id == chunk.id)
                        .first()
                    )
                    if embedding_record:
                        all_embeddings.append(embedding_record.embedding)

                if all_embeddings and len(all_embeddings) == len(chunks):
                    logger.info(
                        f"Indexing {len(all_embeddings)} chunks in Elasticsearch"
                    )
                    try:
                        processor = DocumentProcessor()
                        processor._index_document(
                            existing_doc,
                            [
                                {
                                    "content": chunk.content,
                                    "metadata": {
                                        "word_count": len(chunk.content.split())
                                    },
                                }
                                for chunk in chunks
                            ],
                            all_embeddings,
                        )
                        logger.info(
                            f"Successfully indexed document {existing_doc.id} in search"
                        )
                    except Exception as index_error:
                        logger.error(
                            f"Failed to index document {existing_doc.id} in search: {index_error}"
                        )
                else:
                    logger.warning(
                        f"Embeddings mismatch for document {existing_doc.id}: {len(all_embeddings)} embeddings vs {len(chunks)} chunks - skipping search indexing"
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

            # Add chunks and chapters with enhanced metadata
            chunks_created = 0
            chapters_created = 0

            if "chunks" in processing_result:
                for chunk_data in processing_result["chunks"]:
                    metadata = chunk_data.get("metadata", {})
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        content=chunk_data["content"],
                        chunk_index=chunk_data["chunk_index"],
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

                # Update search index with enhanced metadata
                # Get all embeddings for this document
                all_embeddings = []
                for chunk in chunks:
                    embedding_record = (
                        self.db.query(DocumentEmbedding)
                        .filter(DocumentEmbedding.chunk_id == chunk.id)
                        .first()
                    )
                    if embedding_record:
                        all_embeddings.append(embedding_record.embedding)

                if all_embeddings and len(all_embeddings) == len(chunks):
                    logger.info(
                        f"Indexing {len(all_embeddings)} chunks in Elasticsearch"
                    )

                    # Debug: Check what content we're indexing
                    chunk_list = []
                    for i, chunk in enumerate(chunks[:2]):  # Check first 2 chunks
                        content = chunk.content
                        logger.info(
                            f"Indexing chunk {i} content preview: {content[:150]}..."
                        )
                        chunk_list.append(
                            {
                                "content": content,
                                "metadata": {"word_count": len(content.split())},
                            }
                        )

                    for chunk in chunks[2:]:  # Rest of chunks
                        chunk_list.append(
                            {
                                "content": chunk.content,
                                "metadata": {"word_count": len(chunk.content.split())},
                            }
                        )

                    try:
                        processor = DocumentProcessor()
                        processor._index_document(
                            doc,
                            chunk_list,
                            all_embeddings,
                        )
                        logger.info(f"Successfully indexed document {doc.id} in search")
                    except Exception as index_error:
                        logger.error(
                            f"Failed to index document {doc.id} in search: {index_error}"
                        )
                else:
                    logger.warning(
                        f"Embeddings mismatch for document {doc.id}: {len(all_embeddings)} embeddings vs {len(chunks)} chunks - skipping search indexing"
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
            # Auto-detect scanned PDFs and enable advanced processing
            use_advanced_processing = False
            if file_path.lower().endswith(".pdf"):
                from src.data.loader import is_scanned_pdf

                is_scanned = is_scanned_pdf(file_path)
                use_advanced_processing = is_scanned
                logger.info(
                    f"Auto-detected scanned PDF for parallel processing: {is_scanned}, using advanced processing: {use_advanced_processing}"
                )

            processor = DocumentProcessor()
            result = processor.process_document(
                file_path, use_advanced_processing=use_advanced_processing
            )
            result["file_path"] = file_path
            return result
        except Exception as e:
            return {"file_path": file_path, "success": False, "error": str(e)}

    def upload_files(
        self,
        uploaded_files,
        data_dir="data",
        use_parallel=False,
        max_workers=4,
        enable_streaming=True,
        memory_limit_mb=512,
        large_file_threshold_mb=50,
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
