#!/usr/bin/env python3
"""
Lightweight document management facade for LocalRAG.

Provides a unified interface to the extracted document management components.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core.categorization.category_manager import CategoryManager
from src.core.processing.document_processor import DocumentProcessor
from src.core.tagging.tag_manager import TagManager
from src.database.models import DocumentCategory, DocumentTag, SessionLocal

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Lightweight facade for document management operations.

    Provides a unified interface to the extracted document management components.
    """

    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self.tag_manager = TagManager(self.db)
        self.category_manager = CategoryManager(self.db)
        self.document_processor = DocumentProcessor(self.db)

    # Delegate tag operations
    def get_tag_by_name(self, name: str) -> Optional[DocumentTag]:
        """Get a tag by name."""
        return self.tag_manager.get_tag_by_name(name)

    def create_tag(self, name: str, color: Optional[str] = None) -> DocumentTag:
        """Create a new tag with optional color."""
        return self.tag_manager.create_tag(name, color)

    def add_tag_to_document(self, document_id: int, tag_id: int) -> bool:
        """Add a tag to a document."""
        return self.tag_manager.add_tag_to_document(document_id, tag_id)

    def remove_tag_from_document(self, document_id: int, tag_id: int) -> bool:
        """Remove a tag from a document."""
        return self.tag_manager.remove_tag_from_document(document_id, tag_id)

    def get_document_tags(self, document_id: int) -> List[DocumentTag]:
        """Get all tags for a document."""
        return self.tag_manager.get_document_tags(document_id)

    def suggest_tags_for_document(self, document_id: int, max_suggestions: int = 5) -> List[str]:
        """Suggest tags for a document using AI."""
        return self.tag_manager.suggest_tags_for_document(document_id, max_suggestions)

    def delete_tag(self, tag_name: str) -> bool:
        """Delete a tag and all its assignments."""
        return self.tag_manager.delete_tag(tag_name)

    def get_all_tags(self) -> List[DocumentTag]:
        """Get all tags."""
        return self.tag_manager.get_all_tags()

    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most popular tags by usage count."""
        return self.tag_manager.get_popular_tags(limit)

    # Delegate category operations
    def get_category_by_name(
        self, name: str, parent_id: Optional[int] = None
    ) -> Optional[DocumentCategory]:
        """Get a category by name and optional parent."""
        return self.category_manager.get_category_by_name(name, parent_id)

    def create_category(
        self,
        name: str,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> DocumentCategory:
        """Create a new category."""
        return self.category_manager.create_category(name, description, parent_id)

    def add_category_to_document(self, document_id: int, category_id: int) -> bool:
        """Add a category to a document."""
        return self.category_manager.add_category_to_document(document_id, category_id)

    def remove_category_from_document(self, document_id: int, category_id: int) -> bool:
        """Remove a category from a document."""
        return self.category_manager.remove_category_from_document(document_id, category_id)

    def get_document_categories(self, document_id: int) -> List[DocumentCategory]:
        """Get all categories for a document."""
        return self.category_manager.get_document_categories(document_id)

    def get_category_hierarchy(self, category_id: int) -> List[DocumentCategory]:
        """Get the full hierarchy path for a category."""
        return self.category_manager.get_category_hierarchy(category_id)

    def get_root_categories(self) -> List[DocumentCategory]:
        """Get all root categories (categories with no parent)."""
        return self.category_manager.get_root_categories()

    def get_category_tree(self) -> List[Dict[str, Any]]:
        """Get the complete category tree with hierarchy."""
        return self.category_manager.get_category_tree()

    def delete_category(self, category_id: int) -> bool:
        """Delete a category and all its subcategories."""
        return self.category_manager.delete_category(category_id)

    # Delegate document processing operations
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and return processing results."""
        return self.document_processor.process_document(file_path)
