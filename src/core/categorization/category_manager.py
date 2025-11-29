"""
Category management system for LocalRAG documents.

Handles hierarchical category management with parent-child relationships.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import (
    DocumentCategory,
    DocumentCategoryAssignment,
)

logger = logging.getLogger(__name__)


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

        assignment = DocumentCategoryAssignment(document_id=document_id, category_id=category_id)
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
        return [assignment.category for assignment in assignments if assignment.category]

    def get_category_hierarchy(self, category_id: int) -> List[DocumentCategory]:
        """Get the full hierarchy path for a category."""
        hierarchy = []
        current = self.db.query(DocumentCategory).filter(DocumentCategory.id == category_id).first()

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

        cat_stats = (
            self.db.query(
                DocumentCategory.name,
                DocumentCategory.description,
                func.count(DocumentCategoryAssignment.document_id).label("document_count"),
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
                    self.db.query(DocumentCategory).filter(DocumentCategory.id == cat_id).first()
                )
                if category:
                    self.db.delete(category)

            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete category {category_id}: {e}")
            return False
