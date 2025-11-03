"""
Document management classes for tagging and categorization.

Provides TagManager and CategoryManager classes for organizing documents
with tags and hierarchical categories.
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from .database.models import DocumentTag, DocumentCategory, Document, DocumentTagAssignment, DocumentCategoryAssignment


class TagManager:
    """
    Manager class for document tags.

    Handles CRUD operations for tags and tag-document associations.
    """

    def __init__(self, db: Session):
        self.db = db

    def create_tag(self, name: str, color: Optional[str] = None, description: Optional[str] = None) -> DocumentTag:
        """
        Create a new tag.

        Args:
            name: Tag name (must be unique)
            color: Hex color code (e.g., #FF5733)
            description: Optional description

        Returns:
            Created DocumentTag instance
        """
        tag = DocumentTag(name=name, color=color, description=description)
        self.db.add(tag)
        self.db.commit()
        self.db.refresh(tag)
        return tag

    def get_tag_by_id(self, tag_id: int) -> Optional[DocumentTag]:
        """Get tag by ID."""
        return self.db.query(DocumentTag).filter(DocumentTag.id == tag_id).first()

    def get_tag_by_name(self, name: str) -> Optional[DocumentTag]:
        """Get tag by name."""
        return self.db.query(DocumentTag).filter(DocumentTag.name == name).first()

    def get_all_tags(self) -> List[DocumentTag]:
        """Get all tags ordered by name."""
        return self.db.query(DocumentTag).order_by(DocumentTag.name).all()

    def update_tag(self, tag_id: int, name: Optional[str] = None, color: Optional[str] = None,
                   description: Optional[str] = None) -> Optional[DocumentTag]:
        """
        Update tag properties.

        Args:
            tag_id: Tag ID to update
            name: New name (must be unique if provided)
            color: New color
            description: New description

        Returns:
            Updated tag or None if not found
        """
        tag = self.get_tag_by_id(tag_id)
        if not tag:
            return None

        if name is not None:
            tag.name = name
        if color is not None:
            tag.color = color
        if description is not None:
            tag.description = description

        self.db.commit()
        self.db.refresh(tag)
        return tag

    def delete_tag(self, tag_id: int) -> bool:
        """
        Delete tag and all its associations.

        Args:
            tag_id: Tag ID to delete

        Returns:
            True if deleted, False if not found
        """
        tag = self.get_tag_by_id(tag_id)
        if not tag:
            return False

        # Remove associations first
        self.db.query(DocumentTagAssignment).filter(DocumentTagAssignment.tag_id == tag_id).delete()

        self.db.delete(tag)
        self.db.commit()
        return True

    def add_tag_to_document(self, document_id: int, tag_id: int) -> bool:
        """
        Add tag to document.

        Args:
            document_id: Document ID
            tag_id: Tag ID

        Returns:
            True if added, False if already exists or invalid IDs
        """
        # Check if association already exists
        existing = self.db.query(DocumentTagAssignment).filter(
            DocumentTagAssignment.document_id == document_id,
            DocumentTagAssignment.tag_id == tag_id
        ).first()

        if existing:
            return False

        association = DocumentTagAssignment(document_id=document_id, tag_id=tag_id)
        self.db.add(association)
        self.db.commit()
        return True

    def remove_tag_from_document(self, document_id: int, tag_id: int) -> bool:
        """
        Remove tag from document.

        Args:
            document_id: Document ID
            tag_id: Tag ID

        Returns:
            True if removed, False if not found
        """
        association = self.db.query(DocumentTagAssignment).filter(
            DocumentTagAssignment.document_id == document_id,
            DocumentTagAssignment.tag_id == tag_id
        ).first()

        if not association:
            return False

        self.db.delete(association)
        self.db.commit()
        return True

    def get_document_tags(self, document_id: int) -> List[DocumentTag]:
        """Get all tags for a document."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        return document.tags if document else []

    def get_tag_usage_stats(self) -> List[Dict[str, Any]]:
        """
        Get usage statistics for all tags.

        Returns:
            List of dicts with tag info and document count
        """
        from sqlalchemy import func

        result = self.db.query(
            DocumentTag,
            func.count(DocumentTagAssignment.document_id).label('document_count')
        ).outerjoin(DocumentTagAssignment).group_by(DocumentTag.id).all()

        return [
            {
                'id': tag.id,
                'name': tag.name,
                'color': tag.color,
                'description': tag.description,
                'document_count': count
            }
            for tag, count in result
        ]


class CategoryManager:
    """
    Manager class for document categories.

    Handles CRUD operations for categories and category-document associations.
    Supports hierarchical categories.
    """

    def __init__(self, db: Session):
        self.db = db

    def create_category(self, name: str, description: Optional[str] = None,
                       parent_id: Optional[int] = None) -> DocumentCategory:
        """
        Create a new category.

        Args:
            name: Category name
            description: Optional description
            parent_id: Parent category ID for hierarchy

        Returns:
            Created DocumentCategory instance
        """
        category = DocumentCategory(name=name, description=description, parent_id=parent_id)
        self.db.add(category)
        self.db.commit()
        self.db.refresh(category)
        return category

    def get_category_by_id(self, category_id: int) -> Optional[DocumentCategory]:
        """Get category by ID."""
        return self.db.query(DocumentCategory).filter(DocumentCategory.id == category_id).first()

    def get_category_by_name(self, name: str, parent_id: Optional[int] = None) -> Optional[DocumentCategory]:
        """Get category by name and optional parent."""
        query = self.db.query(DocumentCategory).filter(DocumentCategory.name == name)
        if parent_id is not None:
            query = query.filter(DocumentCategory.parent_id == parent_id)
        return query.first()

    def get_root_categories(self) -> List[DocumentCategory]:
        """Get all root categories (no parent)."""
        return self.db.query(DocumentCategory).filter(DocumentCategory.parent_id.is_(None)).order_by(DocumentCategory.name).all()

    def get_category_children(self, parent_id: int) -> List[DocumentCategory]:
        """Get child categories of a parent category."""
        return self.db.query(DocumentCategory).filter(DocumentCategory.parent_id == parent_id).order_by(DocumentCategory.name).all()

    def get_category_tree(self, category_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get hierarchical category tree.

        Args:
            category_id: Root category ID, or None for all roots

        Returns:
            List of category dicts with children
        """
        if category_id:
            roots = [self.get_category_by_id(category_id)]
        else:
            roots = self.get_root_categories()

        def build_tree(category):
            return {
                'id': category.id,
                'name': category.name,
                'description': category.description,
                'children': [build_tree(child) for child in self.get_category_children(category.id)]
            }

        return [build_tree(root) for root in roots if root]

    def update_category(self, category_id: int, name: Optional[str] = None,
                       description: Optional[str] = None, parent_id: Optional[int] = None) -> Optional[DocumentCategory]:
        """
        Update category properties.

        Args:
            category_id: Category ID to update
            name: New name
            description: New description
            parent_id: New parent ID

        Returns:
            Updated category or None if not found
        """
        category = self.get_category_by_id(category_id)
        if not category:
            return None

        if name is not None:
            category.name = name
        if description is not None:
            category.description = description
        if parent_id is not None:
            category.parent_id = parent_id

        self.db.commit()
        self.db.refresh(category)
        return category

    def delete_category(self, category_id: int) -> bool:
        """
        Delete category and all its associations.
        Note: This will also delete child categories recursively.

        Args:
            category_id: Category ID to delete

        Returns:
            True if deleted, False if not found
        """
        category = self.get_category_by_id(category_id)
        if not category:
            return False

        # Recursively delete children first
        children = self.get_category_children(category_id)
        for child in children:
            self.delete_category(child.id)

        # Remove associations
        self.db.query(DocumentCategoryAssignment).filter(DocumentCategoryAssignment.category_id == category_id).delete()

        self.db.delete(category)
        self.db.commit()
        return True

    def add_category_to_document(self, document_id: int, category_id: int) -> bool:
        """
        Add category to document.

        Args:
            document_id: Document ID
            category_id: Category ID

        Returns:
            True if added, False if already exists or invalid IDs
        """
        # Check if association already exists
        existing = self.db.query(DocumentCategoryAssignment).filter(
            DocumentCategoryAssignment.document_id == document_id,
            DocumentCategoryAssignment.category_id == category_id
        ).first()

        if existing:
            return False

        association = DocumentCategoryAssignment(document_id=document_id, category_id=category_id)
        self.db.add(association)
        self.db.commit()
        return True

    def remove_category_from_document(self, document_id: int, category_id: int) -> bool:
        """
        Remove category from document.

        Args:
            document_id: Document ID
            category_id: Category ID

        Returns:
            True if removed, False if not found
        """
        association = self.db.query(DocumentCategoryAssignment).filter(
            DocumentCategoryAssignment.document_id == document_id,
            DocumentCategoryAssignment.category_id == category_id
        ).first()

        if not association:
            return False

        self.db.delete(association)
        self.db.commit()
        return True

    def get_document_categories(self, document_id: int) -> List[DocumentCategory]:
        """Get all categories for a document."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        return document.categories if document else []

    def get_category_usage_stats(self) -> List[Dict[str, Any]]:
        """
        Get usage statistics for all categories.

        Returns:
            List of dicts with category info and document count
        """
        from sqlalchemy import func

        result = self.db.query(
            DocumentCategory,
            func.count(DocumentCategoryAssignment.document_id).label('document_count')
        ).outerjoin(DocumentCategoryAssignment).group_by(DocumentCategory.id).all()

        return [
            {
                'id': cat.id,
                'name': cat.name,
                'description': cat.description,
                'parent_id': cat.parent_id,
                'document_count': count
            }
            for cat, count in result
        ]