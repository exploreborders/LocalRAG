"""
Document management classes for tagging and categorization.

Provides TagManager and CategoryManager classes for organizing documents
with tags and hierarchical categories, including AI-powered suggestions.
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from .database.models import DocumentTag, DocumentCategory, Document, DocumentTagAssignment, DocumentCategoryAssignment
from .ai_tag_suggester import AITagSuggester
from .tag_colors import TagColorManager
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
                'usage_count': tag.usage_count,
                'document_count': count,
                'created_at': tag.created_at
            }
            for tag, count in result
        ]

    def suggest_tags_for_document(self, document_id: int, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate AI-powered tag suggestions for a document.

        Args:
            document_id: Document ID to generate suggestions for
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of tag suggestions with confidence scores
        """
        try:
            # Get document content
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document or not document.full_content:
                return []

            # Get existing tags for this document
            existing_tags = [tag.name for tag in self.get_document_tags(document_id)]

            # Generate AI suggestions
            suggestions = self.ai_suggester.suggest_tags(
                document.full_content,
                document.filename,
                existing_tags,
                max_suggestions
            )

            return suggestions

        except Exception as e:
            logger.error(f"Error generating tag suggestions for document {document_id}: {e}")
            return []

    def auto_assign_tags(self, document_id: int, min_confidence: float = 0.7) -> List[str]:
        """
        Automatically assign high-confidence AI-suggested tags to a document.

        Args:
            document_id: Document ID to auto-tag
            min_confidence: Minimum confidence threshold for auto-assignment

        Returns:
            List of auto-assigned tag names
        """
        try:
            suggestions = self.suggest_tags_for_document(document_id, max_suggestions=10)
            assigned_tags = []

            for suggestion in suggestions:
                if suggestion['confidence'] >= min_confidence:
                    tag_name = suggestion['tag']

                    # Check if tag exists, create if not
                    existing_tag = self.get_tag_by_name(tag_name)
                    if not existing_tag:
                        # Create tag with AI-generated unique color
                        existing_tag = self.create_tag_with_ai_color(tag_name)

                    # Assign tag to document
                    if existing_tag and self.add_tag_to_document(document_id, existing_tag.id):
                        assigned_tags.append(tag_name)

            return assigned_tags

        except Exception as e:
            logger.error(f"Error auto-assigning tags to document {document_id}: {e}")
            return []

    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular tags by usage count.

        Args:
            limit: Maximum number of tags to return

        Returns:
            List of popular tags with usage statistics
        """
        try:
            stats = self.get_tag_usage_stats()
            # Sort by document count descending
            popular = sorted(stats, key=lambda x: x['document_count'], reverse=True)
            return popular[:limit]
        except Exception as e:
            logger.error(f"Error getting popular tags: {e}")
            return []

    def get_related_tags(self, tag_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find tags that are often used together with the given tag.

        Args:
            tag_name: Base tag name
            limit: Maximum number of related tags to return

        Returns:
            List of related tags with co-occurrence statistics
        """
        try:
            # Get documents that have this tag
            tag = self.get_tag_by_name(tag_name)
            if not tag:
                return []

            # Find other tags used on the same documents
            from sqlalchemy import func

            # Get document IDs that have this tag
            doc_ids_with_tag = self.db.query(DocumentTagAssignment.document_id).filter(
                DocumentTagAssignment.tag_id == tag.id
            ).subquery()

            # Find other tags on those documents
            related_tags = self.db.query(
                DocumentTag,
                func.count(DocumentTagAssignment.document_id).label('co_occurrence')
            ).join(DocumentTagAssignment).filter(
                DocumentTagAssignment.document_id.in_(doc_ids_with_tag),
                DocumentTag.id != tag.id
            ).group_by(DocumentTag.id).order_by(func.count(DocumentTagAssignment.document_id).desc()).limit(limit).all()

            return [
                {
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color,
                    'co_occurrence': count
                }
                for tag, count in related_tags
            ]

        except Exception as e:
            logger.error(f"Error getting related tags for {tag_name}: {e}")
            return []

    def create_tag_with_ai_color(self, name: str, description: Optional[str] = None) -> DocumentTag:
        """
        Create a tag with AI-generated color suggestion, ensuring color uniqueness.

        Args:
            name: Tag name
            description: Optional description

        Returns:
            Created DocumentTag instance
        """
        # Get existing colors to avoid duplicates
        existing_colors = set()
        try:
            existing_tags = self.db.query(DocumentTag).all()
            existing_colors = {tag.color for tag in existing_tags}
        except Exception:
            # If we can't query existing tags, continue with basic color generation
            pass

        # Generate color and ensure uniqueness
        color = self.color_manager.generate_color(name)

        # If color is already used, find an alternative
        if color in existing_colors:
            color = self.color_manager.get_similar_color(color, existing_colors)

        return self.create_tag(name, color=color, description=description)

    def update_tag_color(self, tag_name: str, new_color: str) -> bool:
        """
        Update the color of a tag by name.

        Args:
            tag_name: Name of the tag to update
            new_color: New hex color code

        Returns:
            True if update was successful
        """
        try:
            tag = self.get_tag_by_name(tag_name)
            if not tag:
                return False

            # Validate color format
            if not self.color_manager.validate_hex_color(new_color):
                return False

            tag.color = new_color
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating tag color for '{tag_name}': {e}")
            self.db.rollback()
            return False

    def get_color_palette(self) -> List[str]:
        """
        Get available color palette for tag creation.

        Returns:
            List of hex color codes
        """
        return self.color_manager.get_color_palette()

    def validate_tag_name(self, name: str) -> Dict[str, Any]:
        """
        Validate tag name and provide suggestions if invalid.

        Args:
            name: Tag name to validate

        Returns:
            Dict with validation result and suggestions
        """
        result = {
            'valid': True,
            'errors': [],
            'suggestions': []
        }

        # Check length
        if len(name.strip()) < 2:
            result['valid'] = False
            result['errors'].append("Tag name must be at least 2 characters long")

        if len(name.strip()) > 50:
            result['valid'] = False
            result['errors'].append("Tag name must be less than 50 characters long")

        # Check for existing tag
        if self.get_tag_by_name(name.strip()):
            result['valid'] = False
            result['errors'].append("Tag name already exists")

        # Check for special characters (allow basic punctuation)
        if not name.replace(' ', '').replace('-', '').replace('_', '').isalnum():
            result['valid'] = False
            result['errors'].append("Tag name can only contain letters, numbers, spaces, hyphens, and underscores")

        # Generate suggestions if invalid
        if not result['valid']:
            # Simple suggestions: capitalize, trim, etc.
            suggestions = []
            clean_name = name.strip()
            if clean_name:
                suggestions.append(clean_name.title())
                suggestions.append(clean_name.lower())
                if len(clean_name) > 50:
                    suggestions.append(clean_name[:47] + "...")

            result['suggestions'] = suggestions[:3]  # Max 3 suggestions

        return result

    def bulk_tag_operation(self, document_ids: List[int], tag_names: List[str],
                          operation: str = 'add') -> Dict[str, Any]:
        """
        Perform bulk tag operations on multiple documents.

        Args:
            document_ids: List of document IDs
            tag_names: List of tag names
            operation: 'add' or 'remove'

        Returns:
            Dict with operation results
        """
        results = {
            'success': True,
            'total_operations': 0,
            'successful_operations': 0,
            'errors': []
        }

        try:
            for doc_id in document_ids:
                for tag_name in tag_names:
                    results['total_operations'] += 1

                    try:
                        # Get or create tag
                        tag = self.get_tag_by_name(tag_name)
                        if not tag and operation == 'add':
                            tag = self.create_tag_with_ai_color(tag_name)

                        if tag:
                            if operation == 'add':
                                success = self.add_tag_to_document(doc_id, tag.id)
                            elif operation == 'remove':
                                success = self.remove_tag_from_document(doc_id, tag.id)
                            else:
                                success = False

                            if success:
                                results['successful_operations'] += 1
                        else:
                            results['errors'].append(f"Could not find or create tag: {tag_name}")

                    except Exception as e:
                        results['errors'].append(f"Error with document {doc_id}, tag {tag_name}: {str(e)}")

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Bulk operation failed: {str(e)}")

        return results


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
        category = DocumentCategory(name=name, description=description, parent_category_id=parent_id)
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
            query = query.filter(DocumentCategory.parent_category_id == parent_id)
        return query.first()

    def get_root_categories(self) -> List[DocumentCategory]:
        """Get all root categories (no parent)."""
        return self.db.query(DocumentCategory).filter(DocumentCategory.parent_category_id.is_(None)).order_by(DocumentCategory.name).all()

    def get_category_children(self, parent_id: int) -> List[DocumentCategory]:
        """Get child categories of a parent category."""
        return self.db.query(DocumentCategory).filter(DocumentCategory.parent_category_id == parent_id).order_by(DocumentCategory.name).all()

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
            category.parent_category_id = parent_id

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
                'parent_category_id': cat.parent_category_id,
                'document_count': count
            }
            for cat, count in result
        ]