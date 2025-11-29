"""
Tag management system for LocalRAG documents.

Handles CRUD operations for tags and tag-document associations,
with AI-powered suggestions and intelligent color management.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.ai.tag_suggester import AITagSuggester
from src.database.models import (
    Document,
    DocumentChunk,
    DocumentTag,
    DocumentTagAssignment,
)
from src.utils.tag_colors import TagColorManager

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

    def suggest_tags_for_document(self, document_id: int, max_suggestions: int = 5) -> List[str]:
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
            suggestions = self.ai_suggester.suggest_tags(
                document_content=content,
                document_title=str(document.filename),
                max_suggestions=max_suggestions,
            )
            # Extract tag names from the suggestion dictionaries
            return [
                suggestion.get("tag", "")
                for suggestion in suggestions
                if isinstance(suggestion, dict)
            ]
        return []

    def get_tag_usage_stats(self) -> List[Dict[str, Any]]:
        """Get usage statistics for all tags."""
        # Get all tags with their usage counts

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
