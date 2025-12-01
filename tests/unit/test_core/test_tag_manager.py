"""
Unit tests for TagManager class.

Tests tag CRUD operations, document associations, AI suggestions,
and color management functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.tagging.tag_manager import TagManager
from src.database.models import (
    DocumentTag,
    DocumentTagAssignment,
)


class TestTagManager:
    """Test the TagManager class functionality."""

    @pytest.fixture
    def tag_manager(self, mock_db_session):
        """Create TagManager instance with mocked database."""
        return TagManager(mock_db_session)

    @pytest.fixture
    def mock_tag(self):
        """Create a mock DocumentTag."""
        tag = MagicMock(spec=DocumentTag)
        tag.id = 1
        tag.name = "test_tag"
        tag.color = "#FF5733"
        return tag

    @pytest.fixture
    def mock_assignment(self):
        """Create a mock DocumentTagAssignment."""
        assignment = MagicMock(spec=DocumentTagAssignment)
        assignment.document_id = 1
        assignment.tag_id = 1
        return assignment

    def test_init(self, mock_db_session):
        """Test TagManager initialization."""
        manager = TagManager(mock_db_session)
        assert manager.db == mock_db_session
        assert hasattr(manager, "ai_suggester")
        assert hasattr(manager, "color_manager")

    def test_get_tag_by_name_found(self, tag_manager, mock_db_session, mock_tag):
        """Test getting existing tag by name."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_tag

        result = tag_manager.get_tag_by_name("test_tag")

        assert result == mock_tag
        mock_db_session.query.assert_called_once_with(DocumentTag)
        mock_db_session.query.return_value.filter.assert_called_once()

    def test_get_tag_by_name_not_found(self, tag_manager, mock_db_session):
        """Test getting non-existent tag by name."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        result = tag_manager.get_tag_by_name("nonexistent")

        assert result is None

    def test_create_tag_with_color(self, tag_manager, mock_db_session):
        """Test creating tag with specified color."""
        mock_tag = MagicMock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        # Mock the DocumentTag constructor
        with patch("src.core.tagging.tag_manager.DocumentTag") as mock_tag_class:
            mock_tag_class.return_value = mock_tag

            result = tag_manager.create_tag("test_tag", "#FF5733")

            mock_tag_class.assert_called_once_with(name="test_tag", color="#FF5733")
            assert result == mock_tag
            mock_db_session.add.assert_called_once_with(mock_tag)
            mock_db_session.commit.assert_called_once()

    def test_create_tag_auto_color(self, tag_manager, mock_db_session):
        """Test creating tag with auto-generated color."""
        mock_tag = MagicMock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        # Mock color generation
        with patch.object(tag_manager.color_manager, "generate_color", return_value="#FF5733"):
            with patch("src.core.tagging.tag_manager.DocumentTag") as mock_tag_class:
                mock_tag_class.return_value = mock_tag

                result = tag_manager.create_tag("test_tag")

                tag_manager.color_manager.generate_color.assert_called_once_with("test_tag")
                mock_tag_class.assert_called_once_with(name="test_tag", color="#FF5733")
                assert result == mock_tag

    def test_add_tag_to_document_success(self, tag_manager, mock_db_session):
        """Test successfully adding tag to document."""
        mock_assignment = MagicMock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        # Mock the existing check query to return None (no existing assignment)
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch("src.core.tagging.tag_manager.DocumentTagAssignment") as mock_assignment_class:
            mock_assignment_class.return_value = mock_assignment

            result = tag_manager.add_tag_to_document(1, 2)

            assert result is True
            mock_db_session.add.assert_called_once_with(mock_assignment)
            mock_db_session.commit.assert_called_once()

    def test_add_tag_to_document_failure(self, tag_manager, mock_db_session):
        """Test failure when adding tag to document."""
        # Mock the existing check to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        mock_db_session.commit.side_effect = Exception("Database error")

        with patch("src.core.tagging.tag_manager.DocumentTagAssignment"):
            with pytest.raises(Exception, match="Database error"):
                tag_manager.add_tag_to_document(1, 2)

    def test_remove_tag_from_document_success(self, tag_manager, mock_db_session, mock_assignment):
        """Test successfully removing tag from document."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            mock_assignment
        )
        mock_db_session.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = tag_manager.remove_tag_from_document(1, 2)

        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_remove_tag_from_document_not_found(self, tag_manager, mock_db_session):
        """Test removing tag when assignment doesn't exist."""
        # Mock the query chain to return None
        mock_query = MagicMock()
        mock_db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        result = tag_manager.remove_tag_from_document(1, 2)

        assert result is False
        mock_db_session.delete.assert_not_called()

    def test_get_document_tags(self, tag_manager, mock_db_session, mock_tag):
        """Test getting all tags for a document."""
        mock_assignment = MagicMock()
        mock_assignment.tag = mock_tag
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_assignment]

        result = tag_manager.get_document_tags(1)

        assert result == [mock_tag]
        mock_db_session.query.assert_called_once_with(DocumentTagAssignment)

    def test_suggest_tags_for_document(self, tag_manager, mock_db_session):
        """Test AI-powered tag suggestions."""
        # Mock AI suggester to return list of dictionaries as expected
        tag_manager.ai_suggester.suggest_tags = MagicMock(
            return_value=[
                {
                    "tag": "ai_tag1",
                    "confidence": 0.9,
                    "relevance_score": 0.8,
                    "source": "ai_generated",
                },
                {
                    "tag": "ai_tag2",
                    "confidence": 0.7,
                    "relevance_score": 0.6,
                    "source": "ai_generated",
                },
            ]
        )

        # Mock the database calls
        mock_document = MagicMock()
        mock_document.filename = "test.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        mock_chunk = MagicMock()
        mock_chunk.content = "Test content"
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_chunk
        ]

        result = tag_manager.suggest_tags_for_document(1, 3)

        assert result == ["ai_tag1", "ai_tag2"]
        tag_manager.ai_suggester.suggest_tags.assert_called_once()

    def test_delete_tag_success(self, tag_manager, mock_db_session, mock_tag):
        """Test successfully deleting a tag."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_tag
        mock_db_session.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = tag_manager.delete_tag("test_tag")

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_tag)
        mock_db_session.commit.assert_called_once()

    def test_delete_tag_not_found(self, tag_manager, mock_db_session):
        """Test deleting non-existent tag."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        result = tag_manager.delete_tag("nonexistent")

        assert result is False
        mock_db_session.delete.assert_not_called()

    def test_get_all_tags(self, tag_manager, mock_db_session, mock_tag):
        """Test getting all tags."""
        mock_db_session.query.return_value.all.return_value = [mock_tag]

        result = tag_manager.get_all_tags()

        assert result == [mock_tag]
        mock_db_session.query.assert_called_once_with(DocumentTag)

    def test_get_popular_tags(self, tag_manager, mock_db_session):
        """Test getting popular tags by usage count."""
        # Mock the complex query result
        mock_result = MagicMock()
        mock_result.name = "popular_tag"
        mock_result.color = "#FF5733"
        mock_result.document_count = 5

        # Mock the query chain
        mock_query = MagicMock()
        mock_db_session.query.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_result]

        result = tag_manager.get_popular_tags(10)

        assert len(result) == 1
        assert result[0]["name"] == "popular_tag"
        assert result[0]["document_count"] == 5

    def test_get_related_tags(self, tag_manager, mock_db_session):
        """Test getting related tags based on co-occurrence."""
        # Mock the tag lookup
        mock_tag = MagicMock()
        mock_tag.id = 1
        tag_manager.get_tag_by_name = MagicMock(return_value=mock_tag)

        # Mock document IDs query
        mock_assignment = MagicMock()
        mock_assignment.document_id = 1
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_assignment]

        # Mock the related tags query result
        mock_result = MagicMock()
        mock_result.name = "related_tag"
        mock_result.color = "#FF5733"
        mock_result.co_occurrence = 3

        # Set up query mock for the complex query
        mock_query = MagicMock()
        mock_db_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_result]

        result = tag_manager.get_related_tags("base_tag", 5)

        assert len(result) == 1
        assert result[0]["name"] == "related_tag"
        assert result[0]["co_occurrence"] == 3
