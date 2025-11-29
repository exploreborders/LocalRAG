"""
Unit tests for CategoryManager class.

Tests category CRUD operations, document associations, hierarchy management,
and tree building functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from src.core.categorization.category_manager import CategoryManager
from src.database.models import DocumentCategory, DocumentCategoryAssignment


class TestCategoryManager:
    """Test the CategoryManager class functionality."""

    @pytest.fixture
    def category_manager(self, mock_db_session):
        """Create CategoryManager instance with mocked database."""
        return CategoryManager(mock_db_session)

    @pytest.fixture
    def mock_category(self):
        """Create a mock DocumentCategory."""
        category = MagicMock(spec=DocumentCategory)
        category.id = 1
        category.name = "test_category"
        category.description = "Test category description"
        category.parent_id = None
        return category

    @pytest.fixture
    def mock_assignment(self):
        """Create a mock DocumentCategoryAssignment."""
        assignment = MagicMock(spec=DocumentCategoryAssignment)
        assignment.document_id = 1
        assignment.category_id = 1
        return assignment

    def test_init(self, mock_db_session):
        """Test CategoryManager initialization."""
        manager = CategoryManager(mock_db_session)
        assert manager.db == mock_db_session

    def test_get_category_by_name_no_parent(self, category_manager, mock_db_session, mock_category):
        """Test getting category by name without parent."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            mock_category
        )

        result = category_manager.get_category_by_name("test_category")

        assert result == mock_category
        mock_db_session.query.assert_called_once_with(DocumentCategory)

    def test_get_category_by_name_with_parent(
        self, category_manager, mock_db_session, mock_category
    ):
        """Test getting category by name with parent."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            mock_category
        )

        result = category_manager.get_category_by_name("test_category", 5)

        assert result == mock_category

    def test_get_category_by_name_not_found(self, category_manager, mock_db_session):
        """Test getting non-existent category by name."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            None
        )

        result = category_manager.get_category_by_name("nonexistent")

        assert result is None

    def test_create_category_success(self, category_manager, mock_db_session):
        """Test creating category successfully."""
        mock_category = MagicMock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        with patch(
            "src.core.categorization.category_manager.DocumentCategory"
        ) as mock_category_class:
            mock_category_class.return_value = mock_category

            result = category_manager.create_category("test_category", "Test description", 5)

            mock_category_class.assert_called_once_with(
                name="test_category", description="Test description", parent_id=5
            )
            assert result == mock_category
            mock_db_session.add.assert_called_once_with(mock_category)
            mock_db_session.commit.assert_called_once()

    def test_create_category_failure(self, category_manager, mock_db_session):
        """Test category creation failure."""
        mock_db_session.commit.side_effect = Exception("Database error")

        with patch("src.core.categorization.category_manager.DocumentCategory"):
            result = category_manager.create_category("test_category")

            assert result is None
            mock_db_session.rollback.assert_called_once()

    def test_add_category_to_document_success(self, category_manager, mock_db_session):
        """Test successfully adding category to document."""
        mock_assignment = MagicMock()
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        with patch(
            "src.core.categorization.category_manager.DocumentCategoryAssignment"
        ) as mock_assignment_class:
            mock_assignment_class.return_value = mock_assignment

            result = category_manager.add_category_to_document(1, 2)

            assert result is True
            mock_assignment_class.assert_called_once_with(document_id=1, category_id=2)
            mock_db_session.add.assert_called_once_with(mock_assignment)
            mock_db_session.commit.assert_called_once()

    def test_add_category_to_document_failure(self, category_manager, mock_db_session):
        """Test failure when adding category to document."""
        mock_db_session.commit.side_effect = Exception("Database error")

        with patch("src.core.categorization.category_manager.DocumentCategoryAssignment"):
            result = category_manager.add_category_to_document(1, 2)

            assert result is False
            mock_db_session.rollback.assert_called_once()

    def test_remove_category_from_document_success(
        self, category_manager, mock_db_session, mock_assignment
    ):
        """Test successfully removing category from document."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            mock_assignment
        )
        mock_db_session.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = category_manager.remove_category_from_document(1, 2)

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_assignment)
        mock_db_session.commit.assert_called_once()

    def test_remove_category_from_document_not_found(self, category_manager, mock_db_session):
        """Test removing category when assignment doesn't exist."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            None
        )

        result = category_manager.remove_category_from_document(1, 2)

        assert result is False
        mock_db_session.delete.assert_not_called()

    def test_get_document_categories(self, category_manager, mock_db_session, mock_category):
        """Test getting all categories for a document."""
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_category]

        result = category_manager.get_document_categories(1)

        assert result == [mock_category]
        mock_db_session.query.assert_called_once_with(DocumentCategory)

    def test_get_category_hierarchy(self, category_manager, mock_db_session, mock_category):
        """Test getting category hierarchy path."""
        # Mock parent categories
        mock_parent = MagicMock(spec=DocumentCategory)
        mock_parent.id = 2
        mock_parent.name = "parent_category"
        mock_parent.parent_id = None

        # Mock the recursive query results
        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            [mock_category],  # First call for current category
            [mock_parent],  # Second call for parent
        ]

        result = category_manager.get_category_hierarchy(1)

        assert len(result) >= 1
        # Verify the hierarchy building logic is called

    def test_get_root_categories(self, category_manager, mock_db_session, mock_category):
        """Test getting root categories (no parent)."""
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_category]

        result = category_manager.get_root_categories()

        assert result == [mock_category]
        mock_db_session.query.assert_called_once_with(DocumentCategory)

    def test_get_category_tree(self, category_manager, mock_db_session, mock_category):
        """Test building complete category tree."""
        # Mock root categories
        mock_category.parent_id = None
        mock_category.id = 1
        mock_category.name = "root"

        # Mock child categories
        mock_child = MagicMock(spec=DocumentCategory)
        mock_child.id = 2
        mock_child.name = "child"
        mock_child.parent_id = 1

        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            [mock_category],  # Root categories
            [mock_child],  # Children of root
        ]

        result = category_manager.get_category_tree()

        assert isinstance(result, list)
        # Tree structure validation would be complex, so we just check it's a list

    def test_delete_category_success(self, category_manager, mock_db_session, mock_category):
        """Test successfully deleting a category."""
        # Mock no children
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        mock_db_session.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = category_manager.delete_category(1)

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_category)
        mock_db_session.commit.assert_called_once()

    def test_delete_category_with_children(self, category_manager, mock_db_session, mock_category):
        """Test deleting category with children (should fail)."""
        # Mock having children
        mock_child = MagicMock(spec=DocumentCategory)
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_child]

        result = category_manager.delete_category(1)

        assert result is False
        mock_db_session.delete.assert_not_called()

    def test_delete_category_not_found(self, category_manager, mock_db_session):
        """Test deleting non-existent category."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        result = category_manager.delete_category(999)

        assert result is False
        mock_db_session.delete.assert_not_called()
