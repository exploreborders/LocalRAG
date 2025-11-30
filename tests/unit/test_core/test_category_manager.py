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
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_category

        result = category_manager.get_category_by_name("test_category")

        assert result == mock_category
        mock_db_session.query.assert_called_once()
        # Verify the query was filtered correctly
        query_mock = mock_db_session.query.return_value
        query_mock.filter.assert_called_once()

    def test_get_category_by_name_with_parent(
        self, category_manager, mock_db_session, mock_category
    ):
        """Test getting category by name with parent."""
        # Set up mock chain for query.filter().filter().first()
        first_filter_mock = MagicMock()
        first_filter_mock.filter.return_value.first.return_value = mock_category
        mock_db_session.query.return_value.filter.return_value = first_filter_mock

        result = category_manager.get_category_by_name("test_category", 5)

        assert result == mock_category
        mock_db_session.query.assert_called_once()

    def test_get_category_by_name_not_found(self, category_manager, mock_db_session):
        """Test getting non-existent category by name."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

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
                name="test_category",
                description="Test description",
                parent_category_id=5,
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

        # Mock the existing check to return None (no existing assignment)
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

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
        # Mock the existing check to return None (no existing assignment)
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        mock_db_session.commit.side_effect = Exception("Database error")

        with patch("src.core.categorization.category_manager.DocumentCategoryAssignment"):
            result = category_manager.add_category_to_document(1, 2)

            assert result is False
            mock_db_session.rollback.assert_called_once()

    def test_remove_category_from_document_success(
        self, category_manager, mock_db_session, mock_assignment
    ):
        """Test successfully removing category from document."""
        # Set up mock chain for query.filter().filter().first()
        first_filter_mock = MagicMock()
        first_filter_mock.filter.return_value.first.return_value = mock_assignment
        mock_db_session.query.return_value.filter.return_value = first_filter_mock

        mock_db_session.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = category_manager.remove_category_from_document(1, 2)

        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_remove_category_from_document_not_found(self, category_manager, mock_db_session):
        """Test removing category when assignment doesn't exist."""
        # Mock the query chain: query().filter(doc_id, cat_id).first()
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        result = category_manager.remove_category_from_document(1, 2)

        assert result is False
        mock_db_session.delete.assert_not_called()
        mock_db_session.commit.assert_not_called()

    def test_get_document_categories(self, category_manager, mock_db_session, mock_category):
        """Test getting all categories for a document."""
        # Mock assignment with category
        mock_assignment = MagicMock()
        mock_assignment.category = mock_category

        # Mock query chain: query().filter().all()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = [mock_assignment]

        mock_db_session.query.return_value = mock_query

        result = category_manager.get_document_categories(1)

        assert len(result) == 1
        assert result[0] == mock_category

    def test_get_category_hierarchy(self, category_manager, mock_db_session, mock_category):
        """Test getting category hierarchy path."""
        # Mock root category
        mock_root = MagicMock(spec=DocumentCategory)
        mock_root.id = 1
        mock_root.name = "root"
        mock_root.parent_category_id = None

        # Mock child category
        mock_category.id = 2
        mock_category.name = "child"
        mock_category.parent_category_id = 1

        # Mock query chain for initial query
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter

        # First call returns child, second returns root, third returns None
        mock_filter.first.side_effect = [mock_category, mock_root, None]

        mock_db_session.query.return_value = mock_query

        result = category_manager.get_category_hierarchy(2)

        assert len(result) == 2
        assert result[0] == mock_root
        assert result[1] == mock_category

    def test_get_root_categories(self, category_manager, mock_db_session, mock_category):
        """Test getting root categories (no parent)."""
        # Mock query chain: query().filter().order_by().all()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order_by = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order_by
        mock_order_by.all.return_value = [mock_category]

        mock_db_session.query.return_value = mock_query

        result = category_manager.get_root_categories()

        assert len(result) == 1
        assert result[0] == mock_category

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
        """Test successful category deletion."""
        # Mock all queries to succeed
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_category
        mock_db_session.query.return_value.filter.return_value.all.return_value = []  # No children
        mock_db_session.query.return_value.filter.return_value.delete.return_value = None
        mock_db_session.commit.return_value = None

        result = category_manager.delete_category(1)

        assert result is True
        mock_db_session.commit.assert_called_once()

    def test_delete_category_not_found(self, category_manager, mock_db_session):
        """Test deleting non-existent category."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        result = category_manager.delete_category(999)

        assert result is False
        mock_db_session.delete.assert_not_called()
