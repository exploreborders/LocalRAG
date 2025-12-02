"""
Unit tests for error handling utilities.
"""

from unittest.mock import Mock

import pytest

from src.utils.error_handler import (
    DatabaseError,
    ErrorHandler,
    ProcessingError,
    ValidationError,
    database_transaction,
)


class TestErrorHandler:
    """Test the ErrorHandler class."""

    def test_error_handler_init(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler()
        assert handler is not None
        assert handler.error_stats["total_errors"] == 0
        assert handler.error_stats["error_types"] == {}
        assert handler.error_stats["recent_errors"] == []

    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()

        try:
            raise ValueError("test error")
        except ValueError as e:
            result = handler.handle_error(e, {"operation": "test"}, reraise=False)

        assert result is not None
        assert result["error_type"] == "ValueError"
        assert "test error" in result["message"]
        assert result["context"] == {"operation": "test"}
        assert handler.error_stats["total_errors"] == 1

    def test_handle_error_with_reraise(self):
        """Test error handling with reraise."""
        handler = ErrorHandler()

        with pytest.raises(ValueError):
            try:
                raise ValueError("test error")
            except ValueError as e:
                handler.handle_error(e, reraise=True)

    def test_get_error_stats(self):
        """Test getting error statistics."""
        handler = ErrorHandler()
        stats = handler.get_error_stats()

        assert isinstance(stats, dict)
        assert "total_errors" in stats
        assert "error_types" in stats
        assert "recent_errors" in stats

    def test_reset_stats(self):
        """Test resetting error statistics."""
        handler = ErrorHandler()

        # Add an error (don't reraise)
        try:
            raise ValueError("test")
        except ValueError as e:
            handler.handle_error(e, reraise=False)

        assert handler.error_stats["total_errors"] == 1

        # Reset stats
        handler.reset_stats()
        assert handler.error_stats["total_errors"] == 0
        assert handler.error_stats["error_types"] == {}
        assert handler.error_stats["recent_errors"] == []


class TestDatabaseTransaction:
    """Test database transaction decorator."""

    def test_database_transaction_success(self):
        """Test successful database transaction."""
        mock_session = Mock()

        @database_transaction(mock_session)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"
        mock_session.commit.assert_called_once()

    def test_database_transaction_error(self):
        """Test database transaction with error."""
        mock_session = Mock()

        @database_transaction(mock_session, rollback_on_error=True)
        def test_func():
            raise ValueError("database error")

        with pytest.raises(DatabaseError):
            test_func()

        mock_session.rollback.assert_called_once()


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationError("test message")
        assert str(error) == "test message"
        assert error.message == "test message"

    def test_processing_error_creation(self):
        """Test ProcessingError creation."""
        error = ProcessingError("processing failed")
        assert str(error) == "processing failed"
        assert error.message == "processing failed"

    def test_database_error_creation(self):
        """Test DatabaseError creation."""
        error = DatabaseError("database error")
        assert str(error) == "database error"
        assert error.message == "database error"
