"""
Unit tests for error handling utilities.
"""

from unittest.mock import Mock, patch

import pytest

from src.utils.error_handler import (
    DatabaseError,
    ProcessingError,
    ValidationError,
    error_context,
    handle_errors,
    safe_execute,
    validate_and_handle,
)


class TestSafeExecute:
    """Test the safe_execute function."""

    def test_successful_execution(self):
        """Test successful function execution."""
        result = safe_execute(lambda: 42)
        assert result == 42

    def test_exception_with_default(self):
        """Test exception handling with default return value."""
        result = safe_execute(lambda: 1 / 0, default_return=0)
        assert result == 0

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""

        def failing_func():
            raise ValidationError("test")

        result = safe_execute(failing_func, error_types=ValidationError, default_return="caught")
        assert result == "caught"

    def test_multiple_exception_types(self):
        """Test catching multiple exception types."""
        result = safe_execute(
            lambda: (_ for _ in ()).throw(ValueError("test")),
            error_types=(ValueError, TypeError),
            default_return="caught",
        )
        assert result == "caught"

    def test_unexpected_exception_passthrough(self):
        """Test that unexpected exceptions are not caught."""
        with pytest.raises(ZeroDivisionError):
            safe_execute(lambda: 1 / 0, error_types=(ValueError,), default_return=0)


class TestErrorContext:
    """Test error context management."""

    def test_error_context_basic(self):
        """Test basic error context functionality."""
        with error_context("test_operation"):
            # Should not raise
            pass

    def test_error_context_with_exception(self):
        """Test error context with exception."""
        with pytest.raises(ValueError) as exc_info:
            with error_context("test_operation"):
                raise ValueError("test error")

        # The original exception should be preserved
        assert str(exc_info.value) == "test error"


class TestHandleErrors:
    """Test the handle_errors decorator."""

    def test_handle_errors_success(self):
        """Test successful execution with error handler."""

        @handle_errors(default_return="error")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_handle_errors_exception(self):
        """Test exception handling with decorator."""

        @handle_errors(default_return="error")
        def test_func():
            raise ValueError("test error")

        result = test_func()
        assert result == "error"


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
