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
    error_context,
    get_error_handler,
    handle_errors,
    retry_on_error,
    safe_execute,
    set_error_handler,
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


class TestRetryOnError:
    """Test the retry_on_error decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""

        @retry_on_error(max_retries=3)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        call_count = 0

        @retry_on_error(max_retries=3, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test when all retries are exhausted."""

        @retry_on_error(max_retries=2, delay=0.01)
        def test_func():
            raise ValueError("persistent error")

        with pytest.raises(ValueError):
            test_func()


class TestValidateAndHandle:
    """Test the validate_and_handle decorator."""

    def test_validate_and_handle_success(self):
        """Test successful validation and execution."""

        @validate_and_handle(lambda x: x > 0, "Value must be positive")
        def test_func(value):
            return value * 2

        result = test_func(5)
        assert result == 10

    def test_validate_and_handle_validation_failure(self):
        """Test validation failure."""

        @validate_and_handle(lambda x: x > 0, "Value must be positive")
        def test_func(value):
            return value * 2

        with pytest.raises(ValidationError) as exc_info:
            test_func(-1)

        assert "Value must be positive" in str(exc_info.value)


class TestGlobalErrorHandler:
    """Test global error handler functions."""

    def test_get_error_handler(self):
        """Test getting the global error handler."""
        handler = get_error_handler()
        assert isinstance(handler, ErrorHandler)

    def test_set_error_handler(self):
        """Test setting the global error handler."""
        original_handler = get_error_handler()
        new_handler = ErrorHandler()

        set_error_handler(new_handler)
        assert get_error_handler() is new_handler

        # Restore original
        set_error_handler(original_handler)


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
