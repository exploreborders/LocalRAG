"""
Enhanced error handling utilities for LocalRAG system.
Provides consistent error handling, logging, and recovery mechanisms.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LocalRAGError(Exception):
    """Base exception class for LocalRAG system."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc(),
        }


class DatabaseError(LocalRAGError):
    """Database-related errors."""

    pass


class ValidationError(LocalRAGError):
    """Data validation errors."""

    pass


class ProcessingError(LocalRAGError):
    """Document processing errors."""

    pass


class ConfigurationError(LocalRAGError):
    """Configuration-related errors."""

    pass


class SecurityError(LocalRAGError):
    """Security-related errors."""

    pass


class NetworkError(LocalRAGError):
    """Network/communication errors."""

    pass


class ErrorHandler:
    """Centralized error handler for consistent error management."""

    def __init__(self, logger_name: Optional[str] = None):
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger(__name__)
        self.error_stats: Dict[str, Any] = {
            "total_errors": 0,
            "error_types": {},
            "recent_errors": [],
        }

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
        log_level: str = "error",
    ) -> Optional[Dict[str, Any]]:
        """
        Handle an error with consistent logging and tracking.

        Args:
            error: The exception to handle
            context: Additional context information
            reraise: Whether to reraise the exception
            log_level: Logging level to use

        Returns:
            Error information dictionary if not reraising
        """
        # Update error statistics
        self.error_stats["total_errors"] += 1
        error_type = error.__class__.__name__
        self.error_stats["error_types"][error_type] = (
            self.error_stats["error_types"].get(error_type, 0) + 1
        )

        # Create error info
        if isinstance(error, LocalRAGError):
            error_info = error.to_dict()
        else:
            error_info = {
                "error_type": error.__class__.__name__,
                "error_code": error.__class__.__name__,
                "message": str(error),
                "context": context or {},
                "timestamp": time.time(),
                "traceback": traceback.format_exc(),
            }

        # Add additional context
        if context:
            error_info["context"].update(context)

        # Store recent error (keep last 10)
        self.error_stats["recent_errors"].append(error_info)
        if len(self.error_stats["recent_errors"]) > 10:
            self.error_stats["recent_errors"].pop(0)

        # Log the error
        log_message = f"{error_info['error_type']}: {error_info['message']}"
        if context:
            log_message += f" | Context: {context}"

        getattr(self.logger, log_level)(log_message)

        if log_level == "debug":
            self.logger.debug(f"Full traceback: {error_info['traceback']}")

        # Reraise if requested
        if reraise:
            raise error

        return error_info

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self.error_stats.copy()

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_stats = {"total_errors": 0, "error_types": {}, "recent_errors": []}


# Global error handler instance
global_error_handler = ErrorHandler()


# Database transaction helper
@contextmanager
def database_transaction(db_session, rollback_on_error: bool = True):
    """
    Context manager for database transactions with error handling.

    Args:
        db_session: Database session object
        rollback_on_error: Whether to rollback on errors
    """
    try:
        yield db_session
        db_session.commit()
    except Exception as e:
        if rollback_on_error:
            try:
                db_session.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback database transaction: {rollback_error}")

        global_error_handler.handle_error(
            DatabaseError(f"Database transaction failed: {e}"),
            context={"operation": "database_transaction"},
            reraise=True,
        )
