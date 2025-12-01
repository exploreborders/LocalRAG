"""
Enhanced error handling utilities for LocalRAG system.
Provides consistent error handling, logging, and recovery mechanisms.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, Union

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


def handle_errors(
    error_types: Union[Type[Exception], tuple] = Exception,
    default_return: Any = None,
    log_level: str = "error",
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = False,
):
    """
    Decorator for consistent error handling in functions.

    Args:
        error_types: Exception types to catch
        default_return: Value to return on error
        log_level: Logging level for errors
        context: Additional context to log with errors
        reraise: Whether to reraise exceptions
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Build context from function call
                func_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }
                if context:
                    func_context.update(context)

                global_error_handler.handle_error(
                    e, context=func_context, reraise=reraise, log_level=log_level
                )

                return default_return

        return wrapper

    return decorator


@contextmanager
def error_context(operation: str, context: Optional[Dict[str, Any]] = None, reraise: bool = True):
    """
    Context manager for error handling in blocks of code.

    Args:
        operation: Description of the operation being performed
        context: Additional context information
        reraise: Whether to reraise exceptions
    """
    start_time = time.time()
    operation_context: Dict[str, Any] = {"operation": operation}
    if context:
        operation_context.update(context)

    try:
        logger.debug(f"Starting operation: {operation}")
        yield operation_context
        duration = time.time() - start_time
        logger.debug(f"Completed operation: {operation} in {duration:.2f}s")

    except Exception as e:
        duration = time.time() - start_time
        operation_context["duration"] = duration
        operation_context["failed"] = True

        global_error_handler.handle_error(e, context=operation_context, reraise=reraise)


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_types: Union[Type[Exception], tuple] = Exception,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        error_types: Exception types to catch
        context: Additional context
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except error_types as e:
        exec_context = {
            "function": func.__name__ if hasattr(func, "__name__") else str(func),
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
        }
        if context:
            exec_context.update(context)

        global_error_handler.handle_error(e, context=exec_context, reraise=False)

        return default_return


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    error_types: Union[Type[Exception], tuple] = Exception,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for retrying functions on specific errors.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay after each retry
        error_types: Exception types that trigger retries
        context: Additional context for logging
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except error_types as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed
                        retry_context = {
                            "function": func.__name__,
                            "attempts": attempt + 1,
                            "max_retries": max_retries,
                            "final_attempt": True,
                        }
                        if context:
                            retry_context.update(context)

                        global_error_handler.handle_error(e, context=retry_context, reraise=True)

                    # Log retry attempt
                    retry_context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "delay": current_delay,
                        "retrying": True,
                    }
                    if context:
                        retry_context.update(context)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # This should never be reached due to reraise on final attempt
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError("Unexpected error in retry logic")

        return wrapper

    return decorator


def validate_and_handle(validation_func: Callable, error_message: Optional[str] = None):
    """
    Decorator for input validation with error handling.

    Args:
        validation_func: Function that validates input and returns bool
        error_message: Custom error message for validation failures
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValidationError(
                        error_message or f"Validation failed for {func.__name__}",
                        context={
                            "function": func.__name__,
                            "args": args,
                            "kwargs": kwargs,
                        },
                    )
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                global_error_handler.handle_error(
                    e,
                    context={"function": func.__name__, "validation": True},
                    reraise=True,
                )

        return wrapper

    return decorator


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return global_error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set a custom global error handler."""
    global global_error_handler
    global_error_handler = handler


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
