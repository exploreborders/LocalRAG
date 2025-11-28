"""
Progress tracking utilities for long-running operations.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Progress information for tracking operations."""

    operation: str
    current: float  # 0-100
    total: float = 100.0
    message: str = ""
    start_time: float = 0.0
    estimated_time_remaining: Optional[float] = None

    @property
    def progress_ratio(self) -> float:
        """Get progress as a ratio (0-1)."""
        return min(self.current / self.total, 1.0)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def update_eta(self, total_items: int, completed_items: int) -> None:
        """Update estimated time remaining based on completed items."""
        if completed_items > 0 and total_items > 0:
            avg_time_per_item = self.elapsed_time / completed_items
            remaining_items = total_items - completed_items
            self.estimated_time_remaining = avg_time_per_item * remaining_items


class ProgressTracker:
    """
    Comprehensive progress tracking for long-running operations.
    """

    def __init__(self, operation: str, total_steps: int = 100):
        """
        Initialize progress tracker.

        Args:
            operation: Name of the operation being tracked
            total_steps: Total number of steps (for percentage calculation)
        """
        self.operation = operation
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.callbacks: list[Callable[[ProgressInfo], None]] = []
        self.sub_operations: Dict[str, "ProgressTracker"] = {}
        self.estimated_time_remaining: Optional[float] = None

    def add_callback(self, callback: Callable[[ProgressInfo], None]) -> None:
        """Add a progress callback."""
        self.callbacks.append(callback)

    def update(self, step: Optional[int] = None, message: str = "", increment: bool = True) -> None:
        """
        Update progress.

        Args:
            step: Specific step number (if None, increments current)
            message: Progress message
            increment: Whether to increment step counter
        """
        if step is not None:
            self.current_step = step
        elif increment:
            self.current_step += 1

        # Calculate progress percentage
        progress = min((self.current_step / self.total_steps) * 100, 100)

        # Create progress info
        info = ProgressInfo(
            operation=self.operation,
            current=progress,
            total=100.0,
            message=message,
            start_time=self.start_time,
        )

        # Update ETA if we have sub-operations
        if self.sub_operations:
            total_eta = 0
            for sub_tracker in self.sub_operations.values():
                if sub_tracker.estimated_time_remaining:
                    total_eta += sub_tracker.estimated_time_remaining
            if total_eta > 0:
                info.estimated_time_remaining = total_eta

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def start_sub_operation(self, name: str, total_steps: int = 100) -> "ProgressTracker":
        """
        Start tracking a sub-operation.

        Args:
            name: Name of the sub-operation
            total_steps: Total steps for the sub-operation

        Returns:
            ProgressTracker for the sub-operation
        """
        sub_tracker = ProgressTracker(f"{self.operation}:{name}", total_steps)
        self.sub_operations[name] = sub_tracker
        return sub_tracker

    def complete_sub_operation(self, name: str) -> None:
        """Mark a sub-operation as completed."""
        if name in self.sub_operations:
            del self.sub_operations[name]

    def get_progress_info(self) -> ProgressInfo:
        """Get current progress information."""
        progress = min((self.current_step / self.total_steps) * 100, 100)
        return ProgressInfo(
            operation=self.operation,
            current=progress,
            total=100.0,
            message="",
            start_time=self.start_time,
        )

    def reset(self) -> None:
        """Reset the progress tracker."""
        self.current_step = 0
        self.start_time = time.time()
        self.sub_operations.clear()


@contextmanager
def track_progress(operation: str, total_steps: int = 100, callback: Optional[Callable] = None):
    """
    Context manager for progress tracking.

    Args:
        operation: Name of the operation
        total_steps: Total number of steps
        callback: Optional progress callback

    Yields:
        ProgressTracker instance
    """
    tracker = ProgressTracker(operation, total_steps)
    if callback:
        tracker.add_callback(callback)

    tracker.update(0, f"Starting {operation}")
    try:
        yield tracker
        tracker.update(total_steps, f"Completed {operation}")
    except Exception as e:
        tracker.update(message=f"Failed {operation}: {e}")
        raise
    finally:
        logger.info(f"Progress tracking completed for {operation}")


def create_progress_callback(operation: str) -> Callable[[ProgressInfo], None]:
    """
    Create a standard progress callback that logs progress.

    Args:
        operation: Operation name for logging

    Returns:
        Progress callback function
    """

    def callback(info: ProgressInfo) -> None:
        elapsed = info.elapsed_time
        eta_str = ""
        if info.estimated_time_remaining:
            eta_str = f" (ETA: {info.estimated_time_remaining:.1f}s)"

        logger.info(
            f"{operation}: {info.current:.1f}% - {info.message} "
            f"(elapsed: {elapsed:.1f}s{eta_str})"
        )

    return callback


class BatchProgressTracker:
    """
    Progress tracker for batch operations with multiple items.
    """

    def __init__(self, operation: str, total_items: int, item_name: str = "item"):
        """
        Initialize batch progress tracker.

        Args:
            operation: Name of the batch operation
            total_items: Total number of items to process
            item_name: Name of individual items (for messages)
        """
        self.operation = operation
        self.total_items = total_items
        self.completed_items = 0
        self.item_name = item_name
        self.start_time = time.time()
        self.callbacks: list[Callable[[ProgressInfo], None]] = []

    def add_callback(self, callback: Callable[[ProgressInfo], None]) -> None:
        """Add a progress callback."""
        self.callbacks.append(callback)

    def item_completed(self, message: str = "") -> None:
        """Mark an item as completed."""
        self.completed_items += 1
        progress = (self.completed_items / self.total_items) * 100

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.completed_items > 0:
            avg_time_per_item = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta = avg_time_per_item * remaining_items
        else:
            eta = None

        default_message = f"Processed {self.completed_items}/{self.total_items} {self.item_name}s"
        if message:
            default_message += f" - {message}"

        info = ProgressInfo(
            operation=self.operation,
            current=progress,
            total=100.0,
            message=default_message,
            start_time=self.start_time,
            estimated_time_remaining=eta,
        )

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.warning(f"Batch progress callback failed: {e}")

    def get_progress_info(self) -> ProgressInfo:
        """Get current progress information."""
        progress = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 100
        elapsed = time.time() - self.start_time

        if self.completed_items > 0:
            avg_time_per_item = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta = avg_time_per_item * remaining_items
        else:
            eta = None

        return ProgressInfo(
            operation=self.operation,
            current=progress,
            total=100.0,
            message=f"Processed {self.completed_items}/{self.total_items} {self.item_name}s",
            start_time=self.start_time,
            estimated_time_remaining=eta,
        )
