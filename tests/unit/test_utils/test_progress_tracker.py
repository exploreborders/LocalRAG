"""
Unit tests for ProgressTracker classes.

Tests progress tracking, callbacks, sub-operations, and batch processing.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.utils.progress_tracker import (
    BatchProgressTracker,
    ProgressInfo,
    ProgressTracker,
    create_progress_callback,
    track_progress,
)


class TestProgressInfo:
    """Test the ProgressInfo dataclass."""

    def test_progress_ratio(self):
        """Test progress ratio calculation."""
        info = ProgressInfo(operation="test", current=50.0, total=100.0)
        assert info.progress_ratio == 0.5

        info_zero = ProgressInfo(operation="test", current=0.0, total=100.0)
        assert info_zero.progress_ratio == 0.0

        info_complete = ProgressInfo(operation="test", current=100.0, total=100.0)
        assert info_complete.progress_ratio == 1.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start_time = time.time() - 5  # 5 seconds ago
        info = ProgressInfo(operation="test", current=50.0, start_time=start_time)
        assert abs(info.elapsed_time - 5.0) < 0.1  # Allow small timing variance

    def test_update_eta(self):
        """Test ETA calculation."""
        info = ProgressInfo(operation="test", current=50.0, start_time=time.time())

        # No completed items yet
        info.update_eta(10, 0)
        assert info.estimated_time_remaining is None

        # With completed items
        time.sleep(0.01)  # Small delay to ensure elapsed time > 0
        info.update_eta(10, 2)  # 2 out of 10 completed
        assert info.estimated_time_remaining is not None
        assert info.estimated_time_remaining > 0


class TestProgressTracker:
    """Test the ProgressTracker class."""

    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker("test_operation", 100)

        assert tracker.operation == "test_operation"
        assert tracker.total_steps == 100
        assert tracker.current_step == 0
        assert tracker.callbacks == []
        assert tracker.sub_operations == {}
        assert tracker.estimated_time_remaining is None

    def test_add_callback(self):
        """Test adding progress callbacks."""
        tracker = ProgressTracker("test")
        callback = MagicMock()

        tracker.add_callback(callback)
        assert callback in tracker.callbacks

    def test_update_increment(self):
        """Test progress update with increment."""
        tracker = ProgressTracker("test", 10)
        callback = MagicMock()
        tracker.add_callback(callback)

        tracker.update(message="Step 1")
        assert tracker.current_step == 1

        # Check callback was called
        assert callback.call_count == 1
        args = callback.call_args[0][0]
        assert args.operation == "test"
        assert args.current == 10.0  # 1/10 * 100
        assert args.message == "Step 1"

    def test_update_specific_step(self):
        """Test progress update with specific step."""
        tracker = ProgressTracker("test", 10)
        callback = MagicMock()
        tracker.add_callback(callback)

        tracker.update(step=5, message="Halfway")
        assert tracker.current_step == 5

        args = callback.call_args[0][0]
        assert args.current == 50.0  # 5/10 * 100

    def test_update_no_increment(self):
        """Test progress update without increment."""
        tracker = ProgressTracker("test", 10)

        tracker.update(step=3, increment=False)
        assert tracker.current_step == 3

        tracker.update(increment=False)  # Should not increment
        assert tracker.current_step == 3

    def test_start_sub_operation(self):
        """Test starting sub-operations."""
        tracker = ProgressTracker("main", 10)
        sub_tracker = tracker.start_sub_operation("sub", 5)

        assert "sub" in tracker.sub_operations
        assert tracker.sub_operations["sub"] == sub_tracker
        assert sub_tracker.operation == "main:sub"
        assert sub_tracker.total_steps == 5

    def test_complete_sub_operation(self):
        """Test completing sub-operations."""
        tracker = ProgressTracker("main", 10)
        tracker.start_sub_operation("sub", 5)

        assert "sub" in tracker.sub_operations

        tracker.complete_sub_operation("sub")
        assert "sub" not in tracker.sub_operations

    def test_get_progress_info(self):
        """Test getting progress information."""
        tracker = ProgressTracker("test", 10)
        tracker.current_step = 3

        info = tracker.get_progress_info()
        assert info.operation == "test"
        assert info.current == 30.0  # 3/10 * 100
        assert info.total == 100.0
        assert info.message == ""

    def test_reset(self):
        """Test resetting the progress tracker."""
        tracker = ProgressTracker("test", 10)
        tracker.current_step = 5
        tracker.start_sub_operation("sub", 5)

        tracker.reset()

        assert tracker.current_step == 0
        assert tracker.sub_operations == {}

    def test_callback_error_handling(self):
        """Test that callback errors don't break progress tracking."""
        tracker = ProgressTracker("test", 10)

        def failing_callback(info):
            raise Exception("Callback failed")

        mock_callback = MagicMock()
        tracker.add_callback(failing_callback)
        tracker.add_callback(mock_callback)  # This one should still work

        # Should not raise exception
        tracker.update()

        # Second callback should have been called
        assert mock_callback.call_count == 1


class TestBatchProgressTracker:
    """Test the BatchProgressTracker class."""

    def test_init(self):
        """Test BatchProgressTracker initialization."""
        tracker = BatchProgressTracker("batch_test", 10, "file")

        assert tracker.operation == "batch_test"
        assert tracker.total_items == 10
        assert tracker.completed_items == 0
        assert tracker.item_name == "file"
        assert tracker.callbacks == []

    def test_item_completed(self):
        """Test marking items as completed."""
        tracker = BatchProgressTracker("batch_test", 10)
        callback = MagicMock()
        tracker.add_callback(callback)

        tracker.item_completed("File processed")

        assert tracker.completed_items == 1

        # Check callback was called
        assert callback.call_count == 1
        args = callback.call_args[0][0]
        assert args.operation == "batch_test"
        assert args.current == 10.0  # 1/10 * 100
        assert "Processed 1/10 items - File processed" in args.message

    def test_get_progress_info(self):
        """Test getting batch progress information."""
        tracker = BatchProgressTracker("batch_test", 10)
        tracker.completed_items = 3

        info = tracker.get_progress_info()
        assert info.operation == "batch_test"
        assert info.current == 30.0  # 3/10 * 100
        assert "Processed 3/10 items" in info.message

    def test_get_progress_info_with_eta(self):
        """Test ETA calculation in batch progress."""
        tracker = BatchProgressTracker("batch_test", 10)
        tracker.completed_items = 2

        # Simulate some elapsed time
        tracker.start_time = time.time() - 2.0  # 2 seconds elapsed

        info = tracker.get_progress_info()
        assert info.estimated_time_remaining is not None
        # With 2 items in 2 seconds, should estimate 8 seconds for remaining 8 items
        assert abs(info.estimated_time_remaining - 8.0) < 1.0


class TestProgressUtilities:
    """Test progress utility functions."""

    def test_create_progress_callback(self):
        """Test creating a progress callback."""
        callback = create_progress_callback("test_operation")

        # Callback should be callable
        assert callable(callback)

        # Test the callback
        info = ProgressInfo(
            operation="test_operation",
            current=50.0,
            estimated_time_remaining=5.0,
        )

        # Should not raise exception
        callback(info)

    def test_track_progress_context_manager(self):
        """Test the track_progress context manager."""
        callback = MagicMock()

        with track_progress("test_operation", 10, callback) as tracker:
            assert isinstance(tracker, ProgressTracker)
            assert tracker.operation == "test_operation"
            assert tracker.total_steps == 10

            # Check initial callback
            assert callback.call_count == 1
            args = callback.call_args[0][0]
            assert "Starting test_operation" in args.message

        # Check completion callback
        assert callback.call_count == 2
        args = callback.call_args[0][0]
        assert "Completed test_operation" in args.message

    def test_track_progress_exception_handling(self):
        """Test exception handling in track_progress."""
        callback = MagicMock()

        with pytest.raises(ValueError):
            with track_progress("test_operation", 10, callback) as _:
                # Check initial callback
                assert callback.call_count == 1

                # Raise exception
                raise ValueError("Test error")

        # Check error callback
        assert callback.call_count == 2
        args = callback.call_args[0][0]
        assert "Failed test_operation: Test error" in args.message
