"""
Unit tests for RateLimiter classes.

Tests token bucket rate limiting, adaptive rate limiting, and circuit breaker functionality.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils.rate_limiter import (
    AdaptiveRateLimiter,
    CircuitBreaker,
    CircuitBreakerOpen,
    RateLimiter,
    RateLimitInfo,
    rate_limit_context,
    rate_limited,
)


class TestRateLimitInfo:
    """Test the RateLimitInfo dataclass."""

    def test_creation(self):
        """Test RateLimitInfo creation."""
        info = RateLimitInfo(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=10,
            current_minute_count=5,
            current_hour_count=50,
            current_burst_count=2,
        )

        assert info.requests_per_minute == 60
        assert info.requests_per_hour == 1000
        assert info.burst_limit == 10
        assert info.current_minute_count == 5
        assert info.current_hour_count == 50
        assert info.current_burst_count == 2


class TestRateLimiter:
    """Test the RateLimiter class functionality."""

    def test_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_limit=5,
            burst_window_seconds=2,
        )

        assert limiter.requests_per_minute == 30
        assert limiter.requests_per_hour == 500
        assert limiter.burst_limit == 5
        assert limiter.burst_window_seconds == 2
        assert limiter.minute_count == 0
        assert limiter.hour_count == 0
        assert limiter.burst_count == 0

    def test_acquire_within_limits(self):
        """Test token acquisition within limits."""
        limiter = RateLimiter(requests_per_minute=10, burst_limit=3)

        # Should succeed
        assert limiter.acquire() is True
        assert limiter.acquire(2) is True
        assert limiter.minute_count == 3
        assert limiter.burst_count == 3

    def test_acquire_burst_limit_exceeded(self):
        """Test token acquisition when burst limit exceeded."""
        limiter = RateLimiter(burst_limit=2)

        # Use up burst tokens
        assert limiter.acquire(2) is True
        # This should fail
        assert limiter.acquire() is False

    def test_acquire_minute_limit_exceeded(self):
        """Test token acquisition when minute limit exceeded."""
        limiter = RateLimiter(requests_per_minute=2)

        assert limiter.acquire(2) is True
        assert limiter.acquire() is False

    def test_acquire_hour_limit_exceeded(self):
        """Test token acquisition when hour limit exceeded."""
        limiter = RateLimiter(requests_per_hour=2)

        assert limiter.acquire(2) is True
        assert limiter.acquire() is False

    def test_window_reset(self):
        """Test window reset functionality."""
        limiter = RateLimiter(requests_per_minute=10, burst_window_seconds=0.1)

        # Use some tokens
        assert limiter.acquire(3) is True
        assert limiter.minute_count == 3
        assert limiter.burst_count == 3

        # Wait for burst window to reset (0.1s)
        time.sleep(0.15)

        # Burst count should reset, but minute count should remain
        assert limiter.acquire(3) is True  # Should work since burst reset
        assert limiter.minute_count == 6
        assert limiter.burst_count == 3

    def test_wait_for_tokens_success(self):
        """Test waiting for tokens successfully."""
        limiter = RateLimiter(burst_limit=1)

        # Use up tokens
        assert limiter.acquire() is True

        # This should wait and succeed when burst window resets
        result = limiter.wait_for_tokens(timeout=1.0)
        assert result is True

    def test_wait_for_tokens_timeout(self):
        """Test waiting for tokens with timeout."""
        limiter = RateLimiter(requests_per_minute=1)

        # Use up tokens
        assert limiter.acquire() is True

        # This should timeout since minute window is long
        result = limiter.wait_for_tokens(timeout=0.1)
        assert result is False

    def test_get_rate_limit_info(self):
        """Test getting rate limit information."""
        limiter = RateLimiter(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=10,
        )

        limiter.minute_count = 5
        limiter.hour_count = 50
        limiter.burst_count = 2

        info = limiter.get_rate_limit_info()

        assert isinstance(info, RateLimitInfo)
        assert info.requests_per_minute == 60
        assert info.requests_per_hour == 1000
        assert info.burst_limit == 10
        assert info.current_minute_count == 5
        assert info.current_hour_count == 50
        assert info.current_burst_count == 2

    def test_thread_safety(self):
        """Test thread safety of rate limiter."""
        limiter = RateLimiter(requests_per_minute=100, burst_limit=10)

        results = []

        def worker():
            for _ in range(5):
                if limiter.acquire():
                    results.append(True)
                else:
                    results.append(False)

        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have exactly 10 successful acquisitions (burst limit)
        assert results.count(True) == 10
        assert results.count(False) == 15  # 25 - 10


class TestAdaptiveRateLimiter:
    """Test the AdaptiveRateLimiter class functionality."""

    def test_init(self):
        """Test AdaptiveRateLimiter initialization."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        assert limiter.requests_per_minute == 50
        assert limiter.success_count == 0
        assert limiter.failure_count == 0
        assert limiter.adjustment_window == 300
        assert limiter.base_requests_per_minute == 50

    def test_record_success(self):
        """Test recording successful requests."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        limiter.record_success()
        assert limiter.success_count == 1
        assert limiter.failure_count == 0

    def test_record_failure(self):
        """Test recording failed requests."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        limiter.record_failure()
        assert limiter.success_count == 0
        assert limiter.failure_count == 1

    def test_adaptive_increase_high_success_rate(self):
        """Test rate limit increase with high success rate."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        # Simulate high success rate
        limiter.success_count = 95
        limiter.failure_count = 5
        limiter.last_adjustment = time.time() - 400  # Force adjustment

        limiter.record_success()  # Trigger adjustment

        # Should increase rate limit
        assert limiter.requests_per_minute > 50

    def test_adaptive_decrease_low_success_rate(self):
        """Test rate limit decrease with low success rate."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        # Simulate low success rate
        limiter.success_count = 3
        limiter.failure_count = 7
        limiter.last_adjustment = time.time() - 400  # Force adjustment

        limiter.record_failure()  # Trigger adjustment

        # Should decrease rate limit
        assert limiter.requests_per_minute < 50

    def test_no_adjustment_insufficient_data(self):
        """Test no adjustment with insufficient data."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        # Insufficient data
        limiter.success_count = 3
        limiter.failure_count = 2
        original_rate = limiter.requests_per_minute

        limiter.record_success()

        # Should not change
        assert limiter.requests_per_minute == original_rate

    def test_no_adjustment_recent_adjustment(self):
        """Test no adjustment when recently adjusted."""
        limiter = AdaptiveRateLimiter(requests_per_minute=50)

        limiter.success_count = 50
        limiter.failure_count = 0
        limiter.last_adjustment = time.time()  # Very recent

        original_rate = limiter.requests_per_minute
        limiter.record_success()

        # Should not change due to recent adjustment
        assert limiter.requests_per_minute == original_rate


class TestCircuitBreaker:
    """Test the CircuitBreaker class functionality."""

    def test_init(self):
        """Test CircuitBreaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_successful_call(self):
        """Test successful function call."""
        cb = CircuitBreaker()

        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_failure_then_recovery(self):
        """Test failure handling and recovery."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 1
        assert cb.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 2
        assert cb.state == "open"

        # Call while open should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "should not execute")

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next call should be in half-open state
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_get_state(self):
        """Test getting circuit breaker state."""
        cb = CircuitBreaker()

        state = cb.get_state()
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert "last_failure_time" in state
        assert "time_since_last_failure" in state


class TestRateLimiterDecorators:
    """Test rate limiter decorators and context managers."""

    def test_rate_limited_decorator_success(self):
        """Test rate limited decorator with successful calls."""
        limiter = RateLimiter(requests_per_minute=10)

        @rate_limited(limiter, tokens=1, wait=False)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_rate_limited_decorator_rate_limited(self):
        """Test rate limited decorator when rate limited."""
        limiter = RateLimiter(requests_per_minute=1)
        limiter.acquire()  # Use up the token

        @rate_limited(limiter, tokens=1, wait=False)
        def test_func():
            return "success"

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            test_func()

    def test_rate_limited_decorator_with_wait(self):
        """Test rate limited decorator with waiting."""
        limiter = RateLimiter(burst_limit=1, burst_window_seconds=1)

        @rate_limited(limiter, tokens=1, wait=True, timeout=1.0)
        def test_func():
            return "success"

        # First call should succeed
        result1 = test_func()
        assert result1 == "success"

        # Second call should wait and succeed after burst reset
        result2 = test_func()
        assert result2 == "success"

    def test_rate_limit_context_success(self):
        """Test rate limit context manager success."""
        limiter = RateLimiter(requests_per_minute=10)

        with rate_limit_context(limiter, tokens=1, wait=False):
            assert True  # Should not raise

    def test_rate_limit_context_rate_limited(self):
        """Test rate limit context manager when rate limited."""
        limiter = RateLimiter(requests_per_minute=1)
        limiter.acquire()  # Use up token

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            with rate_limit_context(limiter, tokens=1, wait=False):
                pass

    def test_adaptive_rate_limiter_integration(self):
        """Test adaptive rate limiter with decorator."""
        limiter = AdaptiveRateLimiter(requests_per_minute=10)

        @rate_limited(limiter, tokens=1, wait=False)
        def test_func():
            return "success"

        # Successful call
        result = test_func()
        assert result == "success"

        # Check that success was recorded
        assert limiter.success_count == 1

    def test_adaptive_rate_limiter_failure_recording(self):
        """Test adaptive rate limiter failure recording."""
        limiter = AdaptiveRateLimiter(requests_per_minute=10)

        @rate_limited(limiter, tokens=1, wait=False)
        def failing_func():
            raise ValueError("test error")

        # This should record a failure
        with pytest.raises(ValueError):
            failing_func()

        assert limiter.failure_count == 1
