"""
Rate limiting utilities for API calls and resource-intensive operations.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitInfo:
    """Information about rate limiting."""

    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    current_minute_count: int = 0
    current_hour_count: int = 0
    current_burst_count: int = 0
    last_minute_reset: float = 0.0
    last_hour_reset: float = 0.0
    last_burst_reset: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter for controlling API call frequency.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        burst_window_seconds: int = 1,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
            burst_limit: Maximum burst requests
            burst_window_seconds: Burst window in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        self.burst_window_seconds = burst_window_seconds

        self.minute_window = 60.0
        self.hour_window = 3600.0

        self.minute_count = 0
        self.hour_count = 0
        self.burst_count = 0

        self.last_minute_reset = time.time()
        self.last_hour_reset = time.time()
        self.last_burst_reset = time.time()

        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens for a request.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        with self._lock:
            now = time.time()

            # Reset counters if windows have passed
            if now - self.last_minute_reset >= self.minute_window:
                self.minute_count = 0
                self.last_minute_reset = now

            if now - self.last_hour_reset >= self.hour_window:
                self.hour_count = 0
                self.last_hour_reset = now

            if now - self.last_burst_reset >= self.burst_window_seconds:
                self.burst_count = 0
                self.last_burst_reset = now

            # Check limits
            if (
                self.minute_count + tokens > self.requests_per_minute
                or self.hour_count + tokens > self.requests_per_hour
                or self.burst_count + tokens > self.burst_limit
            ):
                return False

            # Acquire tokens
            self.minute_count += tokens
            self.hour_count += tokens
            self.burst_count += tokens

            return True

    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait for tokens to become available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Wait a bit before trying again
            time.sleep(0.1)

    def get_rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limiting information."""
        with self._lock:
            now = time.time()

            # Reset counters if windows have passed (for accurate reporting)
            minute_count = self.minute_count
            hour_count = self.hour_count
            burst_count = self.burst_count

            if now - self.last_minute_reset >= self.minute_window:
                minute_count = 0
            if now - self.last_hour_reset >= self.hour_window:
                hour_count = 0
            if now - self.last_burst_reset >= self.burst_window_seconds:
                burst_count = 0

            return RateLimitInfo(
                requests_per_minute=self.requests_per_minute,
                requests_per_hour=self.requests_per_hour,
                burst_limit=self.burst_limit,
                current_minute_count=minute_count,
                current_hour_count=hour_count,
                current_burst_count=burst_count,
                last_minute_reset=self.last_minute_reset,
                last_hour_reset=self.last_hour_reset,
                last_burst_reset=self.last_burst_reset,
            )


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on success/failure rates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_count = 0
        self.failure_count = 0
        self.adjustment_window = 300  # 5 minutes
        self.last_adjustment = time.time()
        self.base_requests_per_minute = self.requests_per_minute

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self.success_count += 1
            self._adjust_limits()

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self.failure_count += 1
            self._adjust_limits()

    def _adjust_limits(self) -> None:
        """Adjust rate limits based on success/failure ratio."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_window:
            return

        total_requests = self.success_count + self.failure_count
        if total_requests < 10:  # Need minimum sample size
            return

        success_rate = self.success_count / total_requests

        # Adjust limits based on success rate
        if success_rate > 0.95:  # Very high success rate
            self.requests_per_minute = min(
                self.base_requests_per_minute * 1.2, self.base_requests_per_minute * 2
            )
        elif success_rate > 0.85:  # Good success rate
            self.requests_per_minute = min(
                self.base_requests_per_minute * 1.1, self.base_requests_per_minute * 1.5
            )
        elif success_rate < 0.7:  # Poor success rate
            self.requests_per_minute = max(self.base_requests_per_minute * 0.8, 10)
        elif success_rate < 0.5:  # Very poor success rate
            self.requests_per_minute = max(self.base_requests_per_minute * 0.5, 5)

        # Reset counters
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = now

        logger.info(
            f"Adjusted rate limit to {self.requests_per_minute} requests per minute (success rate: {success_rate:.2f})"
        )


# Global rate limiter instances
default_rate_limiter = RateLimiter()
adaptive_rate_limiter = AdaptiveRateLimiter()


def rate_limited(
    limiter: Optional[RateLimiter] = None,
    tokens: int = 1,
    wait: bool = True,
    timeout: Optional[float] = 30.0,
):
    """
    Decorator for rate limiting function calls.

    Args:
        limiter: Rate limiter instance to use
        tokens: Number of tokens the function consumes
        wait: Whether to wait for tokens if not available
        timeout: Maximum time to wait for tokens
    """
    limiter = limiter or default_rate_limiter

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if wait:
                if not limiter.wait_for_tokens(tokens, timeout):
                    raise TimeoutError(f"Rate limit timeout waiting for {tokens} tokens")
            else:
                if not limiter.acquire(tokens):
                    raise RuntimeError(f"Rate limit exceeded for {tokens} tokens")

            try:
                result = func(*args, **kwargs)
                if isinstance(limiter, AdaptiveRateLimiter):
                    limiter.record_success()
                return result
            except Exception as e:
                if isinstance(limiter, AdaptiveRateLimiter):
                    limiter.record_failure()
                raise

        return wrapper

    return decorator


@contextmanager
def rate_limit_context(
    limiter: Optional[RateLimiter] = None,
    tokens: int = 1,
    wait: bool = True,
    timeout: Optional[float] = 30.0,
):
    """
    Context manager for rate limiting blocks of code.

    Args:
        limiter: Rate limiter instance to use
        tokens: Number of tokens to acquire
        wait: Whether to wait for tokens
        timeout: Maximum time to wait
    """
    limiter = limiter or default_rate_limiter

    if wait:
        if not limiter.wait_for_tokens(tokens, timeout):
            raise TimeoutError(f"Rate limit timeout waiting for {tokens} tokens")
    else:
        if not limiter.acquire(tokens):
            raise RuntimeError(f"Rate limit exceeded for {tokens} tokens")

    try:
        yield
    except Exception as e:
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.record_failure()
        raise
    else:
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.record_success()


class CircuitBreaker:
    """
    Circuit breaker pattern for handling repeated failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying again
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call a function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpen("Circuit breaker is open")
                else:
                    self.state = "half_open"

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed - service recovered")

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": time.time() - self.last_failure_time,
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# Global circuit breaker instances
default_circuit_breaker = CircuitBreaker()


def with_circuit_breaker(
    breaker: Optional[CircuitBreaker] = None,
    expected_exception: Type[Exception] = Exception,
):
    """
    Decorator for circuit breaker protection.

    Args:
        breaker: Circuit breaker instance to use
        expected_exception: Exception type to catch
    """
    breaker = breaker or default_circuit_breaker
    breaker.expected_exception = expected_exception

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def get_rate_limiter(name: str = "default") -> RateLimiter:
    """Get a named rate limiter instance."""
    limiters = {
        "default": default_rate_limiter,
        "adaptive": adaptive_rate_limiter,
    }
    return limiters.get(name, default_rate_limiter)


def get_circuit_breaker(name: str = "default") -> CircuitBreaker:
    """Get a named circuit breaker instance."""
    breakers = {
        "default": default_circuit_breaker,
    }
    return breakers.get(name, default_circuit_breaker)
