"""
Rate limiter implementations for Bigdata.com API.

- RateLimiter: legacy token-bucket (200 RPM default)
- AsyncSlidingWindowRateLimiter: sliding-window with burst prevention (460 RPM default,
  matches Search_Large_Scale cookbook pattern)
- AsyncConcurrencyLimiter: caps simultaneous in-flight HTTP connections
"""

import asyncio
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API rate limiting.
    
    Attributes:
        max_tokens: Maximum number of tokens (200 for 200 RPM)
        refill_rate: Tokens added per second (200/60 = 3.33 per second)
        tokens: Current available tokens
        last_refill: Last time tokens were refilled
    """
    
    def __init__(self, max_tokens: int = 200, refill_period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_tokens: Maximum tokens in bucket (default: 200 for 200 RPM)
            refill_period: Period in seconds to refill bucket (default: 60 for per minute)
        """
        self.max_tokens = max_tokens
        self.refill_rate = max_tokens / refill_period  # tokens per second
        self.tokens = float(max_tokens)  # Start with full bucket
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.throttle_events = 0
        
        logger.info(
            f"RateLimiter initialized: {max_tokens} tokens, "
            f"{self.refill_rate:.2f} tokens/sec"
        )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed since last refill."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket. Waits if insufficient tokens available.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            Wait time in seconds (0 if no wait was needed)
        """
        total_wait_time = 0.0
        while True:
            async with self._lock:
                self._refill_tokens()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.total_requests += 1
                    self.total_wait_time += total_wait_time
                    if total_wait_time > 0:
                        logger.info(
                            f"Acquired {tokens} token(s) after {total_wait_time:.2f}s wait"
                        )
                    return total_wait_time

                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                logger.debug(
                    f"Insufficient tokens ({self.tokens:.2f}/{tokens}). "
                    f"Waiting {wait_time:.2f}s"
                )

            # Sleep outside the lock so other coroutines can acquire in parallel
            self.throttle_events += 1
            await asyncio.sleep(wait_time)
            total_wait_time += wait_time
    
    async def acquire_many(self, count: int) -> float:
        """
        Acquire multiple tokens (convenience method).
        
        Args:
            count: Number of tokens to acquire
            
        Returns:
            Total wait time in seconds
        """
        return await self.acquire(count)
    
    def get_available_tokens(self) -> float:
        """
        Get current number of available tokens.
        
        Returns:
            Number of available tokens
        """
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        return min(self.max_tokens, self.tokens + tokens_to_add)
    
    def get_metrics(self) -> dict:
        """
        Get rate limiter metrics.
        
        Returns:
            Dictionary with metrics (total requests, wait time, throttle events, etc.)
        """
        avg_wait = (
            self.total_wait_time / self.total_requests 
            if self.total_requests > 0 
            else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "average_wait_time_seconds": round(avg_wait, 3),
            "throttle_events": self.throttle_events,
            "current_tokens": round(self.get_available_tokens(), 2),
            "max_tokens": self.max_tokens,
            "refill_rate_per_second": round(self.refill_rate, 2),
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.throttle_events = 0
        logger.info("Rate limiter metrics reset")
    
    def reset(self) -> None:
        """Reset the rate limiter to initial state (full bucket)."""
        self.tokens = float(self.max_tokens)
        self.last_refill = time.time()
        self.reset_metrics()
        logger.info("Rate limiter fully reset")


class AsyncSlidingWindowRateLimiter:
    """
    Async sliding-window rate limiter with burst prevention.

    Mirrors the Search_Large_Scale cookbook pattern: enforces both a per-minute
    cap and a per-window cap so requests are spread evenly instead of bursting.
    """

    def __init__(
        self,
        max_requests: int = 460,
        period_seconds: int = 60,
        window_size: int = 5,
    ) -> None:
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.window_size = window_size
        self.max_per_window = max(1, int(max_requests * window_size / period_seconds))
        self.request_times: deque[float] = deque()
        self._lock = asyncio.Lock()
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.throttle_events = 0
        self.rate_limit_warnings = 0
        logger.info(
            f"AsyncSlidingWindowRateLimiter: {max_requests} req/{period_seconds}s, "
            f"max {self.max_per_window} per {window_size}s window"
        )

    def _clean_old_requests(self, current_time: float) -> None:
        cutoff_time = current_time - self.period_seconds
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()

    def _requests_in_window(self, current_time: float) -> int:
        window_start = current_time - self.window_size
        return sum(1 for t in self.request_times if t >= window_start)

    async def acquire(self, timeout: float = 120.0) -> float:
        """Acquire permission to make one request. Returns total wait time in seconds."""
        start_time = time.time()
        total_wait = 0.0
        while True:
            wait_time = 0.0
            async with self._lock:
                current_time = time.time()
                self._clean_old_requests(current_time)
                requests_in_period = len(self.request_times)
                requests_in_window = self._requests_in_window(current_time)
                if (
                    requests_in_period < self.max_requests
                    and requests_in_window < self.max_per_window
                ):
                    self.request_times.append(current_time)
                    self.total_requests += 1
                    self.total_wait_time += total_wait
                    return total_wait
                self.throttle_events += 1
                if requests_in_window >= self.max_per_window:
                    wait_time = self.window_size / 10
                elif self.request_times:
                    oldest_request = self.request_times[0]
                    wait_time = (oldest_request + self.period_seconds - current_time) + 0.1
                else:
                    wait_time = 0.1
                if time.time() - start_time > timeout:
                    raise TimeoutError("Rate limiter timeout exceeded")
            await asyncio.sleep(min(max(wait_time, 0.05), 1.0))
            total_wait += min(max(wait_time, 0.05), 1.0)

    def get_metrics(self) -> dict:
        avg_wait = (
            self.total_wait_time / self.total_requests if self.total_requests > 0 else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "average_wait_time_seconds": round(avg_wait, 3),
            "throttle_events": self.throttle_events,
            "rate_limit_warnings": self.rate_limit_warnings,
            "current_requests_in_period": len(self.request_times),
            "max_requests_per_period": self.max_requests,
            "max_requests_per_window": self.max_per_window,
        }


class AsyncConcurrencyLimiter:
    """Limits simultaneous in-flight HTTP connections (async context manager)."""

    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self.active_count = 0
        self.total_acquisitions = 0
        self.peak_concurrent = 0
        logger.info(f"AsyncConcurrencyLimiter: max {max_concurrent} simultaneous connections")

    async def __aenter__(self) -> "AsyncConcurrencyLimiter":
        await self._semaphore.acquire()
        async with self._lock:
            self.active_count += 1
            self.total_acquisitions += 1
            self.peak_concurrent = max(self.peak_concurrent, self.active_count)
        return self

    async def __aexit__(self, *args: object) -> None:
        async with self._lock:
            self.active_count -= 1
        self._semaphore.release()

    def get_metrics(self) -> dict:
        return {
            "active_connections": self.active_count,
            "max_concurrent": self.max_concurrent,
            "total_acquisitions": self.total_acquisitions,
            "peak_concurrent": self.peak_concurrent,
        }

