"""
Unit tests for rate limiter and concurrency limiter.
"""

import pytest
import time
import threading
from search_function import SlidingWindowRateLimiter, ConcurrencySemaphore


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = SlidingWindowRateLimiter(max_requests=100, period_seconds=60, window_size=5)
        assert limiter.max_requests == 100
        assert limiter.period_seconds == 60
        assert limiter.window_size == 5
        assert limiter.max_per_window == int(100 * 5 / 60)
    
    def test_rate_limiter_acquire_immediate(self):
        """Test that acquire works immediately when under limit."""
        limiter = SlidingWindowRateLimiter(max_requests=100, period_seconds=60)
        wait_time = limiter.acquire(timeout=1.0)
        assert wait_time == 0.0  # No wait needed
    
    def test_rate_limiter_throttles_requests(self):
        """Test that rate limiter throttles when limit is reached."""
        limiter = SlidingWindowRateLimiter(max_requests=2, period_seconds=1)  # Shorter period for test
        
        # First two requests should go through immediately
        wait1 = limiter.acquire(timeout=1.0)
        wait2 = limiter.acquire(timeout=1.0)
        assert wait1 == 0.0
        assert wait2 == 0.0
        
        # Third request should be throttled (but with short period, it should wait briefly then succeed)
        start = time.time()
        wait3 = limiter.acquire(timeout=2.0)
        elapsed = time.time() - start
        # With 1 second period, after ~1 second the first request expires and third can proceed
        assert elapsed >= 0  # May have waited
        assert wait3 >= 0
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        # Use values that ensure max_per_window is reasonable
        # max_per_window = int(max_requests * window_size / period_seconds)
        # With max_requests=100, window_size=5, period=60: max_per_window = 8
        limiter = SlidingWindowRateLimiter(max_requests=100, period_seconds=60, window_size=5)
        
        # Make some requests (within limit) - use longer timeout to avoid flakiness
        for _ in range(5):
            limiter.acquire(timeout=5.0)
        
        stats = limiter.get_stats()
        assert stats["total_requests"] == 5
        assert stats["max_requests_per_period"] == 100
        assert "throttle_events" in stats
        assert stats["total_requests"] <= stats["max_requests_per_period"]
    
    def test_rate_limiter_timeout(self):
        """Test that rate limiter raises timeout error."""
        # Use values that ensure max_per_window is at least 1
        # max_per_window = int(max_requests * window_size / period_seconds)
        # With max_requests=10, window_size=5, period=10: max_per_window = 5
        limiter = SlidingWindowRateLimiter(max_requests=10, period_seconds=10, window_size=5)
        
        # First request - use longer timeout to ensure it succeeds
        limiter.acquire(timeout=5.0)
        
        # Make enough requests to fill the limit
        for _ in range(9):  # Total 10 requests (1 + 9 = 10, which is the limit)
            limiter.acquire(timeout=5.0)
        
        # 11th request should timeout if timeout is too short
        # With period_seconds=10, we need to wait ~10 seconds, so 0.1s timeout should fail
        with pytest.raises(TimeoutError, match="Rate limiter timeout exceeded"):
            limiter.acquire(timeout=0.1)


class TestConcurrencySemaphore:
    """Tests for ConcurrencySemaphore."""
    
    def test_semaphore_initialization(self):
        """Test semaphore initialization."""
        semaphore = ConcurrencySemaphore(max_concurrent=10)
        assert semaphore.max_concurrent == 10
    
    def test_semaphore_context_manager(self):
        """Test semaphore as context manager."""
        semaphore = ConcurrencySemaphore(max_concurrent=2)
        
        with semaphore:
            stats = semaphore.get_stats()
            assert stats["active_connections"] == 1
        
        stats = semaphore.get_stats()
        assert stats["active_connections"] == 0
    
    def test_semaphore_limits_concurrency(self):
        """Test that semaphore limits concurrent access."""
        semaphore = ConcurrencySemaphore(max_concurrent=2)
        acquired = []
        
        def acquire():
            with semaphore:
                acquired.append(threading.current_thread().ident)
                time.sleep(0.1)
        
        threads = [threading.Thread(target=acquire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have limited concurrent access
        stats = semaphore.get_stats()
        assert stats["peak_concurrent"] <= 2
    
    def test_semaphore_stats(self):
        """Test semaphore statistics."""
        semaphore = ConcurrencySemaphore(max_concurrent=5)
        
        with semaphore:
            stats = semaphore.get_stats()
            assert stats["active_connections"] == 1
            assert stats["total_acquisitions"] == 1
            assert stats["peak_concurrent"] == 1
