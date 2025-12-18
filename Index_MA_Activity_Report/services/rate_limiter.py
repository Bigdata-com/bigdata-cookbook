"""
Rate limiter implementation using token bucket algorithm
Handles 200 requests per minute (RPM) limit for Bigdata API
"""

import asyncio
import time
import logging
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
        async with self._lock:
            self._refill_tokens()
            
            # Calculate wait time if insufficient tokens
            wait_time = 0.0
            if self.tokens < tokens:
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                
                logger.debug(
                    f"Insufficient tokens ({self.tokens:.2f}/{tokens}). "
                    f"Waiting {wait_time:.2f}s"
                )
                
                self.throttle_events += 1
                
                # Wait for tokens to refill
                await asyncio.sleep(wait_time)
                
                # Refill after waiting
                self._refill_tokens()
            
            # Consume tokens
            self.tokens -= tokens
            
            # Update metrics
            self.total_requests += 1
            self.total_wait_time += wait_time
            
            if wait_time > 0:
                logger.info(f"Acquired {tokens} token(s) after {wait_time:.2f}s wait")
            
            return wait_time
    
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

