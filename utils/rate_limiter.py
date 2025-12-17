"""
Rate Limiter for Gemini API Calls

Implements token-based rate limiting to prevent 429 errors.
Tracks input tokens per minute using a sliding window approach.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from decouple import config

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Represents token usage at a point in time."""
    tokens: int
    timestamp: float


class RateLimiter:
    """
    Token-based rate limiter for Gemini API calls.
    
    Uses a sliding window to track tokens per minute and throttles requests
    when approaching the limit.
    """
    
    def __init__(
        self,
        max_tokens_per_minute: int = 900000,  # 90% of 1M limit for safety margin
        window_seconds: int = 60,
        safety_margin: float = 0.1,  # 10% safety margin
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_tokens_per_minute: Maximum tokens allowed per minute (default: 900k for safety)
            window_seconds: Time window in seconds (default: 60)
            safety_margin: Safety margin as fraction (default: 0.1 = 10%)
        """
        self.max_tokens_per_minute = max_tokens_per_minute
        self.window_seconds = window_seconds
        self.safety_margin = safety_margin
        self.effective_limit = int(max_tokens_per_minute * (1 - safety_margin))
        
        # Sliding window: deque of (timestamp, tokens) tuples
        self._usage_window: deque = deque()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._total_tokens_used = 0
        self._total_requests_throttled = 0
        
        logger.info(
            f"RateLimiter initialized: max={max_tokens_per_minute:,} tokens/min, "
            f"effective_limit={self.effective_limit:,} tokens/min (with {safety_margin*100:.0f}% margin)"
        )
    
    async def acquire(self, estimated_tokens: int) -> None:
        """
        Acquire permission to make an API call with estimated token count.
        
        This will block if the request would exceed the rate limit, waiting
        until enough tokens are available.
        
        Args:
            estimated_tokens: Estimated number of input tokens for this request
        """
        async with self._lock:
            # Clean up old entries outside the window
            self._cleanup_window()
            
            # Calculate current usage in the window
            current_usage = sum(usage.tokens for usage in self._usage_window)
            
            # Check if this request would exceed the limit
            if current_usage + estimated_tokens > self.effective_limit:
                # Calculate how long to wait
                wait_time = self._calculate_wait_time(current_usage, estimated_tokens)
                
                if wait_time > 0:
                    self._total_requests_throttled += 1
                    logger.warning(
                        f"Rate limit approaching: {current_usage:,}/{self.effective_limit:,} tokens used. "
                        f"Waiting {wait_time:.1f}s before allowing {estimated_tokens:,} token request."
                    )
                    
                    # Release lock before sleeping
                    await asyncio.sleep(wait_time)
                    
                    # Re-acquire lock and clean up again after waiting
                    self._cleanup_window()
            
            # Record this usage
            self._usage_window.append(TokenUsage(
                tokens=estimated_tokens,
                timestamp=time.time()
            ))
            self._total_tokens_used += estimated_tokens
    
    def _cleanup_window(self) -> None:
        """Remove entries outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        while self._usage_window and self._usage_window[0].timestamp < cutoff_time:
            self._usage_window.popleft()
    
    def _calculate_wait_time(self, current_usage: int, requested_tokens: int) -> float:
        """
        Calculate how long to wait before allowing the request.
        
        Args:
            current_usage: Current token usage in the window
            requested_tokens: Tokens requested for this call
            
        Returns:
            Wait time in seconds
        """
        if not self._usage_window:
            return 0.0
        
        # Find the oldest entry we need to wait for
        oldest_entry = self._usage_window[0]
        oldest_time = oldest_entry.timestamp
        
        # Calculate when the oldest entry will expire
        current_time = time.time()
        time_until_expiry = self.window_seconds - (current_time - oldest_time)
        
        # If we wait until the oldest expires, we'll have more room
        # But we need to ensure we have enough room for this request
        tokens_to_expire = oldest_entry.tokens
        remaining_after_expiry = current_usage - tokens_to_expire
        
        if remaining_after_expiry + requested_tokens <= self.effective_limit:
            return max(0.0, time_until_expiry)
        
        # Need to wait longer - calculate based on multiple entries
        # Simple approach: wait until we have enough room
        total_to_expire = 0
        wait_time = 0.0
        
        for usage in self._usage_window:
            time_until_this_expires = self.window_seconds - (current_time - usage.timestamp)
            if time_until_this_expires > wait_time:
                total_to_expire += usage.tokens
                wait_time = time_until_this_expires
                
                if current_usage - total_to_expire + requested_tokens <= self.effective_limit:
                    return wait_time
        
        # Fallback: wait until window clears
        return max(0.0, self.window_seconds - (current_time - oldest_time))
    
    def get_current_usage(self) -> int:
        """Get current token usage in the window."""
        self._cleanup_window()
        return sum(usage.tokens for usage in self._usage_window)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "current_usage": self.get_current_usage(),
            "effective_limit": self.effective_limit,
            "max_limit": self.max_tokens_per_minute,
            "total_tokens_used": self._total_tokens_used,
            "total_requests_throttled": self._total_requests_throttled,
            "window_size": len(self._usage_window),
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        max_tokens = config("GEMINI_MAX_TOKENS_PER_MINUTE", default=900000, cast=int)
        _rate_limiter = RateLimiter(max_tokens_per_minute=max_tokens)
    
    return _rate_limiter


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    
    Uses a simple heuristic: ~4 characters per token for English text.
    This is approximate but sufficient for rate limiting purposes.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimate: 4 characters per token for English text
    # This is conservative and works well for rate limiting
    return len(text) // 4

