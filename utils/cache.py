"""
Multi-Level Cache System

Provides TTL-based caching for different data types:
- Geocoding results: 1 hour (locations rarely change)
- Demographics data: 30 minutes (data changes less frequently)
- Layer data: 5 minutes (may change more frequently)

Uses in-memory cache with automatic expiration.
Can be extended to use Redis for distributed caching.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)


# TTL constants (in seconds)
class CacheTTL:
    """Cache TTL values for different data types."""
    GEOCODE = 3600        # 1 hour - locations don't change
    DEMOGRAPHICS = 1800   # 30 minutes - data updates periodically
    LAYER = 300           # 5 minutes - may change more often
    SHORT = 60            # 1 minute - for rapidly changing data


@dataclass
class CacheEntry:
    """A single cache entry with value and expiration."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.expires_at

    def hit(self) -> Any:
        """Record a cache hit and return the value."""
        self.hits += 1
        return self.value


class MultiLevelCache:
    """
    In-memory cache with multiple TTL levels.

    Supports:
    - Different TTLs for different data types
    - Automatic expiration and cleanup
    - Cache statistics
    - Async-compatible operations
    """

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize the cache.

        Args:
            cleanup_interval: Seconds between automatic cleanup runs
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cache cleanup task started")

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Cache cleanup task stopped")

    async def _cleanup_loop(self):
        """Background loop to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    async def _cleanup_expired(self):
        """Remove all expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats["evictions"] += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    @staticmethod
    def _make_key(prefix: str, *args, **kwargs) -> str:
        """
        Create a cache key from prefix and arguments.

        Args:
            prefix: Cache key prefix (e.g., "geocode", "demographics")
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            Hashed cache key
        """
        # Build a stable string representation
        parts = [prefix]
        parts.extend(str(arg) for arg in args)
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        content = ":".join(parts)
        # Use short hash for memory efficiency
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return None
            self._stats["hits"] += 1
            return entry.hit()

    async def set(self, key: str, value: Any, ttl: int) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        async with self._lock:
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
            self._stats["sets"] += 1

    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": f"{hit_rate:.2%}",
        }


# Global cache instance
_cache = MultiLevelCache()


def get_cache() -> MultiLevelCache:
    """Get the global cache instance."""
    return _cache


# Convenience functions for common operations

async def cache_geocode(address: str, result: Dict) -> None:
    """Cache a geocoding result."""
    key = MultiLevelCache._make_key("geocode", address.lower().strip())
    await _cache.set(key, result, CacheTTL.GEOCODE)
    logger.debug(f"Cached geocode result for: {address[:50]}")


async def get_cached_geocode(address: str) -> Optional[Dict]:
    """Get a cached geocoding result."""
    key = MultiLevelCache._make_key("geocode", address.lower().strip())
    result = await _cache.get(key)
    if result:
        logger.debug(f"Cache hit for geocode: {address[:50]}")
    return result


async def cache_reverse_geocode(lat: float, lon: float, result: Dict) -> None:
    """Cache a reverse geocoding result."""
    # Round to 5 decimal places for cache key (about 1 meter precision)
    key = MultiLevelCache._make_key("reverse_geocode", f"{lat:.5f}", f"{lon:.5f}")
    await _cache.set(key, result, CacheTTL.GEOCODE)
    logger.debug(f"Cached reverse geocode result for: ({lat}, {lon})")


async def get_cached_reverse_geocode(lat: float, lon: float) -> Optional[Dict]:
    """Get a cached reverse geocoding result."""
    key = MultiLevelCache._make_key("reverse_geocode", f"{lat:.5f}", f"{lon:.5f}")
    result = await _cache.get(key)
    if result:
        logger.debug(f"Cache hit for reverse geocode: ({lat}, {lon})")
    return result


async def cache_demographics(lat: float, lon: float, result: Dict) -> None:
    """Cache demographics data."""
    key = MultiLevelCache._make_key("demographics", f"{lat:.4f}", f"{lon:.4f}")
    await _cache.set(key, result, CacheTTL.DEMOGRAPHICS)
    logger.debug(f"Cached demographics for: ({lat}, {lon})")


async def get_cached_demographics(lat: float, lon: float) -> Optional[Dict]:
    """Get cached demographics data."""
    key = MultiLevelCache._make_key("demographics", f"{lat:.4f}", f"{lon:.4f}")
    result = await _cache.get(key)
    if result:
        logger.debug(f"Cache hit for demographics: ({lat}, {lon})")
    return result


async def cache_layer_data(layer_id: str, query_hash: str, result: Any) -> None:
    """Cache layer query result."""
    key = MultiLevelCache._make_key("layer", layer_id, query_hash)
    await _cache.set(key, result, CacheTTL.LAYER)
    logger.debug(f"Cached layer data for: {layer_id}")


async def get_cached_layer_data(layer_id: str, query_hash: str) -> Optional[Any]:
    """Get cached layer query result."""
    key = MultiLevelCache._make_key("layer", layer_id, query_hash)
    result = await _cache.get(key)
    if result:
        logger.debug(f"Cache hit for layer: {layer_id}")
    return result


def cached(ttl: int, key_prefix: str):
    """
    Decorator for caching async function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key

    Usage:
        @cached(CacheTTL.GEOCODE, "geocode")
        async def geocode_address(address: str) -> Dict:
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function arguments
            key = MultiLevelCache._make_key(key_prefix, *args, **kwargs)

            # Try cache first
            result = await _cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {key_prefix}: {key[:16]}")
                return result

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await _cache.set(key, result, ttl)
                logger.debug(f"Cached result for {key_prefix}: {key[:16]}")

            return result
        return wrapper
    return decorator
