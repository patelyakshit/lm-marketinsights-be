"""
Semantic Caching System

Caches responses based on semantic similarity of queries, not just exact text match.
This allows similar questions to hit the cache even with different wording:

Example:
- "Dallas population" → Cached
- "What's Dallas's population?" → HIT! (semantically similar)
- "How many people in Dallas?" → HIT! (semantically similar)

Uses embedding vectors and cosine similarity for matching.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedEntry:
    """A cached query-response pair with embedding."""
    query: str
    response: Any
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """
    Semantic cache using embeddings for similarity matching.

    Usage:
        cache = SemanticCache(embedding_func=get_embedding, similarity_threshold=0.92)

        # Store
        await cache.set("What is Dallas population?", response_data)

        # Retrieve (will match similar queries)
        result = await cache.get("Dallas population")  # Returns cached response
    """

    def __init__(
        self,
        embedding_func=None,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
        ttl_seconds: int = 3600,
        use_exact_cache: bool = True,
    ):
        """
        Initialize semantic cache.

        Args:
            embedding_func: Async function to generate embeddings (query: str) -> np.ndarray
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            max_entries: Maximum number of entries to store
            ttl_seconds: Time-to-live for cache entries
            use_exact_cache: Also use exact text matching (faster for identical queries)
        """
        self.embedding_func = embedding_func
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.use_exact_cache = use_exact_cache

        # Storage
        self._entries: List[CachedEntry] = []
        self._exact_cache: Dict[str, CachedEntry] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "stores": 0,
        }

        logger.info(
            f"SemanticCache initialized: threshold={similarity_threshold}, "
            f"max_entries={max_entries}, ttl={ttl_seconds}s"
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for exact matching."""
        return query.lower().strip()

    def _get_query_hash(self, query: str) -> str:
        """Get hash of normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _is_expired(self, entry: CachedEntry) -> bool:
        """Check if entry has expired."""
        return time.time() - entry.timestamp > self.ttl_seconds

    async def _get_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for query."""
        if not self.embedding_func:
            return None

        try:
            embedding = await self.embedding_func(query)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    async def get(self, query: str) -> Tuple[Optional[Any], float]:
        """
        Get cached response for query.

        Args:
            query: Query string to look up

        Returns:
            Tuple of (response, similarity_score) or (None, 0.0) if not found
        """
        async with self._lock:
            # Clean expired entries periodically
            await self._cleanup_expired()

            # Try exact match first (fastest)
            if self.use_exact_cache:
                query_hash = self._get_query_hash(query)
                if query_hash in self._exact_cache:
                    entry = self._exact_cache[query_hash]
                    if not self._is_expired(entry):
                        entry.hit_count += 1
                        self._stats["exact_hits"] += 1
                        logger.debug(f"Semantic cache exact hit: {query[:50]}...")
                        return entry.response, 1.0

            # Try semantic match if embedding function available
            if self.embedding_func and self._entries:
                query_embedding = await self._get_embedding(query)

                if query_embedding is not None:
                    best_match = None
                    best_score = 0.0

                    for entry in self._entries:
                        if self._is_expired(entry):
                            continue

                        if entry.embedding is None:
                            continue

                        score = self._cosine_similarity(query_embedding, entry.embedding)

                        if score > best_score and score >= self.similarity_threshold:
                            best_score = score
                            best_match = entry

                    if best_match:
                        best_match.hit_count += 1
                        self._stats["semantic_hits"] += 1
                        logger.debug(
                            f"Semantic cache hit (score={best_score:.3f}): "
                            f"{query[:30]}... → {best_match.query[:30]}..."
                        )
                        return best_match.response, best_score

            self._stats["misses"] += 1
            return None, 0.0

    async def set(
        self,
        query: str,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store query-response pair in cache.

        Args:
            query: Query string
            response: Response to cache
            metadata: Optional metadata to store with entry
        """
        async with self._lock:
            # Create entry
            entry = CachedEntry(
                query=query,
                response=response,
                timestamp=time.time(),
                metadata=metadata or {},
            )

            # Get embedding if available
            if self.embedding_func:
                entry.embedding = await self._get_embedding(query)

            # Store in exact cache
            if self.use_exact_cache:
                query_hash = self._get_query_hash(query)
                self._exact_cache[query_hash] = entry

            # Store in semantic cache
            self._entries.append(entry)
            self._stats["stores"] += 1

            # Evict if over capacity
            if len(self._entries) > self.max_entries:
                await self._evict()

            logger.debug(f"Semantic cache store: {query[:50]}...")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()

        # Clean exact cache
        expired_hashes = [
            h for h, e in self._exact_cache.items()
            if current_time - e.timestamp > self.ttl_seconds
        ]
        for h in expired_hashes:
            del self._exact_cache[h]

        # Clean semantic entries
        self._entries = [
            e for e in self._entries
            if current_time - e.timestamp <= self.ttl_seconds
        ]

    async def _evict(self) -> None:
        """Evict entries when over capacity (LRU-like based on hits and age)."""
        # Sort by score: lower = evict first
        # Score = hit_count - (age_seconds / 100)
        current_time = time.time()

        def eviction_score(entry: CachedEntry) -> float:
            age = current_time - entry.timestamp
            return entry.hit_count - (age / 100)

        self._entries.sort(key=eviction_score, reverse=True)
        self._entries = self._entries[:self.max_entries]

        # Rebuild exact cache
        self._exact_cache = {
            self._get_query_hash(e.query): e
            for e in self._entries
        }

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate a specific query from cache.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        async with self._lock:
            query_hash = self._get_query_hash(query)

            # Remove from exact cache
            removed = query_hash in self._exact_cache
            if removed:
                del self._exact_cache[query_hash]

            # Remove from semantic entries
            normalized = self._normalize_query(query)
            self._entries = [
                e for e in self._entries
                if self._normalize_query(e.query) != normalized
            ]

            return removed

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._entries.clear()
            self._exact_cache.clear()
            logger.info("Semantic cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self._stats["exact_hits"] +
            self._stats["semantic_hits"] +
            self._stats["misses"]
        )

        hit_rate = 0.0
        if total_requests > 0:
            total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
            hit_rate = total_hits / total_requests

        return {
            **self._stats,
            "total_entries": len(self._entries),
            "exact_cache_size": len(self._exact_cache),
            "hit_rate": hit_rate,
        }


# Lightweight embedding using simple text features
# This provides basic semantic matching without requiring an external embedding model
def _simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Generate a simple embedding based on text features.
    Not as powerful as transformer embeddings but works offline.
    """
    text = text.lower().strip()

    # Feature extraction
    features = []

    # Character n-grams (normalized)
    char_counts = {}
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        char_counts[trigram] = char_counts.get(trigram, 0) + 1

    # Word features
    words = text.split()
    word_set = set(words)

    # Geographic/location keywords
    geo_keywords = {
        'population', 'demographic', 'income', 'location', 'area',
        'city', 'state', 'county', 'zip', 'address', 'near', 'nearby',
        'tapestry', 'segment', 'marketing', 'analyze', 'show', 'zoom'
    }

    # Action keywords
    action_keywords = {
        'show', 'hide', 'zoom', 'pan', 'filter', 'remove', 'add',
        'get', 'find', 'search', 'analyze', 'compare', 'export'
    }

    # Build feature vector
    np.random.seed(hash(text) % (2**32))

    # Hash-based features from trigrams
    for trigram, count in char_counts.items():
        idx = hash(trigram) % dim
        features.append((idx, count))

    # Word overlap features
    geo_overlap = len(word_set & geo_keywords)
    action_overlap = len(word_set & action_keywords)

    # Create vector
    vec = np.zeros(dim)

    for idx, val in features:
        vec[idx] += val

    # Add semantic features
    vec[0] = geo_overlap
    vec[1] = action_overlap
    vec[2] = len(words)
    vec[3] = len(text)

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


async def simple_embedding_func(query: str) -> np.ndarray:
    """Async wrapper for simple embedding."""
    return _simple_embedding(query)


# Global semantic cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache(
    embedding_func=None,
    similarity_threshold: float = 0.92,
) -> SemanticCache:
    """Get or create global semantic cache instance."""
    global _semantic_cache

    if _semantic_cache is None:
        # Use simple embedding if no custom function provided
        func = embedding_func or simple_embedding_func

        _semantic_cache = SemanticCache(
            embedding_func=func,
            similarity_threshold=similarity_threshold,
            max_entries=500,
            ttl_seconds=1800,  # 30 minutes
        )

    return _semantic_cache


async def semantic_cache_get(query: str) -> Tuple[Optional[Any], float]:
    """Convenience function to get from semantic cache."""
    cache = get_semantic_cache()
    return await cache.get(query)


async def semantic_cache_set(
    query: str,
    response: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to store in semantic cache."""
    cache = get_semantic_cache()
    await cache.set(query, response, metadata)


async def semantic_cache_stats() -> Dict[str, Any]:
    """Get semantic cache statistics."""
    cache = get_semantic_cache()
    return cache.get_stats()
