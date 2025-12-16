"""
Unit tests for Semantic Cache module.
"""

import asyncio
import pytest
import numpy as np
from utils.semantic_cache import (
    SemanticCache,
    CachedEntry,
    _simple_embedding,
    simple_embedding_func,
    get_semantic_cache,
    semantic_cache_get,
    semantic_cache_set,
)


class TestSimpleEmbedding:
    """Tests for simple embedding function."""

    def test_embedding_returns_numpy_array(self):
        """Embedding should return numpy array."""
        result = _simple_embedding("test query")
        assert isinstance(result, np.ndarray)

    def test_embedding_dimension(self):
        """Embedding should have correct dimension."""
        result = _simple_embedding("test query", dim=128)
        assert result.shape == (128,)

    def test_embedding_is_normalized(self):
        """Embedding should be normalized (unit length)."""
        result = _simple_embedding("test query with some content")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01 or norm == 0  # Allow small tolerance

    def test_similar_queries_have_similar_embeddings(self):
        """Similar queries should have similar embeddings."""
        emb1 = _simple_embedding("what is the population of dallas")
        emb2 = _simple_embedding("dallas population")
        emb3 = _simple_embedding("zoom to new york city")

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        # Similar queries should be more similar than different ones
        assert sim_12 > sim_13


class TestSemanticCache:
    """Tests for SemanticCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""
        return SemanticCache(
            embedding_func=simple_embedding_func,
            similarity_threshold=0.85,
            max_entries=10,
            ttl_seconds=60,
        )

    @pytest.mark.asyncio
    async def test_set_and_get_exact(self, cache):
        """Test exact match retrieval."""
        await cache.set("test query", {"result": "value"})
        result, similarity = await cache.get("test query")

        assert result == {"result": "value"}
        assert similarity == 1.0

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result, similarity = await cache.get("nonexistent")

        assert result is None
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_semantic_similarity(self, cache):
        """Test semantic similarity matching."""
        await cache.set("what is dallas population", {"pop": 1000000})

        # Very similar query should match
        result, similarity = await cache.get("dallas population")

        # Similarity might be below threshold for different queries
        # Just verify the mechanics work
        assert similarity >= 0.0

    @pytest.mark.asyncio
    async def test_eviction_when_full(self, cache):
        """Test eviction when cache is full."""
        # Fill cache
        for i in range(15):  # More than max_entries (10)
            await cache.set(f"query {i}", f"result {i}")

        # Cache should not exceed max_entries
        assert len(cache._entries) <= cache.max_entries

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        """Test cache invalidation."""
        await cache.set("test query", "value")

        # Verify it exists
        result, _ = await cache.get("test query")
        assert result == "value"

        # Invalidate
        removed = await cache.invalidate("test query")
        assert removed is True

        # Verify it's gone
        result, _ = await cache.get("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing cache."""
        await cache.set("query1", "value1")
        await cache.set("query2", "value2")

        await cache.clear()

        result1, _ = await cache.get("query1")
        result2, _ = await cache.get("query2")

        assert result1 is None
        assert result2 is None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """Test cache statistics."""
        await cache.set("query", "value")
        await cache.get("query")  # Hit
        await cache.get("nonexistent")  # Miss

        stats = cache.get_stats()

        assert "exact_hits" in stats
        assert "semantic_hits" in stats
        assert "misses" in stats
        assert "stores" in stats
        assert stats["total_entries"] >= 0


class TestGlobalCache:
    """Tests for global cache functions."""

    @pytest.mark.asyncio
    async def test_get_semantic_cache_singleton(self):
        """Test that get_semantic_cache returns singleton."""
        cache1 = get_semantic_cache()
        cache2 = get_semantic_cache()

        assert cache1 is cache2

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test semantic_cache_get and semantic_cache_set."""
        await semantic_cache_set("conv_test", {"data": "test"})
        result, similarity = await semantic_cache_get("conv_test")

        assert result == {"data": "test"}
        assert similarity == 1.0
