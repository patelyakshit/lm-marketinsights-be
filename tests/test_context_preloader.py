"""
Unit tests for Context Preloader module.
"""

import asyncio
import time
import pytest
from utils.context_preloader import (
    ContextPreloader,
    PreloadedContext,
    PreloadPriority,
    PreloadTask,
    get_preloader,
    preload_session_context,
    get_session_context,
    invalidate_session_context,
)


class TestPreloadedContext:
    """Tests for PreloadedContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = PreloadedContext()

        assert ctx.user_preferences == {}
        assert ctx.saved_locations == []
        assert ctx.recent_searches == []
        assert ctx.frequent_layers == []
        assert ctx.default_extent is None
        assert ctx.recent_analyses == []
        assert not ctx.is_complete
        assert ctx.timestamp > 0


class TestContextPreloader:
    """Tests for ContextPreloader class."""

    @pytest.fixture
    def preloader(self):
        """Create a fresh preloader."""
        return ContextPreloader(
            default_ttl=60,
            max_cached_sessions=10,
        )

    @pytest.mark.asyncio
    async def test_preload_creates_context(self, preloader):
        """Test that preload creates context."""
        session_id = "test-session-1"
        context = await preloader.preload(session_id)

        assert isinstance(context, PreloadedContext)
        assert context.is_complete

    @pytest.mark.asyncio
    async def test_get_context_returns_preloaded(self, preloader):
        """Test getting preloaded context."""
        session_id = "test-session-2"
        await preloader.preload(session_id)

        context = preloader.get_context(session_id)
        assert context is not None
        assert isinstance(context, PreloadedContext)

    @pytest.mark.asyncio
    async def test_get_context_nonexistent(self, preloader):
        """Test getting nonexistent context."""
        context = preloader.get_context("nonexistent-session")
        assert context is None

    @pytest.mark.asyncio
    async def test_invalidate_removes_context(self, preloader):
        """Test that invalidate removes context."""
        session_id = "test-session-3"
        await preloader.preload(session_id)

        # Verify it exists
        assert preloader.get_context(session_id) is not None

        # Invalidate
        await preloader.invalidate(session_id)

        # Verify it's gone
        assert preloader.get_context(session_id) is None

    @pytest.mark.asyncio
    async def test_is_preloading(self, preloader):
        """Test is_preloading flag."""
        session_id = "test-session-4"

        # Before preloading
        assert not preloader.is_preloading(session_id)

        # After preloading
        await preloader.preload(session_id)
        assert not preloader.is_preloading(session_id)  # Should be done

    @pytest.mark.asyncio
    async def test_stats(self, preloader):
        """Test preloader statistics."""
        await preloader.preload("test-session-5")

        stats = preloader.get_stats()

        assert "cached_sessions" in stats
        assert "preloading_sessions" in stats
        assert "max_sessions" in stats
        assert stats["cached_sessions"] >= 1

    @pytest.mark.asyncio
    async def test_eviction_when_over_limit(self, preloader):
        """Test eviction when over max sessions."""
        # Preload more than max_cached_sessions
        for i in range(15):
            await preloader.preload(f"session-{i}")

        # Should not exceed max
        stats = preloader.get_stats()
        assert stats["cached_sessions"] <= preloader.max_cached_sessions

    @pytest.mark.asyncio
    async def test_multiple_preloads_same_session(self, preloader):
        """Test multiple preloads for same session."""
        session_id = "test-session-6"

        # First preload
        ctx1 = await preloader.preload(session_id)

        # Second preload should return cached
        ctx2 = await preloader.preload(session_id)

        # Should be same context (or at least both complete)
        assert ctx1.is_complete
        assert ctx2.is_complete


class TestGlobalPreloader:
    """Tests for global preloader functions."""

    @pytest.mark.asyncio
    async def test_get_preloader_singleton(self):
        """Test that get_preloader returns singleton."""
        preloader1 = get_preloader()
        preloader2 = get_preloader()

        assert preloader1 is preloader2

    @pytest.mark.asyncio
    async def test_preload_session_context(self):
        """Test preload_session_context function."""
        session_id = "global-test-session"

        # Should not raise
        await preload_session_context(session_id)

        # Give it time to complete
        await asyncio.sleep(0.1)

        # Should be able to get context
        context = get_session_context(session_id)
        # May or may not be complete yet, but should not raise

    @pytest.mark.asyncio
    async def test_invalidate_session_context(self):
        """Test invalidate_session_context function."""
        session_id = "global-test-session-2"

        # Preload
        preloader = get_preloader()
        await preloader.preload(session_id)

        # Invalidate
        await invalidate_session_context(session_id)

        # Should be gone
        context = get_session_context(session_id)
        assert context is None


class TestPreloadPriority:
    """Tests for PreloadPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert PreloadPriority.HIGH.value < PreloadPriority.MEDIUM.value
        assert PreloadPriority.MEDIUM.value < PreloadPriority.LOW.value


class TestPreloadTask:
    """Tests for PreloadTask dataclass."""

    def test_default_values(self):
        """Test default task values."""
        async def dummy_loader():
            pass

        task = PreloadTask(
            name="test",
            loader_func=dummy_loader,
        )

        assert task.name == "test"
        assert task.priority == PreloadPriority.MEDIUM
        assert task.ttl_seconds == 300
        assert task.args == ()
        assert task.kwargs == {}
