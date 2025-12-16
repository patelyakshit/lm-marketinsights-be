"""
Context Preloading System

Preloads common context when a session starts to reduce first-query latency.
Loads in background so it doesn't block the connection.

Preloaded data includes:
- User's saved locations and recent searches
- Frequently used layers and preferences
- Recent analysis results
- Default map extent based on user history
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class PreloadPriority(Enum):
    """Priority levels for preloading."""
    HIGH = 1      # Critical for first query (user preferences)
    MEDIUM = 2    # Helpful but not blocking (recent searches)
    LOW = 3       # Nice to have (cached analyses)


@dataclass
class PreloadTask:
    """A preloading task definition."""
    name: str
    loader_func: callable
    priority: PreloadPriority = PreloadPriority.MEDIUM
    ttl_seconds: int = 300  # How long to keep preloaded data
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


@dataclass
class PreloadedContext:
    """Container for preloaded context data."""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    saved_locations: List[Dict[str, Any]] = field(default_factory=list)
    recent_searches: List[str] = field(default_factory=list)
    frequent_layers: List[str] = field(default_factory=list)
    default_extent: Optional[Dict[str, float]] = None
    recent_analyses: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    is_complete: bool = False


class ContextPreloader:
    """
    Preloads user context in the background when session starts.

    Usage:
        preloader = ContextPreloader()

        # Start preloading when session connects
        asyncio.create_task(preloader.preload(session_id, user_id))

        # Get preloaded context (non-blocking, returns what's available)
        context = preloader.get_context(session_id)
    """

    def __init__(
        self,
        default_ttl: int = 600,  # 10 minutes
        max_cached_sessions: int = 100,
    ):
        """
        Initialize context preloader.

        Args:
            default_ttl: Default TTL for preloaded data in seconds
            max_cached_sessions: Maximum number of sessions to cache
        """
        self.default_ttl = default_ttl
        self.max_cached_sessions = max_cached_sessions

        # Storage: session_id -> PreloadedContext
        self._contexts: Dict[str, PreloadedContext] = {}
        self._preloading: Set[str] = set()  # Sessions currently being preloaded
        self._lock = asyncio.Lock()

        logger.info(
            f"ContextPreloader initialized: ttl={default_ttl}s, "
            f"max_sessions={max_cached_sessions}"
        )

    async def preload(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> PreloadedContext:
        """
        Preload context for a session (runs in background).

        Args:
            session_id: Session identifier
            user_id: Optional user ID for personalization
            organization_id: Optional org ID for org-specific data

        Returns:
            PreloadedContext with loaded data
        """
        async with self._lock:
            # Check if already preloading
            if session_id in self._preloading:
                logger.debug(f"Preload already in progress for {session_id}")
                return self._contexts.get(session_id, PreloadedContext())

            # Check if already preloaded and valid
            if session_id in self._contexts:
                ctx = self._contexts[session_id]
                if time.time() - ctx.timestamp < self.default_ttl:
                    logger.debug(f"Using cached preloaded context for {session_id}")
                    return ctx

            self._preloading.add(session_id)

        try:
            context = PreloadedContext()

            # Run preloading tasks in priority order
            await self._preload_high_priority(context, user_id, organization_id)
            await self._preload_medium_priority(context, user_id, session_id)
            await self._preload_low_priority(context, user_id)

            context.is_complete = True

            # Store context
            async with self._lock:
                self._contexts[session_id] = context
                self._preloading.discard(session_id)

                # Cleanup old contexts if over limit
                await self._cleanup_old_contexts()

            logger.info(
                f"Preloaded context for session {session_id}: "
                f"{len(context.saved_locations)} locations, "
                f"{len(context.recent_searches)} searches, "
                f"{len(context.frequent_layers)} layers"
            )

            return context

        except Exception as e:
            logger.error(f"Error preloading context for {session_id}: {e}")
            async with self._lock:
                self._preloading.discard(session_id)
            return PreloadedContext()

    async def _preload_high_priority(
        self,
        context: PreloadedContext,
        user_id: Optional[str],
        organization_id: Optional[str],
    ) -> None:
        """Preload high priority data (user preferences, default location)."""
        try:
            # Load user preferences (if user_id available)
            if user_id:
                context.user_preferences = await self._load_user_preferences(user_id)

            # Load organization default extent (if available)
            if organization_id:
                context.default_extent = await self._load_default_extent(organization_id)

        except Exception as e:
            logger.warning(f"Error in high priority preload: {e}")

    async def _preload_medium_priority(
        self,
        context: PreloadedContext,
        user_id: Optional[str],
        session_id: str,
    ) -> None:
        """Preload medium priority data (saved locations, recent searches)."""
        try:
            tasks = []

            if user_id:
                tasks.append(self._load_saved_locations(user_id))
                tasks.append(self._load_recent_searches(user_id))
                tasks.append(self._load_frequent_layers(user_id))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Medium priority preload task {i} failed: {result}")
                        continue

                    if i == 0:  # saved_locations
                        context.saved_locations = result or []
                    elif i == 1:  # recent_searches
                        context.recent_searches = result or []
                    elif i == 2:  # frequent_layers
                        context.frequent_layers = result or []

        except Exception as e:
            logger.warning(f"Error in medium priority preload: {e}")

    async def _preload_low_priority(
        self,
        context: PreloadedContext,
        user_id: Optional[str],
    ) -> None:
        """Preload low priority data (recent analyses)."""
        try:
            if user_id:
                context.recent_analyses = await self._load_recent_analyses(user_id)
        except Exception as e:
            logger.warning(f"Error in low priority preload: {e}")

    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from database."""
        # TODO: Integrate with actual user preferences storage
        # For now, return sensible defaults
        return {
            "default_map_style": "streets",
            "preferred_units": "imperial",
            "show_labels": True,
            "auto_zoom": True,
        }

    async def _load_default_extent(self, organization_id: str) -> Optional[Dict[str, float]]:
        """Load organization's default map extent."""
        # TODO: Integrate with organization settings
        # Return None to use system default
        return None

    async def _load_saved_locations(self, user_id: str) -> List[Dict[str, Any]]:
        """Load user's saved/favorited locations."""
        # TODO: Integrate with saved locations storage
        return []

    async def _load_recent_searches(self, user_id: str) -> List[str]:
        """Load user's recent search queries."""
        # TODO: Integrate with search history
        return []

    async def _load_frequent_layers(self, user_id: str) -> List[str]:
        """Load user's frequently used layers."""
        # TODO: Integrate with layer usage tracking
        return []

    async def _load_recent_analyses(self, user_id: str) -> List[Dict[str, Any]]:
        """Load user's recent analysis results (for smart suggestions)."""
        # TODO: Integrate with analysis history
        return []

    async def _cleanup_old_contexts(self) -> None:
        """Remove old contexts if over capacity."""
        if len(self._contexts) <= self.max_cached_sessions:
            return

        # Sort by timestamp (oldest first)
        sorted_sessions = sorted(
            self._contexts.items(),
            key=lambda x: x[1].timestamp
        )

        # Remove oldest contexts until under limit
        to_remove = len(self._contexts) - self.max_cached_sessions
        for session_id, _ in sorted_sessions[:to_remove]:
            del self._contexts[session_id]
            logger.debug(f"Evicted preloaded context for {session_id}")

    def get_context(self, session_id: str) -> Optional[PreloadedContext]:
        """
        Get preloaded context for session (non-blocking).

        Args:
            session_id: Session identifier

        Returns:
            PreloadedContext if available, None otherwise
        """
        ctx = self._contexts.get(session_id)

        if ctx is None:
            return None

        # Check if expired
        if time.time() - ctx.timestamp > self.default_ttl:
            return None

        return ctx

    def is_preloading(self, session_id: str) -> bool:
        """Check if preloading is in progress for session."""
        return session_id in self._preloading

    async def invalidate(self, session_id: str) -> None:
        """Invalidate preloaded context for session."""
        async with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]
                logger.debug(f"Invalidated preloaded context for {session_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get preloader statistics."""
        return {
            "cached_sessions": len(self._contexts),
            "preloading_sessions": len(self._preloading),
            "max_sessions": self.max_cached_sessions,
            "default_ttl": self.default_ttl,
        }


# Global preloader instance
_preloader: Optional[ContextPreloader] = None


def get_preloader() -> ContextPreloader:
    """Get or create global ContextPreloader instance."""
    global _preloader
    if _preloader is None:
        _preloader = ContextPreloader()
    return _preloader


async def preload_session_context(
    session_id: str,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> None:
    """
    Start context preloading for a session (non-blocking).

    Call this when a WebSocket connection is established.

    Args:
        session_id: Session identifier
        user_id: Optional user ID
        organization_id: Optional organization ID
    """
    preloader = get_preloader()
    # Run in background (don't await)
    asyncio.create_task(preloader.preload(session_id, user_id, organization_id))


def get_session_context(session_id: str) -> Optional[PreloadedContext]:
    """
    Get preloaded context for session.

    Args:
        session_id: Session identifier

    Returns:
        PreloadedContext if available
    """
    preloader = get_preloader()
    return preloader.get_context(session_id)


async def invalidate_session_context(session_id: str) -> None:
    """Invalidate preloaded context when session ends."""
    preloader = get_preloader()
    await preloader.invalidate(session_id)
