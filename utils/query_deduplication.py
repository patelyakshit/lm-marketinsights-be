"""
Query Deduplication System

Prevents duplicate API calls when users double-click or submit the same query
multiple times within a short window. This saves costs and improves response time.

How it works:
1. When a query comes in, we create a hash of (session_id + normalized_query)
2. If the same hash is already being processed (in-flight), we wait for that result
3. If it's a new query, we register it and process normally
4. Results are shared with all waiters for the same query
5. Queries expire after 30 seconds to prevent stale blocking
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class InFlightQuery:
    """Represents a query currently being processed."""

    query_hash: str
    content: str
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    result_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    # TTL for in-flight queries (prevent stale blocking)
    TTL_SECONDS: int = 30

    def is_stale(self) -> bool:
        """Check if this query has been in-flight too long."""
        return datetime.utcnow() - self.created_at > timedelta(seconds=self.TTL_SECONDS)

    def set_result(self, result: Any) -> None:
        """Set the result, notifying all waiters."""
        if not self.result_future.done():
            self.result_future.set_result(result)

    def set_error(self, error: Exception) -> None:
        """Set an error, notifying all waiters."""
        if not self.result_future.done():
            self.result_future.set_exception(error)

    async def wait_for_result(self, timeout: float = 30.0) -> Any:
        """Wait for the result with timeout."""
        try:
            return await asyncio.wait_for(self.result_future, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Query {self.query_hash} timed out after {timeout}s")


class QueryDeduplicator:
    """
    Singleton class to manage query deduplication across all sessions.

    Usage:
        deduplicator = QueryDeduplicator()

        # Check if query is duplicate
        is_duplicate, in_flight = await deduplicator.check_or_register(session_id, query)

        if is_duplicate:
            # Wait for existing query's result
            result = await in_flight.wait_for_result()
        else:
            # Process query normally
            result = await process_query(query)
            # Mark as complete
            deduplicator.complete(in_flight, result)
    """

    _instance: Optional['QueryDeduplicator'] = None
    _lock: asyncio.Lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._in_flight = {}
            cls._instance._lock = asyncio.Lock()
            cls._instance._cleanup_task = None
        return cls._instance

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for comparison."""
        # Lowercase, strip whitespace, collapse multiple spaces
        normalized = query.lower().strip()
        normalized = ' '.join(normalized.split())
        return normalized

    @staticmethod
    def _make_hash(session_id: str, query: str) -> str:
        """Create a hash for the query."""
        normalized = QueryDeduplicator._normalize_query(query)
        content = f"{session_id}:{normalized}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def check_or_register(
        self,
        session_id: str,
        query: str
    ) -> tuple[bool, InFlightQuery]:
        """
        Check if query is duplicate or register as new.

        Returns:
            tuple[bool, InFlightQuery]: (is_duplicate, in_flight_query)
            - If is_duplicate=True, wait on in_flight_query.wait_for_result()
            - If is_duplicate=False, process query and call complete()
        """
        query_hash = self._make_hash(session_id, query)

        async with self._lock:
            # Clean up stale queries
            await self._cleanup_stale()

            # Check if same query is already in-flight
            if query_hash in self._in_flight:
                existing = self._in_flight[query_hash]
                if not existing.is_stale():
                    logger.info(
                        f"Duplicate query detected: {query_hash} "
                        f"(session: {session_id}, query: {query[:50]}...)"
                    )
                    return True, existing
                else:
                    # Remove stale query
                    del self._in_flight[query_hash]

            # Register new query
            in_flight = InFlightQuery(
                query_hash=query_hash,
                content=query,
                session_id=session_id
            )
            self._in_flight[query_hash] = in_flight

            logger.debug(
                f"Registered new query: {query_hash} "
                f"(session: {session_id}, query: {query[:50]}...)"
            )

            return False, in_flight

    def complete(self, in_flight: InFlightQuery, result: Any) -> None:
        """Mark a query as complete with its result."""
        in_flight.set_result(result)

        # Remove from in-flight after a short delay (allow late arrivals to get result)
        asyncio.create_task(self._delayed_cleanup(in_flight.query_hash, delay=2.0))

        logger.debug(f"Query completed: {in_flight.query_hash}")

    def fail(self, in_flight: InFlightQuery, error: Exception) -> None:
        """Mark a query as failed with an error."""
        in_flight.set_error(error)

        # Remove from in-flight immediately on failure
        if in_flight.query_hash in self._in_flight:
            del self._in_flight[in_flight.query_hash]

        logger.debug(f"Query failed: {in_flight.query_hash} - {error}")

    async def _delayed_cleanup(self, query_hash: str, delay: float) -> None:
        """Remove query from in-flight after delay."""
        await asyncio.sleep(delay)
        async with self._lock:
            if query_hash in self._in_flight:
                del self._in_flight[query_hash]

    async def _cleanup_stale(self) -> None:
        """Remove all stale queries."""
        stale_hashes = [
            h for h, q in self._in_flight.items()
            if q.is_stale()
        ]
        for h in stale_hashes:
            del self._in_flight[h]
            logger.debug(f"Cleaned up stale query: {h}")

    def get_stats(self) -> dict:
        """Get current deduplication statistics."""
        return {
            "in_flight_count": len(self._in_flight),
            "queries": [
                {
                    "hash": q.query_hash,
                    "session": q.session_id,
                    "age_seconds": (datetime.utcnow() - q.created_at).total_seconds(),
                    "is_stale": q.is_stale()
                }
                for q in self._in_flight.values()
            ]
        }


# Global instance for easy access
query_deduplicator = QueryDeduplicator()
