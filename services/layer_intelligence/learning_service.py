"""
Learning Service for Layer Intelligence System.

Handles:
- Query pattern storage and retrieval
- User feedback processing
- Insight generation and storage
- Query improvement suggestions based on past patterns
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

from .models import AgentResponse, QueryInsight, StructuredQuery
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """A stored query pattern for learning."""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    intent: str = ""
    layers_used: list[str] = field(default_factory=list)
    filters_used: list[dict] = field(default_factory=list)
    structured_queries: list[dict] = field(default_factory=list)
    was_successful: bool = True
    user_feedback: Optional[str] = None  # positive, negative, neutral
    feedback_comment: Optional[str] = None
    execution_time_ms: int = 0
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_embedding_text(self) -> str:
        """Generate text for embedding this pattern."""
        parts = [self.query]
        if self.intent:
            parts.append(f"Intent: {self.intent}")
        if self.layers_used:
            parts.append(f"Layers: {', '.join(self.layers_used)}")
        return " | ".join(parts)


@dataclass
class LearningInsight:
    """An insight learned from query patterns."""
    id: str = field(default_factory=lambda: str(uuid4()))
    insight_type: str = ""  # pattern, optimization, suggestion
    title: str = ""
    content: str = ""
    related_layers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class LearningService:
    """
    Service for learning from query patterns and improving responses.

    Features:
    - Store successful query patterns
    - Find similar past queries for better responses
    - Track user feedback
    - Generate query suggestions
    - Improve layer recommendations over time
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        db=None,
        qdrant_client=None,
    ):
        self.embeddings = embedding_service or get_embedding_service()
        self.db = db
        self.qdrant = qdrant_client

        # In-memory cache for frequently used patterns
        self._pattern_cache: dict[str, QueryPattern] = {}
        self._insight_cache: dict[str, LearningInsight] = {}

        # Collection name for query pattern embeddings
        self.collection_name = "query_patterns"

    async def initialize(self):
        """Initialize the learning service (create collections, etc.)."""
        if self.qdrant:
            try:
                from qdrant_client.models import Distance, VectorParams

                # Create collection if not exists
                collections = await self.qdrant.get_collections()
                collection_names = [c.name for c in collections.collections]

                if self.collection_name not in collection_names:
                    await self.qdrant.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embeddings.dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"Created Qdrant collection: {self.collection_name}")

            except Exception as e:
                logger.warning(f"Could not initialize Qdrant for learning: {e}")

    # =========================================================================
    # Query Pattern Storage
    # =========================================================================

    async def store_query_pattern(
        self,
        query: str,
        response: AgentResponse,
        structured_queries: list[StructuredQuery],
        intent: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> QueryPattern:
        """
        Store a query pattern from a successful execution.

        Args:
            query: Original natural language query
            response: Agent response
            structured_queries: Queries that were executed
            intent: Classified intent
            user_id: Optional user ID
            session_id: Optional session ID

        Returns:
            Stored QueryPattern
        """
        pattern = QueryPattern(
            query=query,
            intent=intent,
            layers_used=response.layers_used,
            structured_queries=[q.to_arcgis_params() for q in structured_queries],
            filters_used=[
                {"field": f.field, "operator": f.operator.value, "value": f.value}
                for q in structured_queries
                for f in q.filters
            ],
            was_successful=response.confidence > 0.5,
            execution_time_ms=response.execution_time_ms,
            confidence_score=response.confidence,
        )

        # Store in database if available
        if self.db:
            try:
                pattern_id = await self.db.store_query_pattern(
                    query=pattern.query,
                    intent=pattern.intent,
                    layers_used=pattern.layers_used,
                    structured_queries=pattern.structured_queries,
                    filters_used=pattern.filters_used,
                    was_successful=pattern.was_successful,
                    execution_time_ms=pattern.execution_time_ms,
                    confidence_score=pattern.confidence_score,
                    user_id=user_id,
                    session_id=session_id,
                )
                pattern.id = pattern_id
            except Exception as e:
                logger.warning(f"Could not store pattern in DB: {e}")

        # Store embedding in Qdrant for similarity search
        if self.qdrant:
            try:
                embedding = await self.embeddings.embed_single(pattern.to_embedding_text())

                from qdrant_client.models import PointStruct

                await self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=pattern.id,
                            vector=embedding,
                            payload={
                                "query": pattern.query,
                                "intent": pattern.intent,
                                "layers_used": pattern.layers_used,
                                "was_successful": pattern.was_successful,
                                "confidence": pattern.confidence_score,
                            },
                        )
                    ],
                )
            except Exception as e:
                logger.warning(f"Could not store pattern embedding: {e}")

        # Cache in memory
        self._pattern_cache[pattern.id] = pattern

        logger.info(f"Stored query pattern: {pattern.id}")
        return pattern

    async def find_similar_patterns(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.6,
    ) -> list[QueryPattern]:
        """
        Find similar successful query patterns.

        Args:
            query: Query to find similar patterns for
            limit: Maximum patterns to return
            min_confidence: Minimum confidence score

        Returns:
            List of similar QueryPatterns
        """
        if not self.qdrant:
            return []

        try:
            # Generate embedding for the query
            query_embedding = await self.embeddings.embed_query(query)

            # Search in Qdrant
            results = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter={
                    "must": [
                        {"key": "was_successful", "match": {"value": True}},
                    ]
                },
            )

            patterns = []
            for result in results:
                if result.score >= min_confidence:
                    pattern = QueryPattern(
                        id=result.id,
                        query=result.payload.get("query", ""),
                        intent=result.payload.get("intent", ""),
                        layers_used=result.payload.get("layers_used", []),
                        confidence_score=result.payload.get("confidence", 0),
                    )
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.warning(f"Could not search similar patterns: {e}")
            return []

    # =========================================================================
    # Feedback Processing
    # =========================================================================

    async def record_feedback(
        self,
        pattern_id: str,
        feedback: str,
        comment: Optional[str] = None,
    ):
        """
        Record user feedback for a query pattern.

        Args:
            pattern_id: ID of the pattern
            feedback: "positive", "negative", or "neutral"
            comment: Optional feedback comment
        """
        # Update in database
        if self.db:
            try:
                await self.db.update_query_feedback(pattern_id, feedback, comment)
            except Exception as e:
                logger.warning(f"Could not update feedback in DB: {e}")

        # Update in cache
        if pattern_id in self._pattern_cache:
            self._pattern_cache[pattern_id].user_feedback = feedback
            self._pattern_cache[pattern_id].feedback_comment = comment

        # Update Qdrant payload
        if self.qdrant:
            try:
                await self.qdrant.set_payload(
                    collection_name=self.collection_name,
                    payload={"user_feedback": feedback},
                    points=[pattern_id],
                )
            except Exception as e:
                logger.warning(f"Could not update feedback in Qdrant: {e}")

        logger.info(f"Recorded {feedback} feedback for pattern {pattern_id}")

    # =========================================================================
    # Query Improvement
    # =========================================================================

    async def suggest_query_improvements(
        self,
        query: str,
        current_layers: list[str],
    ) -> dict:
        """
        Suggest improvements based on past successful patterns.

        Args:
            query: Current query
            current_layers: Currently selected layers

        Returns:
            Dict with suggestions
        """
        similar_patterns = await self.find_similar_patterns(query, limit=3)

        suggestions = {
            "additional_layers": [],
            "filter_suggestions": [],
            "query_reformulation": None,
        }

        if not similar_patterns:
            return suggestions

        # Find layers used in successful similar queries but not in current
        for pattern in similar_patterns:
            for layer in pattern.layers_used:
                if layer not in current_layers and layer not in suggestions["additional_layers"]:
                    suggestions["additional_layers"].append(layer)

            # Collect filter patterns
            for filter_info in pattern.filters_used:
                suggestions["filter_suggestions"].append(filter_info)

        # If the best pattern has much higher confidence, suggest reformulation
        best_pattern = similar_patterns[0]
        if best_pattern.confidence_score > 0.8:
            suggestions["query_reformulation"] = best_pattern.query

        return suggestions

    async def get_recommended_layers(
        self,
        query: str,
        limit: int = 3,
    ) -> list[str]:
        """
        Get recommended layers based on past successful queries.

        Args:
            query: Query to get recommendations for
            limit: Maximum layers to recommend

        Returns:
            List of recommended layer names
        """
        similar_patterns = await self.find_similar_patterns(query, limit=5)

        # Count layer occurrences in successful patterns
        layer_scores: dict[str, float] = {}
        for pattern in similar_patterns:
            for layer in pattern.layers_used:
                if layer not in layer_scores:
                    layer_scores[layer] = 0
                layer_scores[layer] += pattern.confidence_score

        # Sort by score
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)

        return [layer for layer, score in sorted_layers[:limit]]

    # =========================================================================
    # Insight Generation
    # =========================================================================

    async def generate_insight(
        self,
        query: str,
        response: AgentResponse,
        data_summary: dict,
    ) -> Optional[LearningInsight]:
        """
        Generate an insight from a successful query execution.

        Args:
            query: Original query
            response: Agent response
            data_summary: Summary of returned data

        Returns:
            Generated insight or None
        """
        if response.confidence < 0.7:
            return None

        insight = LearningInsight(
            insight_type="pattern",
            title=f"Query pattern: {query[:50]}...",
            content=response.answer[:500],
            related_layers=response.layers_used,
            confidence=response.confidence,
        )

        # Store in database
        if self.db:
            try:
                insight_id = await self.db.store_insight(
                    query_pattern_id=None,
                    layers_involved=insight.related_layers,
                    insight_type=insight.insight_type,
                    title=insight.title,
                    content=insight.content,
                    confidence_score=insight.confidence,
                )
                insight.id = insight_id
            except Exception as e:
                logger.warning(f"Could not store insight: {e}")

        self._insight_cache[insight.id] = insight
        return insight

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_popular_layers(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently used layers."""
        layer_counts: dict[str, int] = {}

        for pattern in self._pattern_cache.values():
            if pattern.was_successful:
                for layer in pattern.layers_used:
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

        sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_layers[:limit]

    async def get_query_success_rate(self) -> float:
        """Get overall query success rate."""
        if not self._pattern_cache:
            return 0.0

        successful = sum(1 for p in self._pattern_cache.values() if p.was_successful)
        return successful / len(self._pattern_cache)

    def get_stats(self) -> dict:
        """Get learning service statistics."""
        return {
            "patterns_cached": len(self._pattern_cache),
            "insights_cached": len(self._insight_cache),
            "collection_name": self.collection_name,
        }


# =============================================================================
# Factory Functions
# =============================================================================

_learning_service: Optional[LearningService] = None


def get_learning_service(
    force_new: bool = False,
) -> LearningService:
    """Get or create the global learning service."""
    global _learning_service

    if _learning_service is None or force_new:
        _learning_service = LearningService()

    return _learning_service


def set_learning_service(service: LearningService):
    """Set the global learning service."""
    global _learning_service
    _learning_service = service


async def initialize_learning_service(
    embedding_service=None,
    db=None,
    qdrant_client=None,
) -> LearningService:
    """Initialize the learning service with dependencies."""
    global _learning_service

    _learning_service = LearningService(
        embedding_service=embedding_service,
        db=db,
        qdrant_client=qdrant_client,
    )
    await _learning_service.initialize()

    return _learning_service
