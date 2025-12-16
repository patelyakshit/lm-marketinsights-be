"""
Layer Intelligence System - Dynamic layer discovery and querying for AI agents.

This module provides:
- LayerCatalogService: Auto-discovers and indexes ArcGIS layers
- KnowledgeGraphService: Manages layer relationships for multi-hop reasoning
- SelfQueryRetriever: Converts natural language to structured queries
- AgenticOrchestrator: Autonomous agent for complex geospatial queries
- LearningService: Query pattern learning and feedback processing
- LayerIntelligenceDB: PostgreSQL persistence layer
"""

import logging
from typing import Optional

from .config import LayerIntelligenceConfig, get_config
from .models import (
    LayerType,
    GeometryType,
    FieldMetadata,
    LayerMetadata,
    LayerSearchResult,
    StructuredQuery,
    QueryPlan,
    QueryFilter,
    FilterOperator,
    QueryOperation,
    AgentAction,
    AgentStep,
    AgentResponse,
    QueryInsight,
    RelationshipType,
    GraphNode,
    GraphRelationship,
    ReasoningPath,
)
from .embedding_service import EmbeddingService, get_embedding_service
from .layer_catalog import LayerCatalogService, get_layer_catalog_service
from .knowledge_graph import KnowledgeGraphService, get_knowledge_graph_service
from .self_query import SelfQueryRetriever, QueryExecutor, get_self_query_retriever
from .orchestrator import AgenticOrchestrator, get_orchestrator
from .learning_service import LearningService, get_learning_service
from .database import LayerIntelligenceDB, get_layer_intelligence_db

logger = logging.getLogger(__name__)

__all__ = [
    # Config
    "LayerIntelligenceConfig",
    "get_config",
    # Models
    "LayerType",
    "GeometryType",
    "FieldMetadata",
    "LayerMetadata",
    "LayerSearchResult",
    "StructuredQuery",
    "QueryPlan",
    "QueryFilter",
    "FilterOperator",
    "QueryOperation",
    "AgentAction",
    "AgentStep",
    "AgentResponse",
    "QueryInsight",
    "RelationshipType",
    "GraphNode",
    "GraphRelationship",
    "ReasoningPath",
    # Services
    "EmbeddingService",
    "LayerCatalogService",
    "KnowledgeGraphService",
    "SelfQueryRetriever",
    "QueryExecutor",
    "AgenticOrchestrator",
    "LearningService",
    "LayerIntelligenceDB",
    # Factory functions
    "get_embedding_service",
    "get_layer_catalog_service",
    "get_knowledge_graph_service",
    "get_self_query_retriever",
    "get_orchestrator",
    "get_learning_service",
    "get_layer_intelligence_db",
    # Initialization
    "initialize_layer_intelligence",
]


# Global initialization state
_initialized = False


async def initialize_layer_intelligence(
    config: Optional[LayerIntelligenceConfig] = None,
    skip_db: bool = False,
    skip_qdrant: bool = False,
) -> dict:
    """
    Initialize the Layer Intelligence system.

    This should be called once at application startup to set up:
    - Embedding service
    - Qdrant vector database connection
    - Knowledge graph
    - Database tables
    - Learning service

    Args:
        config: Optional configuration (uses env vars if not provided)
        skip_db: Skip PostgreSQL initialization (for testing)
        skip_qdrant: Skip Qdrant initialization (for testing)

    Returns:
        Dict with initialization status for each component
    """
    global _initialized

    if _initialized:
        logger.info("Layer Intelligence already initialized")
        return {"status": "already_initialized"}

    logger.info("Initializing Layer Intelligence System...")
    status = {}

    try:
        # Load configuration
        if config is None:
            config = get_config()

        # 1. Initialize embedding service
        try:
            embedding_service = get_embedding_service()
            status["embedding_service"] = {
                "status": "ok",
                "provider": config.embedding.provider,
                "dimension": embedding_service.dimension,
            }
            logger.info(f"Embedding service initialized: {config.embedding.provider}")
        except Exception as e:
            status["embedding_service"] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to initialize embedding service: {e}")

        # 2. Initialize Qdrant (for vector search)
        qdrant_client = None
        if not skip_qdrant:
            try:
                from qdrant_client import QdrantClient

                qdrant_client = QdrantClient(
                    host=config.qdrant.host,
                    port=config.qdrant.port,
                    api_key=config.qdrant.api_key,
                    https=config.qdrant.https,
                )

                # Test connection
                qdrant_client.get_collections()

                status["qdrant"] = {
                    "status": "ok",
                    "host": config.qdrant.host,
                    "port": config.qdrant.port,
                }
                logger.info(f"Qdrant connected: {config.qdrant.host}:{config.qdrant.port}")
            except Exception as e:
                status["qdrant"] = {"status": "error", "error": str(e)}
                logger.warning(f"Qdrant not available: {e}")

        # 3. Initialize Layer Catalog Service
        try:
            from .layer_catalog import get_layer_catalog_service_sync
            catalog_service = get_layer_catalog_service_sync()
            if qdrant_client:
                catalog_service._qdrant = qdrant_client
            await catalog_service.initialize()
            status["layer_catalog"] = {"status": "ok"}
            logger.info("Layer Catalog Service initialized")
        except Exception as e:
            status["layer_catalog"] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to initialize Layer Catalog: {e}")

        # 4. Initialize Knowledge Graph
        try:
            from .knowledge_graph import get_knowledge_graph_service_sync
            graph_service = get_knowledge_graph_service_sync()
            await graph_service.initialize()
            status["knowledge_graph"] = {
                "status": "ok",
                "backend": config.knowledge_graph.backend,
            }
            logger.info(f"Knowledge Graph initialized: {config.knowledge_graph.backend}")
        except Exception as e:
            status["knowledge_graph"] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to initialize Knowledge Graph: {e}")

        # 5. Initialize Database (PostgreSQL)
        if not skip_db:
            try:
                db = await get_layer_intelligence_db()
                status["database"] = {"status": "ok"}
                logger.info("Layer Intelligence database tables initialized")
            except Exception as e:
                status["database"] = {"status": "error", "error": str(e)}
                logger.warning(f"Database initialization failed: {e}")

        # 6. Initialize Learning Service
        try:
            learning_service = get_learning_service()
            await learning_service.initialize()
            status["learning_service"] = {"status": "ok"}
            logger.info("Learning Service initialized")
        except Exception as e:
            status["learning_service"] = {"status": "error", "error": str(e)}
            logger.warning(f"Learning Service initialization failed: {e}")

        # 7. Initialize Self-Query Retriever
        try:
            retriever = get_self_query_retriever()
            status["self_query_retriever"] = {"status": "ok"}
            logger.info("Self-Query Retriever initialized")
        except Exception as e:
            status["self_query_retriever"] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to initialize Self-Query Retriever: {e}")

        # 8. Initialize Orchestrator
        try:
            orchestrator = get_orchestrator()
            status["orchestrator"] = {"status": "ok"}
            logger.info("Agentic Orchestrator initialized")
        except Exception as e:
            status["orchestrator"] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to initialize Orchestrator: {e}")

        _initialized = True
        status["overall"] = "ok" if all(
            s.get("status") == "ok" for s in status.values() if isinstance(s, dict)
        ) else "partial"

        logger.info(f"Layer Intelligence initialization complete: {status['overall']}")
        return status

    except Exception as e:
        logger.error(f"Layer Intelligence initialization failed: {e}")
        return {"overall": "error", "error": str(e)}


def is_initialized() -> bool:
    """Check if Layer Intelligence is initialized."""
    return _initialized
