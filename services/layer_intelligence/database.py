"""
Database schema and operations for Layer Intelligence System.

Provides PostgreSQL tables for:
- Layer catalog metadata
- Field metadata with semantic descriptions
- Query patterns for learning
- Insights storage
- Sync and audit logs
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from decouple import config as env_config

logger = logging.getLogger(__name__)

# SQL Schema for Layer Intelligence
LAYER_INTELLIGENCE_SCHEMA = """
-- ============================================================================
-- Layer Intelligence Database Schema
-- ============================================================================

-- Enable required extensions (run as superuser if needed)
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Layer Catalog Tables
-- ============================================================================

-- Main layer metadata table
CREATE TABLE IF NOT EXISTS layer_catalog (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    arcgis_item_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) UNIQUE NOT NULL,
    display_name VARCHAR(500) NOT NULL,

    -- Source
    layer_url TEXT NOT NULL,
    portal_url TEXT,
    owner VARCHAR(200),

    -- Type
    layer_type VARCHAR(50) NOT NULL DEFAULT 'feature',
    geometry_type VARCHAR(50),

    -- AI-generated content
    description TEXT,
    category VARCHAR(100),
    semantic_tags TEXT[],

    -- Query assistance
    common_queries JSONB DEFAULT '[]',
    query_templates JSONB DEFAULT '{}',

    -- Metadata
    record_count INTEGER,
    extent JSONB,
    last_synced_at TIMESTAMP WITH TIME ZONE,

    -- Access control
    requires_authentication BOOLEAN DEFAULT TRUE,
    organization_id UUID,

    -- Vector reference
    embedding_id VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Layer fields with semantic information
CREATE TABLE IF NOT EXISTS layer_fields (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_id UUID NOT NULL REFERENCES layer_catalog(id) ON DELETE CASCADE,

    -- Field identity
    name VARCHAR(200) NOT NULL,
    alias VARCHAR(500),
    field_type VARCHAR(100) NOT NULL,

    -- AI-generated semantics
    semantic_description TEXT,
    related_concepts TEXT[],

    -- Query assistance
    sample_values JSONB,
    value_range JSONB,
    is_filterable BOOLEAN DEFAULT TRUE,
    is_numeric BOOLEAN DEFAULT FALSE,
    is_date BOOLEAN DEFAULT FALSE,
    common_operators TEXT[],

    -- Vector reference
    embedding_id VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(layer_id, name)
);

-- Index for fast field lookups
CREATE INDEX IF NOT EXISTS idx_layer_fields_layer_id ON layer_fields(layer_id);
CREATE INDEX IF NOT EXISTS idx_layer_fields_name ON layer_fields(name);
CREATE INDEX IF NOT EXISTS idx_layer_catalog_category ON layer_catalog(category);
CREATE INDEX IF NOT EXISTS idx_layer_catalog_name ON layer_catalog(name);

-- ============================================================================
-- Learning & Insights Tables
-- ============================================================================

-- Store successful query patterns for learning
CREATE TABLE IF NOT EXISTS query_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query info
    natural_language_query TEXT NOT NULL,
    intent_classification VARCHAR(100),

    -- Execution info
    layers_used TEXT[],
    structured_queries JSONB,
    filters_used JSONB,

    -- Results
    was_successful BOOLEAN DEFAULT TRUE,
    user_feedback VARCHAR(50),
    feedback_comment TEXT,

    -- Performance
    execution_time_ms INTEGER,
    confidence_score FLOAT,

    -- Vector reference for similarity search
    query_embedding_id VARCHAR(100),

    -- User context
    user_id UUID,
    session_id UUID,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Store generated insights
CREATE TABLE IF NOT EXISTS query_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Context
    query_pattern_id UUID REFERENCES query_patterns(id),
    layers_involved TEXT[],

    -- Insight content
    insight_type VARCHAR(100),
    title VARCHAR(500),
    content TEXT NOT NULL,

    -- Spatial context
    location_name VARCHAR(500),
    geometry JSONB,

    -- Metadata
    confidence_score FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- Sync & Audit Tables
-- ============================================================================

-- Track layer sync history
CREATE TABLE IF NOT EXISTS layer_sync_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_id UUID REFERENCES layer_catalog(id) ON DELETE CASCADE,

    sync_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,

    items_processed INTEGER DEFAULT 0,
    errors JSONB,

    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Agent execution audit
CREATE TABLE IF NOT EXISTS agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query info
    query TEXT NOT NULL,

    -- Execution steps
    steps JSONB NOT NULL,

    -- Results
    response TEXT,
    layers_used TEXT[],
    confidence_score FLOAT,

    -- Performance
    execution_time_ms INTEGER,
    token_usage JSONB,

    -- User context
    user_id UUID,
    session_id UUID,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for analytics
CREATE INDEX IF NOT EXISTS idx_query_patterns_created ON query_patterns(created_at);
CREATE INDEX IF NOT EXISTS idx_query_patterns_intent ON query_patterns(intent_classification);
CREATE INDEX IF NOT EXISTS idx_agent_executions_created ON agent_executions(created_at);
"""


class LayerIntelligenceDB:
    """
    Database operations for Layer Intelligence system.

    Handles PostgreSQL storage for layer metadata, query patterns,
    and learning data.
    """

    def __init__(self, connection_pool=None):
        self._pool = connection_pool
        self._initialized = False

    async def initialize(self, pool=None):
        """Initialize database tables."""
        if pool:
            self._pool = pool

        if not self._pool:
            # Try to get from existing config
            try:
                from config.database import get_db_pool
                self._pool = await get_db_pool()
            except ImportError:
                logger.warning("No database pool available, using standalone connection")
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    host=env_config("DB_HOST", default="localhost"),
                    port=env_config("DB_PORT", default=5432, cast=int),
                    user=env_config("DB_USER", default="postgres"),
                    password=env_config("DB_PASSWORD", default=""),
                    database=env_config("DB_NAME", default="market_insights"),
                    min_size=1,
                    max_size=5,
                )

        # Create tables
        async with self._pool.acquire() as conn:
            await conn.execute(LAYER_INTELLIGENCE_SCHEMA)

        self._initialized = True
        logger.info("Layer Intelligence database tables initialized")

    async def close(self):
        """Close database connections."""
        if self._pool:
            await self._pool.close()

    # =========================================================================
    # Layer Catalog Operations
    # =========================================================================

    async def upsert_layer(self, layer: "LayerMetadata") -> str:
        """Insert or update a layer in the catalog."""
        from .models import LayerMetadata

        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO layer_catalog (
                    arcgis_item_id, name, display_name, layer_url, portal_url,
                    owner, layer_type, geometry_type, description, category,
                    semantic_tags, common_queries, query_templates, record_count,
                    extent, requires_authentication, organization_id, embedding_id,
                    last_synced_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, NOW())
                ON CONFLICT (arcgis_item_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    display_name = EXCLUDED.display_name,
                    layer_url = EXCLUDED.layer_url,
                    description = EXCLUDED.description,
                    category = EXCLUDED.category,
                    semantic_tags = EXCLUDED.semantic_tags,
                    common_queries = EXCLUDED.common_queries,
                    record_count = EXCLUDED.record_count,
                    extent = EXCLUDED.extent,
                    embedding_id = EXCLUDED.embedding_id,
                    last_synced_at = EXCLUDED.last_synced_at,
                    updated_at = NOW()
                RETURNING id
            """,
                layer.arcgis_item_id,
                layer.name,
                layer.display_name,
                layer.layer_url,
                layer.portal_url,
                layer.owner,
                layer.layer_type.value if layer.layer_type else "feature",
                layer.geometry_type.value if layer.geometry_type else None,
                layer.description,
                layer.category,
                layer.semantic_tags,
                layer.common_queries,
                layer.query_templates,
                layer.record_count,
                layer.extent,
                layer.requires_authentication,
                layer.organization_id,
                layer.embedding_id,
                layer.last_synced,
            )
            return str(result["id"])

    async def upsert_field(self, layer_id: str, field: "FieldMetadata") -> str:
        """Insert or update a field for a layer."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO layer_fields (
                    layer_id, name, alias, field_type, semantic_description,
                    related_concepts, sample_values, value_range, is_filterable,
                    is_numeric, is_date, common_operators, embedding_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (layer_id, name) DO UPDATE SET
                    alias = EXCLUDED.alias,
                    semantic_description = EXCLUDED.semantic_description,
                    related_concepts = EXCLUDED.related_concepts,
                    sample_values = EXCLUDED.sample_values,
                    value_range = EXCLUDED.value_range,
                    is_filterable = EXCLUDED.is_filterable,
                    is_numeric = EXCLUDED.is_numeric,
                    is_date = EXCLUDED.is_date,
                    common_operators = EXCLUDED.common_operators,
                    embedding_id = EXCLUDED.embedding_id
                RETURNING id
            """,
                layer_id,
                field.name,
                field.alias,
                field.field_type,
                field.semantic_description,
                field.related_concepts,
                field.sample_values,
                field.value_range,
                field.is_filterable,
                field.is_numeric,
                field.is_date,
                field.common_operators,
                field.embedding_id,
            )
            return str(result["id"])

    async def get_layer_by_name(self, name: str) -> Optional[dict]:
        """Get a layer by its unique name."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM layer_catalog WHERE name = $1", name
            )
            return dict(row) if row else None

    async def get_all_layers(self, category: Optional[str] = None) -> list[dict]:
        """Get all layers, optionally filtered by category."""
        async with self._pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    "SELECT * FROM layer_catalog WHERE category = $1 ORDER BY display_name",
                    category,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM layer_catalog ORDER BY display_name"
                )
            return [dict(row) for row in rows]

    async def get_layer_fields(self, layer_id: str) -> list[dict]:
        """Get all fields for a layer."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM layer_fields WHERE layer_id = $1 ORDER BY name",
                layer_id,
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # Query Pattern Operations
    # =========================================================================

    async def store_query_pattern(
        self,
        query: str,
        intent: str,
        layers_used: list[str],
        structured_queries: list[dict],
        filters_used: list[dict],
        was_successful: bool,
        execution_time_ms: int,
        confidence_score: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Store a query pattern for learning."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO query_patterns (
                    natural_language_query, intent_classification, layers_used,
                    structured_queries, filters_used, was_successful,
                    execution_time_ms, confidence_score, user_id, session_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """,
                query,
                intent,
                layers_used,
                structured_queries,
                filters_used,
                was_successful,
                execution_time_ms,
                confidence_score,
                user_id,
                session_id,
            )
            return str(result["id"])

    async def update_query_feedback(
        self,
        pattern_id: str,
        feedback: str,
        comment: Optional[str] = None,
    ):
        """Update user feedback for a query pattern."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE query_patterns
                SET user_feedback = $2, feedback_comment = $3
                WHERE id = $1
            """, pattern_id, feedback, comment)

    async def get_similar_patterns(
        self,
        intent: str,
        layers: list[str],
        limit: int = 5,
    ) -> list[dict]:
        """Get similar successful query patterns for learning."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM query_patterns
                WHERE was_successful = TRUE
                AND intent_classification = $1
                AND layers_used && $2
                ORDER BY confidence_score DESC, created_at DESC
                LIMIT $3
            """, intent, layers, limit)
            return [dict(row) for row in rows]

    # =========================================================================
    # Insight Operations
    # =========================================================================

    async def store_insight(
        self,
        query_pattern_id: Optional[str],
        layers_involved: list[str],
        insight_type: str,
        title: str,
        content: str,
        location_name: Optional[str] = None,
        geometry: Optional[dict] = None,
        confidence_score: float = 0.0,
    ) -> str:
        """Store a generated insight."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO query_insights (
                    query_pattern_id, layers_involved, insight_type,
                    title, content, location_name, geometry, confidence_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                query_pattern_id,
                layers_involved,
                insight_type,
                title,
                content,
                location_name,
                geometry,
                confidence_score,
            )
            return str(result["id"])

    # =========================================================================
    # Audit Operations
    # =========================================================================

    async def log_agent_execution(
        self,
        query: str,
        steps: list[dict],
        response: str,
        layers_used: list[str],
        confidence_score: float,
        execution_time_ms: int,
        token_usage: Optional[dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log an agent execution for audit."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO agent_executions (
                    query, steps, response, layers_used, confidence_score,
                    execution_time_ms, token_usage, user_id, session_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                query,
                steps,
                response,
                layers_used,
                confidence_score,
                execution_time_ms,
                token_usage,
                user_id,
                session_id,
            )
            return str(result["id"])

    async def log_sync_start(self, layer_id: Optional[str], sync_type: str) -> str:
        """Log the start of a sync operation."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO layer_sync_log (layer_id, sync_type, status)
                VALUES ($1, $2, 'started')
                RETURNING id
            """, layer_id, sync_type)
            return str(result["id"])

    async def log_sync_complete(
        self,
        sync_id: str,
        items_processed: int,
        errors: Optional[list] = None,
    ):
        """Log the completion of a sync operation."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE layer_sync_log
                SET status = $2, items_processed = $3, errors = $4, completed_at = NOW()
                WHERE id = $1
            """, sync_id, "completed" if not errors else "failed", items_processed, errors)


# =============================================================================
# Factory Functions
# =============================================================================

_db_instance: Optional[LayerIntelligenceDB] = None


async def get_layer_intelligence_db(
    force_new: bool = False,
) -> LayerIntelligenceDB:
    """Get or create the database instance."""
    global _db_instance

    if _db_instance is None or force_new:
        _db_instance = LayerIntelligenceDB()
        await _db_instance.initialize()

    return _db_instance


async def initialize_layer_intelligence_db(pool=None) -> LayerIntelligenceDB:
    """Initialize the database with an existing pool."""
    global _db_instance

    _db_instance = LayerIntelligenceDB(pool)
    await _db_instance.initialize(pool)

    return _db_instance
