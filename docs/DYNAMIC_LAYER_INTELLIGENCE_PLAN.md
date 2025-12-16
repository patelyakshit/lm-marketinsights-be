# Dynamic Layer Intelligence System - Comprehensive Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to build an **Agentic RAG system with Knowledge Graph** that enables your AI platform to automatically understand and query ANY data layer without manual knowledge base creation.

**Goal**: When you add a new layer (like DwellsyIQ), the AI automatically discovers it, understands its schema, and can answer user queries about it - with ZERO manual configuration.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack Recommendations](#2-technology-stack-recommendations)
3. [Component Design](#3-component-design)
4. [Database Schema](#4-database-schema)
5. [Implementation Phases](#5-implementation-phases)
6. [API Design](#6-api-design)
7. [Integration with Existing System](#7-integration-with-existing-system)
8. [Testing Strategy](#8-testing-strategy)
9. [Performance Considerations](#9-performance-considerations)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Architecture Overview

### Current vs Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query ──> Root Agent ──> Hardcoded Knowledge ──> Response            │
│                      │                                                       │
│                      ├──> GIS Agent (knows only Tapestry fields)            │
│                      └──> RAG Agent (Vertex AI - static corpus)             │
│                                                                              │
│   PROBLEM: Agent doesn't know about DwellsyIQ, Crime Data, Traffic, etc.    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────────┐                            │
│                          │    User Query       │                            │
│                          └──────────┬──────────┘                            │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     AGENTIC ORCHESTRATOR                             │   │
│   │  • Intent Classification                                             │   │
│   │  • Query Planning                                                    │   │
│   │  • Multi-step Execution                                              │   │
│   │  • Result Synthesis                                                  │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
│               ┌───────────────────┼───────────────────┐                     │
│               │                   │                   │                     │
│               ▼                   ▼                   ▼                     │
│   ┌───────────────────┐  ┌───────────────┐  ┌───────────────────┐         │
│   │  LAYER DISCOVERY  │  │  KNOWLEDGE    │  │  SELF-QUERY       │         │
│   │    SERVICE        │  │    GRAPH      │  │   RETRIEVER       │         │
│   │                   │  │               │  │                   │         │
│   │ • Semantic search │  │ • Layer       │  │ • NL → Structured │         │
│   │ • Schema lookup   │  │   relationships│  │   Query           │         │
│   │ • Field metadata  │  │ • Multi-hop   │  │ • Filter          │         │
│   │                   │  │   reasoning   │  │   generation      │         │
│   └─────────┬─────────┘  └───────┬───────┘  └─────────┬─────────┘         │
│             │                    │                    │                     │
│             └────────────────────┼────────────────────┘                     │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     LAYER CATALOG (PostgreSQL + pgvector)            │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│   │  │DwellsyIQ │  │ Tapestry │  │ Traffic  │  │ 90+ more │            │   │
│   │  │ Fields   │  │  Fields  │  │  Fields  │  │  layers  │            │   │
│   │  │ Metadata │  │ Metadata │  │ Metadata │  │          │            │   │
│   │  │ Embedding│  │ Embedding│  │ Embedding│  │          │            │   │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       LEARNING LOOP                                  │   │
│   │  • Store successful queries as insights                              │   │
│   │  • Improve recommendations over time                                 │   │
│   │  • Track query patterns                                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack Recommendations

### Based on 2025 Research & Best Practices

| Component | Recommended | Alternative | Rationale |
|-----------|-------------|-------------|-----------|
| **Vector Database** | **Qdrant** | pgvector | Rust-based, excellent filtering, scales to 10M+ vectors, LangChain native support |
| **Knowledge Graph** | **FalkorDB** | Neo4j / NetworkX | Ultra-low latency (140ms p99), LLM-optimized, 10K+ tenant support, open-source |
| **Embedding Model** | **Cohere embed-v4** | OpenAI text-embedding-3-small | Better cost ($0.50/1M vs $1.30/1M), strong multilingual, excellent recall |
| **Query Generation** | **Custom Self-Query** | LangChain SelfQueryRetriever | Geospatial-aware, ArcGIS-specific SQL generation |
| **Agent Framework** | **Google ADK** (existing) | LangGraph | Already in use, proven, minimal migration |
| **Observability** | **LangSmith** or **Arize** | Custom logging | Critical for debugging agent decisions |

### Why These Choices?

#### Qdrant over pgvector
- pgvector struggles beyond 5M vectors; you have 90+ layers with potentially millions of records
- Qdrant has native metadata filtering - critical for layer-specific queries
- Production-proven: 140ms p99 latency at scale
- LangChain/LlamaIndex native integration

#### FalkorDB over Neo4j
- 40x faster for aggregate expansion operations (140ms vs 40+ seconds)
- Purpose-built for GraphRAG and LLM integration
- Open-source with horizontal scaling
- Supports multi-tenant architecture (your organization model)

#### Cohere over OpenAI Embeddings
- Better price/performance ratio for high-volume use cases
- Superior multilingual support (100+ languages)
- Works well with Qdrant
- Doesn't require reranking for good results

---

## 3. Component Design

### 3.1 Layer Catalog Service

```python
# /services/layer_catalog.py

"""
Layer Catalog Service - Auto-discovers and indexes ArcGIS layers for AI querying.

Responsibilities:
1. Sync layers from ArcGIS Online groups
2. Extract field metadata and generate semantic descriptions
3. Create vector embeddings for semantic search
4. Maintain layer relationships in knowledge graph
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class LayerType(str, Enum):
    FEATURE = "feature"
    MAP_IMAGE = "map_image"
    TILE = "tile"
    GROUP = "group"


class GeometryType(str, Enum):
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"


@dataclass
class FieldMetadata:
    """Rich metadata about a layer field"""
    name: str
    alias: str
    field_type: str  # esriFieldTypeString, esriFieldTypeInteger, etc.

    # AI-generated semantic information
    semantic_description: str = ""  # LLM-generated: "Median rent for 2-bedroom apartments in USD"
    sample_values: list[Any] = field(default_factory=list)
    value_range: Optional[tuple] = None  # For numeric fields: (min, max)

    # Query assistance
    is_filterable: bool = True
    is_numeric: bool = False
    common_operators: list[str] = field(default_factory=list)  # ["=", ">", "<", "LIKE"]

    # Relationships
    related_concepts: list[str] = field(default_factory=list)  # ["rental_market", "housing", "affordability"]


@dataclass
class LayerMetadata:
    """Complete metadata for an ArcGIS layer"""
    # Identity
    id: str
    arcgis_item_id: str
    name: str  # Unique identifier: "dwellsyiq_median_rent"
    display_name: str  # Human-readable: "DwellsyIQ Median Rent (MSA)"

    # Source
    layer_url: str
    portal_url: str
    owner: str

    # Type information
    layer_type: LayerType
    geometry_type: Optional[GeometryType]

    # Schema
    fields: list[FieldMetadata]

    # AI-generated content
    description: str  # LLM-enhanced description
    semantic_tags: list[str]  # ["rental", "housing", "market_trends", "msa"]
    category: str  # "demographics", "residential", "commercial", etc.

    # Query assistance
    common_queries: list[str]  # Example natural language queries
    query_templates: dict[str, str]  # Named SQL templates

    # Vector embedding (stored separately in Qdrant)
    embedding_id: Optional[str] = None

    # Relationships (stored in FalkorDB)
    related_layers: list[str] = field(default_factory=list)
    complements: list[str] = field(default_factory=list)  # Layers that work well together

    # Metadata
    last_synced: Optional[datetime] = None
    record_count: Optional[int] = None
    extent: Optional[dict] = None

    # Access control
    requires_authentication: bool = True
    organization_id: Optional[str] = None


@dataclass
class LayerSearchResult:
    """Result from semantic layer search"""
    layer: LayerMetadata
    similarity_score: float
    match_reason: str  # "Matched on: rental, median rent, MSA"
    suggested_fields: list[str]  # Fields most relevant to query


class LayerCatalogService:
    """
    Central service for layer discovery and metadata management.

    Features:
    - Auto-sync from ArcGIS Online groups
    - Semantic search using vector embeddings
    - Field-level metadata with AI-generated descriptions
    - Query template generation
    """

    def __init__(
        self,
        qdrant_client,
        falkordb_client,
        embedding_service,
        arcgis_client,
    ):
        self.qdrant = qdrant_client
        self.graph = falkordb_client
        self.embeddings = embedding_service
        self.arcgis = arcgis_client

        # Collection name for layer embeddings
        self.collection_name = "layer_catalog"

    # =========================================================================
    # Layer Discovery
    # =========================================================================

    async def sync_from_arcgis_group(self, group_id: str) -> list[LayerMetadata]:
        """
        Sync all layers from an ArcGIS Online group.

        Steps:
        1. Fetch all items from group
        2. Extract layer metadata from each item
        3. Generate semantic descriptions using LLM
        4. Create embeddings and store in Qdrant
        5. Build relationships in FalkorDB
        """
        pass

    async def sync_single_layer(self, item_id: str) -> LayerMetadata:
        """Sync a single layer by its ArcGIS item ID"""
        pass

    async def _extract_layer_metadata(self, arcgis_item) -> LayerMetadata:
        """Extract metadata from ArcGIS item and feature service"""
        pass

    async def _generate_semantic_descriptions(self, layer: LayerMetadata) -> LayerMetadata:
        """
        Use LLM to generate semantic descriptions for layer and fields.

        Example:
        Field: "F2bed_apt_median"
        Generated: "Median monthly rent for 2-bedroom apartments in USD"
        """
        pass

    # =========================================================================
    # Semantic Search
    # =========================================================================

    async def search_layers(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.3
    ) -> list[LayerSearchResult]:
        """
        Find layers relevant to a natural language query.

        Example:
        query: "What's the median rent in Dallas?"
        Returns: [
            LayerSearchResult(
                layer=DwellsyIQ,
                similarity_score=0.92,
                match_reason="Matched on: rental, median rent",
                suggested_fields=["F2bed_apt_median", "NAME"]
            )
        ]
        """
        pass

    async def get_layer_by_name(self, name: str) -> Optional[LayerMetadata]:
        """Get a specific layer by its unique name"""
        pass

    async def get_layers_by_category(self, category: str) -> list[LayerMetadata]:
        """Get all layers in a category (e.g., 'residential', 'demographics')"""
        pass

    # =========================================================================
    # Query Assistance
    # =========================================================================

    async def suggest_fields_for_query(
        self,
        layer: LayerMetadata,
        query: str
    ) -> list[FieldMetadata]:
        """
        Suggest which fields to query based on user's intent.

        Example:
        layer: DwellsyIQ
        query: "rent for 2 bedroom apartments"
        Returns: [FieldMetadata(name="F2bed_apt_median", ...)]
        """
        pass

    async def generate_query_template(
        self,
        layer: LayerMetadata,
        intent: str
    ) -> str:
        """
        Generate a SQL-like query template for the layer.

        Example:
        layer: DwellsyIQ
        intent: "filter by rent above $2000"
        Returns: "F2bed_apt_median > 2000"
        """
        pass

    # =========================================================================
    # Relationship Management
    # =========================================================================

    async def get_related_layers(
        self,
        layer_name: str,
        relationship_type: Optional[str] = None
    ) -> list[LayerMetadata]:
        """
        Get layers related to a given layer from knowledge graph.

        Example:
        layer_name: "dwellsyiq_median_rent"
        Returns: [Tapestry (complements), Demographics (relates_to)]
        """
        pass

    async def suggest_cross_layer_analysis(
        self,
        query: str
    ) -> list[tuple[LayerMetadata, str]]:
        """
        Suggest multiple layers for complex queries.

        Example:
        query: "Compare rental affordability with income levels"
        Returns: [
            (DwellsyIQ, "provides rental data"),
            (Demographics, "provides income data")
        ]
        """
        pass
```

### 3.2 Knowledge Graph Service

```python
# /services/knowledge_graph.py

"""
Knowledge Graph Service - Manages layer relationships for multi-hop reasoning.

Uses FalkorDB (Redis-based graph database) for:
1. Layer-to-Layer relationships
2. Layer-to-Concept mappings
3. Field-to-Metric associations
4. Cross-layer reasoning paths
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class RelationshipType(str, Enum):
    # Layer relationships
    RELATES_TO = "relates_to"
    COMPLEMENTS = "complements"
    CONTAINS = "contains"
    DERIVED_FROM = "derived_from"

    # Concept relationships
    MEASURES = "measures"
    INDICATES = "indicates"
    CORRELATES_WITH = "correlates_with"

    # Field relationships
    HAS_FIELD = "has_field"
    SAME_AS = "same_as"  # Same field in different layers


@dataclass
class GraphNode:
    """A node in the knowledge graph"""
    id: str
    node_type: str  # "layer", "concept", "field", "metric"
    name: str
    properties: dict


@dataclass
class GraphRelationship:
    """A relationship between nodes"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: dict = None


@dataclass
class ReasoningPath:
    """A path through the graph for multi-hop reasoning"""
    nodes: list[GraphNode]
    relationships: list[GraphRelationship]
    explanation: str  # Human-readable explanation of the path


class KnowledgeGraphService:
    """
    Manages the knowledge graph for layer relationships and reasoning.

    Graph Schema:

    (:Layer {name, display_name, category})
        -[:RELATES_TO]->(:Layer)
        -[:COMPLEMENTS]->(:Layer)
        -[:MEASURES]->(:Concept {name: "RentalMarket"})
        -[:HAS_FIELD]->(:Field {name, type, description})

    (:Concept {name})
        -[:CORRELATES_WITH]->(:Concept)
        -[:INDICATES]->(:Metric)

    (:Field {name, layer_name})
        -[:SAME_AS]->(:Field)
        -[:INDICATES]->(:Metric)
    """

    def __init__(self, falkordb_client):
        self.graph = falkordb_client
        self.graph_name = "layer_intelligence"

    # =========================================================================
    # Node Management
    # =========================================================================

    async def add_layer_node(self, layer: "LayerMetadata") -> str:
        """
        Add a layer as a node in the graph.

        Cypher:
        CREATE (l:Layer {
            name: $name,
            display_name: $display_name,
            category: $category,
            layer_url: $layer_url
        })
        """
        pass

    async def add_concept_node(self, name: str, description: str) -> str:
        """Add a domain concept (e.g., "RentalMarket", "Affordability")"""
        pass

    async def add_field_node(
        self,
        field_name: str,
        layer_name: str,
        field_type: str,
        description: str
    ) -> str:
        """Add a field as a node"""
        pass

    # =========================================================================
    # Relationship Management
    # =========================================================================

    async def create_relationship(
        self,
        source_name: str,
        relationship_type: RelationshipType,
        target_name: str,
        properties: dict = None
    ) -> bool:
        """
        Create a relationship between two nodes.

        Examples:
        - ("DwellsyIQ", MEASURES, "RentalMarket")
        - ("DwellsyIQ", COMPLEMENTS, "Tapestry")
        - ("RentalMarket", CORRELATES_WITH, "HousingDemand")
        """
        pass

    async def seed_domain_knowledge(self):
        """
        Seed the graph with domain concepts and common relationships.

        Concepts:
        - RentalMarket, HousingDemand, Affordability
        - ConsumerBehavior, Demographics, IncomeLevel
        - TrafficPattern, Accessibility, Walkability

        Relationships:
        - RentalMarket CORRELATES_WITH HousingDemand
        - Affordability INDICATES RentalMarket + IncomeLevel
        - Tapestry MEASURES ConsumerBehavior
        """
        pass

    # =========================================================================
    # Multi-Hop Reasoning
    # =========================================================================

    async def find_reasoning_path(
        self,
        start_concept: str,
        end_concept: str,
        max_hops: int = 3
    ) -> list[ReasoningPath]:
        """
        Find paths between concepts for multi-hop reasoning.

        Example:
        start: "rental affordability"
        end: "consumer segments"

        Path: RentalMarket -> Affordability -> IncomeLevel -> Tapestry
        Explanation: "Rental affordability relates to income, which maps to Tapestry segments"
        """
        pass

    async def suggest_analysis_layers(
        self,
        query: str
    ) -> list[tuple["LayerMetadata", str, float]]:
        """
        Use graph traversal to suggest layers for a complex query.

        Query: "Best areas for luxury apartments"

        Graph Traversal:
        1. "luxury" -> HighIncome concept
        2. HighIncome -> Tapestry (Affluent segments)
        3. "apartments" -> RentalMarket concept
        4. RentalMarket -> DwellsyIQ
        5. Combine: Tapestry + DwellsyIQ

        Returns: [
            (DwellsyIQ, "rental data for apartments", 0.9),
            (Tapestry, "affluent consumer segments", 0.85),
            (Demographics, "income levels", 0.7)
        ]
        """
        pass

    async def explain_layer_relationship(
        self,
        layer1_name: str,
        layer2_name: str
    ) -> str:
        """
        Generate a natural language explanation of how two layers relate.

        Example:
        layer1: "DwellsyIQ"
        layer2: "Tapestry"

        Returns: "DwellsyIQ provides rental market data while Tapestry provides
        consumer segmentation. Together they can identify which consumer segments
        are associated with different rental price points."
        """
        pass

    # =========================================================================
    # Query Support
    # =========================================================================

    async def get_field_relationships(
        self,
        field_name: str,
        layer_name: str
    ) -> list[dict]:
        """
        Find fields in other layers that relate to a given field.

        Example:
        field: "MEDHINC_CY" (median income) in Demographics

        Returns: [
            {"layer": "Tapestry", "field": "TSEGCODE", "relationship": "correlates_with"},
            {"layer": "DwellsyIQ", "field": "F2bed_apt_median", "relationship": "indicates"}
        ]
        """
        pass

    async def visualize_graph(
        self,
        center_node: str,
        depth: int = 2
    ) -> dict:
        """Return graph data for visualization (nodes and edges)"""
        pass
```

### 3.3 Self-Query Retriever

```python
# /services/self_query.py

"""
Self-Query Retriever - Converts natural language to structured layer queries.

This is the "brain" that translates user questions into:
1. Which layer(s) to query
2. Which fields to select
3. What filters to apply
4. What spatial operations to perform
"""

from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


class QueryOperation(str, Enum):
    """Types of operations that can be performed"""
    QUERY = "query"  # Simple feature query
    STATISTICS = "statistics"  # Aggregate statistics
    SPATIAL_QUERY = "spatial_query"  # Query within geometry
    COMPARE = "compare"  # Compare two areas
    TREND = "trend"  # Analyze trends over time


class FilterOperator(str, Enum):
    """SQL-like filter operators"""
    EQUALS = "="
    NOT_EQUALS = "<>"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    LIKE = "LIKE"
    IN = "IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


@dataclass
class QueryFilter:
    """A filter condition for the query"""
    field: str
    operator: FilterOperator
    value: Any

    def to_where_clause(self) -> str:
        """Convert to SQL WHERE clause fragment"""
        if self.operator == FilterOperator.IN:
            values = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in self.value])
            return f"{self.field} IN ({values})"
        elif self.operator == FilterOperator.LIKE:
            return f"{self.field} LIKE '%{self.value}%'"
        elif self.operator == FilterOperator.BETWEEN:
            return f"{self.field} BETWEEN {self.value[0]} AND {self.value[1]}"
        elif self.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return f"{self.field} {self.operator.value}"
        else:
            if isinstance(self.value, str):
                return f"{self.field} {self.operator.value} '{self.value}'"
            return f"{self.field} {self.operator.value} {self.value}"


@dataclass
class StructuredQuery:
    """A structured query ready for execution"""
    # Target
    layer_name: str
    layer_url: str

    # Operation
    operation: QueryOperation

    # Fields to return
    out_fields: list[str]

    # Filters
    where_clause: Optional[str] = None
    filters: list[QueryFilter] = None

    # Spatial
    geometry: Optional[dict] = None  # GeoJSON or ArcGIS geometry
    spatial_relationship: str = "esriSpatialRelIntersects"

    # Statistics (for STATISTICS operation)
    statistics_fields: Optional[list[dict]] = None  # [{"field": "F2bed_apt_median", "type": "avg"}]
    group_by: Optional[list[str]] = None

    # Pagination
    result_offset: int = 0
    result_record_count: int = 1000

    # Ordering
    order_by: Optional[str] = None

    def to_arcgis_params(self) -> dict:
        """Convert to ArcGIS REST API query parameters"""
        params = {
            "where": self.where_clause or "1=1",
            "outFields": ",".join(self.out_fields),
            "returnGeometry": "true",
            "f": "json",
            "resultOffset": self.result_offset,
            "resultRecordCount": self.result_record_count,
        }

        if self.geometry:
            params["geometry"] = self.geometry
            params["spatialRel"] = self.spatial_relationship

        if self.statistics_fields:
            params["outStatistics"] = self.statistics_fields
            if self.group_by:
                params["groupByFieldsForStatistics"] = ",".join(self.group_by)

        if self.order_by:
            params["orderByFields"] = self.order_by

        return params


@dataclass
class QueryPlan:
    """A plan for executing one or more queries"""
    queries: list[StructuredQuery]
    reasoning: str  # Explanation of the plan
    requires_post_processing: bool = False
    post_processing_instructions: Optional[str] = None


class SelfQueryRetriever:
    """
    Converts natural language queries to structured layer queries.

    Process:
    1. Parse user intent using LLM
    2. Match intent to relevant layers (via LayerCatalogService)
    3. Identify required fields
    4. Generate filters from constraints in the query
    5. Return executable StructuredQuery
    """

    def __init__(
        self,
        llm,
        layer_catalog: "LayerCatalogService",
        knowledge_graph: "KnowledgeGraphService",
    ):
        self.llm = llm
        self.catalog = layer_catalog
        self.graph = knowledge_graph

    async def parse_query(
        self,
        natural_language_query: str,
        context: Optional[dict] = None
    ) -> QueryPlan:
        """
        Parse a natural language query into an executable plan.

        Example:
        Input: "Show me areas in Texas with median rent above $2000 for 2BR apartments"

        Output: QueryPlan(
            queries=[
                StructuredQuery(
                    layer_name="dwellsyiq_median_rent",
                    operation=QueryOperation.QUERY,
                    out_fields=["NAME", "F2bed_apt_median", "lat", "lon"],
                    where_clause="NAME LIKE '%Texas%' AND F2bed_apt_median > 2000",
                    filters=[
                        QueryFilter("NAME", FilterOperator.LIKE, "Texas"),
                        QueryFilter("F2bed_apt_median", FilterOperator.GREATER_THAN, 2000)
                    ]
                )
            ],
            reasoning="User wants to filter DwellsyIQ rental data by location (Texas) and rent threshold ($2000 for 2BR)"
        )
        """
        # Step 1: Extract intent and entities
        intent = await self._extract_intent(natural_language_query)

        # Step 2: Find relevant layers
        layers = await self.catalog.search_layers(natural_language_query)

        # Step 3: For complex queries, check knowledge graph
        if intent.is_complex:
            related_layers = await self.graph.suggest_analysis_layers(natural_language_query)
            layers = self._merge_layer_suggestions(layers, related_layers)

        # Step 4: Generate structured queries for each layer
        queries = []
        for layer_result in layers:
            query = await self._generate_structured_query(
                layer_result.layer,
                natural_language_query,
                intent
            )
            if query:
                queries.append(query)

        # Step 5: Create execution plan
        return QueryPlan(
            queries=queries,
            reasoning=self._generate_reasoning(intent, layers),
            requires_post_processing=len(queries) > 1,
            post_processing_instructions=self._generate_post_processing(queries) if len(queries) > 1 else None
        )

    async def _extract_intent(self, query: str) -> "QueryIntent":
        """
        Use LLM to extract intent from natural language.

        Extracts:
        - Operation type (query, statistics, compare, etc.)
        - Geographic constraints (location names, coordinates)
        - Value constraints (thresholds, ranges)
        - Temporal constraints (dates, periods)
        - Entities (field names, layer names mentioned)
        """
        pass

    async def _generate_structured_query(
        self,
        layer: "LayerMetadata",
        query: str,
        intent: "QueryIntent"
    ) -> Optional[StructuredQuery]:
        """
        Generate a structured query for a specific layer.

        Uses:
        - Layer schema (field names, types)
        - Field semantic descriptions (to match intent to fields)
        - Query templates (for common patterns)
        """
        pass

    async def generate_filter_from_constraint(
        self,
        constraint: str,
        layer: "LayerMetadata"
    ) -> Optional[QueryFilter]:
        """
        Generate a filter from a constraint phrase.

        Examples:
        "above $2000" + field "F2bed_apt_median" -> QueryFilter("F2bed_apt_median", ">", 2000)
        "in Texas" + field "NAME" -> QueryFilter("NAME", "LIKE", "Texas")
        "between 2020 and 2024" + field "year" -> QueryFilter("year", "BETWEEN", [2020, 2024])
        """
        pass

    def _generate_reasoning(
        self,
        intent: "QueryIntent",
        layers: list["LayerSearchResult"]
    ) -> str:
        """Generate human-readable explanation of the query plan"""
        pass

    def _generate_post_processing(self, queries: list[StructuredQuery]) -> str:
        """Generate instructions for combining results from multiple queries"""
        pass
```

### 3.4 Agentic Orchestrator

```python
# /services/agentic_orchestrator.py

"""
Agentic Orchestrator - Autonomous agent for complex geospatial queries.

Features:
1. Dynamic layer discovery
2. Multi-step query planning
3. Cross-layer analysis
4. Learning from results
"""

from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


class AgentAction(str, Enum):
    """Actions the agent can take"""
    DISCOVER_LAYERS = "discover_layers"
    QUERY_LAYER = "query_layer"
    ANALYZE_RESULTS = "analyze_results"
    CROSS_REFERENCE = "cross_reference"
    GENERATE_INSIGHT = "generate_insight"
    STORE_LEARNING = "store_learning"


@dataclass
class AgentStep:
    """A single step in the agent's execution"""
    action: AgentAction
    reasoning: str
    inputs: dict
    outputs: Optional[dict] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentResponse:
    """Final response from the agent"""
    answer: str
    steps: list[AgentStep]
    layers_used: list[str]
    confidence: float
    suggestions: list[str]  # Follow-up questions


class AgenticOrchestrator:
    """
    Autonomous agent that orchestrates complex geospatial queries.

    Workflow:
    1. Receive natural language query
    2. Discover relevant layers (no hardcoding!)
    3. Plan multi-step analysis
    4. Execute queries with self-correction
    5. Synthesize results
    6. Learn from interaction
    """

    def __init__(
        self,
        llm,
        layer_catalog: "LayerCatalogService",
        knowledge_graph: "KnowledgeGraphService",
        self_query: "SelfQueryRetriever",
        gis_executor,
        learning_service: "LearningService",
    ):
        self.llm = llm
        self.catalog = layer_catalog
        self.graph = knowledge_graph
        self.self_query = self_query
        self.gis = gis_executor
        self.learning = learning_service

        # Execution state
        self.steps: list[AgentStep] = []
        self.max_iterations = 5

    async def process(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> AgentResponse:
        """
        Process a natural language query end-to-end.

        Example:
        Query: "Compare rental affordability between Dallas and Austin,
                and show which consumer segments live in affordable areas"

        Agent Steps:
        1. DISCOVER_LAYERS: Find DwellsyIQ (rent), Demographics (income), Tapestry (segments)
        2. QUERY_LAYER: Get median rent for Dallas MSA from DwellsyIQ
        3. QUERY_LAYER: Get median rent for Austin MSA from DwellsyIQ
        4. QUERY_LAYER: Get median income for both from Demographics
        5. ANALYZE_RESULTS: Calculate affordability ratio (rent/income)
        6. CROSS_REFERENCE: Match affordable areas to Tapestry segments
        7. GENERATE_INSIGHT: Synthesize findings
        8. STORE_LEARNING: Save query pattern for future use
        """
        self.steps = []

        # Step 1: Discover relevant layers
        layers = await self._discover_layers(query)

        # Step 2: Plan execution
        plan = await self._plan_execution(query, layers)

        # Step 3: Execute with ReAct loop
        results = {}
        for step in plan:
            result = await self._execute_step(step, results)
            if not result.success:
                # Self-correction: try alternative approach
                result = await self._self_correct(step, result.error, results)
            results[step.action.value] = result.outputs

        # Step 4: Synthesize response
        answer = await self._synthesize_response(query, results)

        # Step 5: Learn from this interaction
        await self._store_learning(query, results, answer)

        return AgentResponse(
            answer=answer,
            steps=self.steps,
            layers_used=[l.name for l in layers],
            confidence=self._calculate_confidence(),
            suggestions=await self._generate_suggestions(query, results)
        )

    async def _discover_layers(self, query: str) -> list["LayerMetadata"]:
        """
        Autonomously discover which layers are needed.

        Uses:
        1. Semantic search in layer catalog
        2. Knowledge graph for related layers
        3. No hardcoded layer names!
        """
        step = AgentStep(
            action=AgentAction.DISCOVER_LAYERS,
            reasoning="Finding layers relevant to the query",
            inputs={"query": query}
        )

        # Semantic search
        search_results = await self.catalog.search_layers(query, limit=5)

        # Graph-based suggestions
        graph_suggestions = await self.graph.suggest_analysis_layers(query)

        # Merge and deduplicate
        layers = self._merge_layer_results(search_results, graph_suggestions)

        step.outputs = {
            "layers_found": [l.name for l in layers],
            "reasoning": self._explain_layer_selection(layers)
        }
        self.steps.append(step)

        return layers

    async def _plan_execution(
        self,
        query: str,
        layers: list["LayerMetadata"]
    ) -> list[AgentStep]:
        """
        Plan the execution steps using LLM.

        Consider:
        - Dependencies between queries
        - Optimal order of operations
        - Whether cross-layer analysis is needed
        """
        pass

    async def _execute_step(
        self,
        step: AgentStep,
        previous_results: dict
    ) -> AgentStep:
        """Execute a single step and handle errors"""
        pass

    async def _self_correct(
        self,
        failed_step: AgentStep,
        error: str,
        previous_results: dict
    ) -> AgentStep:
        """
        Attempt to self-correct after a failure.

        Strategies:
        - Try alternative layer
        - Modify filter criteria
        - Use broader search
        - Ask for clarification (as last resort)
        """
        pass

    async def _synthesize_response(
        self,
        query: str,
        results: dict
    ) -> str:
        """Generate final answer from all results"""
        pass

    async def _store_learning(
        self,
        query: str,
        results: dict,
        answer: str
    ):
        """
        Store this interaction for future learning.

        Stores:
        - Query pattern
        - Layers used
        - Successful filters
        - User feedback (if provided)
        """
        pass

    def _calculate_confidence(self) -> float:
        """Calculate confidence based on step success rates"""
        if not self.steps:
            return 0.0
        successful = sum(1 for s in self.steps if s.success)
        return successful / len(self.steps)

    async def _generate_suggestions(
        self,
        query: str,
        results: dict
    ) -> list[str]:
        """Generate follow-up question suggestions"""
        pass
```

---

## 4. Database Schema

### 4.1 PostgreSQL Tables

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Layer Catalog Tables
-- ============================================================================

-- Main layer metadata table
CREATE TABLE layer_catalog (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity
    arcgis_item_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) UNIQUE NOT NULL,  -- "dwellsyiq_median_rent"
    display_name VARCHAR(500) NOT NULL,  -- "DwellsyIQ Median Rent (MSA)"

    -- Source
    layer_url TEXT NOT NULL,
    portal_url TEXT,
    owner VARCHAR(200),

    -- Type
    layer_type VARCHAR(50) NOT NULL DEFAULT 'feature',  -- feature, map_image, tile, group
    geometry_type VARCHAR(50),  -- point, polygon, polyline

    -- AI-generated content
    description TEXT,
    category VARCHAR(100),  -- demographics, residential, commercial
    semantic_tags TEXT[],  -- ["rental", "housing", "market"]

    -- Query assistance
    common_queries JSONB DEFAULT '[]',  -- Example NL queries
    query_templates JSONB DEFAULT '{}',  -- Named SQL templates

    -- Metadata
    record_count INTEGER,
    extent JSONB,
    last_synced_at TIMESTAMP WITH TIME ZONE,

    -- Access control
    requires_authentication BOOLEAN DEFAULT TRUE,
    organization_id UUID,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Layer fields with semantic information
CREATE TABLE layer_fields (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    layer_id UUID NOT NULL REFERENCES layer_catalog(id) ON DELETE CASCADE,

    -- Field identity
    name VARCHAR(200) NOT NULL,
    alias VARCHAR(500),
    field_type VARCHAR(100) NOT NULL,  -- esriFieldTypeString, etc.

    -- AI-generated semantics
    semantic_description TEXT,  -- "Median rent for 2BR apartments in USD"
    related_concepts TEXT[],  -- ["rental_market", "housing", "affordability"]

    -- Query assistance
    sample_values JSONB,
    value_range JSONB,  -- {"min": 500, "max": 5000}
    is_filterable BOOLEAN DEFAULT TRUE,
    is_numeric BOOLEAN DEFAULT FALSE,
    common_operators TEXT[],  -- ["=", ">", "<", "LIKE"]

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(layer_id, name)
);

-- Index for fast field lookups
CREATE INDEX idx_layer_fields_layer_id ON layer_fields(layer_id);
CREATE INDEX idx_layer_fields_name ON layer_fields(name);

-- ============================================================================
-- Learning & Insights Tables
-- ============================================================================

-- Store successful query patterns
CREATE TABLE query_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Query info
    natural_language_query TEXT NOT NULL,
    intent_classification VARCHAR(100),

    -- Execution info
    layers_used TEXT[],
    structured_queries JSONB,

    -- Results
    was_successful BOOLEAN DEFAULT TRUE,
    user_feedback VARCHAR(50),  -- positive, negative, null

    -- Embeddings for similarity search
    query_embedding_id VARCHAR(100),  -- Reference to Qdrant

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Store generated insights
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Context
    query_pattern_id UUID REFERENCES query_patterns(id),
    layers_involved TEXT[],

    -- Insight content
    insight_type VARCHAR(100),  -- "market_trend", "correlation", "anomaly"
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
CREATE TABLE layer_sync_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    layer_id UUID REFERENCES layer_catalog(id) ON DELETE CASCADE,

    sync_type VARCHAR(50) NOT NULL,  -- full, incremental, metadata_only
    status VARCHAR(50) NOT NULL,  -- started, completed, failed

    items_processed INTEGER DEFAULT 0,
    errors JSONB,

    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Agent execution audit
CREATE TABLE agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

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

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 4.2 Qdrant Collections

```python
# Qdrant collection schemas

# Collection: layer_catalog
# Stores embeddings for layer descriptions
layer_catalog_collection = {
    "name": "layer_catalog",
    "vectors": {
        "size": 1024,  # Cohere embed-v4 dimension
        "distance": "Cosine"
    },
    "payload_schema": {
        "layer_id": "keyword",
        "layer_name": "keyword",
        "display_name": "text",
        "category": "keyword",
        "semantic_tags": "keyword[]",
        "field_names": "keyword[]"
    }
}

# Collection: layer_fields
# Stores embeddings for field semantic descriptions
layer_fields_collection = {
    "name": "layer_fields",
    "vectors": {
        "size": 1024,
        "distance": "Cosine"
    },
    "payload_schema": {
        "field_id": "keyword",
        "layer_id": "keyword",
        "layer_name": "keyword",
        "field_name": "keyword",
        "field_type": "keyword",
        "related_concepts": "keyword[]"
    }
}

# Collection: query_patterns
# Stores embeddings for past queries (for learning)
query_patterns_collection = {
    "name": "query_patterns",
    "vectors": {
        "size": 1024,
        "distance": "Cosine"
    },
    "payload_schema": {
        "pattern_id": "keyword",
        "intent": "keyword",
        "layers_used": "keyword[]",
        "was_successful": "bool"
    }
}
```

### 4.3 FalkorDB Graph Schema

```cypher
// FalkorDB Graph Schema for Layer Intelligence

// Node types
// -----------

// Layer node
CREATE (l:Layer {
    name: "dwellsyiq_median_rent",
    display_name: "DwellsyIQ Median Rent (MSA)",
    category: "residential",
    layer_url: "https://..."
})

// Concept node (domain concepts)
CREATE (c:Concept {
    name: "RentalMarket",
    description: "Market for residential rental properties"
})

// Field node
CREATE (f:Field {
    name: "F2bed_apt_median",
    layer_name: "dwellsyiq_median_rent",
    field_type: "esriFieldTypeDouble",
    semantic_description: "Median rent for 2-bedroom apartments"
})

// Metric node (measurable outcomes)
CREATE (m:Metric {
    name: "Affordability",
    formula: "Rent / Income * 100",
    description: "Housing affordability ratio"
})


// Relationship types
// ------------------

// Layer to Layer
(:Layer)-[:RELATES_TO]->(:Layer)
(:Layer)-[:COMPLEMENTS]->(:Layer)
(:Layer)-[:DERIVED_FROM]->(:Layer)

// Layer to Concept
(:Layer)-[:MEASURES]->(:Concept)

// Layer to Field
(:Layer)-[:HAS_FIELD]->(:Field)

// Concept to Concept
(:Concept)-[:CORRELATES_WITH]->(:Concept)

// Field to Concept
(:Field)-[:INDICATES]->(:Concept)

// Field to Field (cross-layer)
(:Field)-[:SAME_AS]->(:Field)
(:Field)-[:COMBINES_WITH]->(:Field)

// Concept to Metric
(:Concept)-[:MEASURED_BY]->(:Metric)


// Example graph structure
// -----------------------

// Create nodes
CREATE (dwellsy:Layer {name: "dwellsyiq_median_rent", display_name: "DwellsyIQ Median Rent", category: "residential"})
CREATE (tapestry:Layer {name: "tapestry_segments", display_name: "Tapestry Segmentation", category: "demographics"})
CREATE (demographics:Layer {name: "usa_demographics", display_name: "USA Demographics", category: "demographics"})

CREATE (rental:Concept {name: "RentalMarket"})
CREATE (housing:Concept {name: "HousingDemand"})
CREATE (consumer:Concept {name: "ConsumerBehavior"})
CREATE (income:Concept {name: "IncomeLevel"})

CREATE (afford:Metric {name: "Affordability", formula: "Rent/Income*100"})

// Create relationships
CREATE (dwellsy)-[:MEASURES]->(rental)
CREATE (dwellsy)-[:COMPLEMENTS]->(tapestry)
CREATE (tapestry)-[:MEASURES]->(consumer)
CREATE (demographics)-[:MEASURES]->(income)
CREATE (rental)-[:CORRELATES_WITH]->(housing)
CREATE (rental)-[:COMBINED_WITH {via: afford}]->(income)
CREATE (consumer)-[:CORRELATES_WITH]->(income)
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Days 1-4)

#### Day 1: Infrastructure Setup
- [ ] Set up Qdrant (Docker or Qdrant Cloud)
- [ ] Set up FalkorDB (Docker)
- [ ] Create PostgreSQL tables
- [ ] Add required Python dependencies

```bash
# Dependencies to add to requirements.txt
qdrant-client>=1.7.0
falkordb>=1.0.0
cohere>=5.0.0
langchain>=0.1.0
langchain-community>=0.0.20
```

#### Day 2: Embedding Service
- [ ] Create Cohere embedding service
- [ ] Implement batch embedding for layers
- [ ] Create Qdrant collection management
- [ ] Test embedding generation

#### Day 3: Layer Catalog Service (Core)
- [ ] Implement `LayerCatalogService` base class
- [ ] Create ArcGIS metadata extraction
- [ ] Implement layer sync from group
- [ ] Store in PostgreSQL + Qdrant

#### Day 4: Semantic Search
- [ ] Implement `search_layers()` method
- [ ] Add metadata filtering
- [ ] Test with DwellsyIQ layer
- [ ] Benchmark search performance

### Phase 2: Knowledge Graph (Days 5-7)

#### Day 5: FalkorDB Setup
- [ ] Create graph schema
- [ ] Implement `KnowledgeGraphService`
- [ ] Seed domain concepts
- [ ] Create basic relationships

#### Day 6: Layer Relationships
- [ ] Auto-generate layer relationships
- [ ] Implement relationship queries
- [ ] Build multi-hop traversal

#### Day 7: Graph-Enhanced Search
- [ ] Integrate graph with layer discovery
- [ ] Implement `suggest_analysis_layers()`
- [ ] Test cross-layer reasoning

### Phase 3: Self-Query (Days 8-10)

#### Day 8: Query Parser
- [ ] Implement intent extraction
- [ ] Create constraint parser
- [ ] Build filter generator

#### Day 9: Structured Query Generation
- [ ] Implement `StructuredQuery` generation
- [ ] Support all filter operators
- [ ] Add spatial query support

#### Day 10: Query Plan Generation
- [ ] Implement multi-layer query plans
- [ ] Add post-processing instructions
- [ ] Test complex queries

### Phase 4: Agent Integration (Days 11-14)

#### Day 11: Agentic Orchestrator
- [ ] Implement `AgenticOrchestrator`
- [ ] Create ReAct execution loop
- [ ] Add self-correction logic

#### Day 12: GIS Agent Enhancement
- [ ] Update GIS Agent to use dynamic discovery
- [ ] Remove hardcoded layer knowledge
- [ ] Add new tools for layer queries

#### Day 13: Root Agent Integration
- [ ] Update Root Agent routing
- [ ] Add layer discovery as first step
- [ ] Test end-to-end flow

#### Day 14: Learning Loop
- [ ] Implement `LearningService`
- [ ] Store query patterns
- [ ] Build insight storage

### Phase 5: Testing & Optimization (Days 15-17)

#### Day 15: Integration Testing
- [ ] Test all scenarios
- [ ] Benchmark performance
- [ ] Fix edge cases

#### Day 16: Observability
- [ ] Add LangSmith/Arize integration
- [ ] Create execution tracing
- [ ] Build debugging tools

#### Day 17: Documentation & Deployment
- [ ] Update API documentation
- [ ] Create admin tools for layer management
- [ ] Deploy to staging

---

## 6. API Design

### REST Endpoints

```python
# /api/v1/layers.py

from fastapi import APIRouter, Depends, Query
from typing import Optional

router = APIRouter(prefix="/api/v1/layers", tags=["Layer Intelligence"])


# ============================================================================
# Layer Discovery
# ============================================================================

@router.get("/search")
async def search_layers(
    q: str = Query(..., description="Natural language search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(5, ge=1, le=20),
):
    """
    Search for layers using natural language.

    Example: GET /api/v1/layers/search?q=rental%20prices%20in%20Texas

    Returns layers ranked by relevance with suggested fields.
    """
    pass


@router.get("/{layer_name}")
async def get_layer_details(layer_name: str):
    """
    Get detailed information about a specific layer.

    Returns schema, fields, query templates, and related layers.
    """
    pass


@router.get("/{layer_name}/fields")
async def get_layer_fields(layer_name: str):
    """
    Get all fields for a layer with semantic descriptions.
    """
    pass


@router.get("/{layer_name}/related")
async def get_related_layers(
    layer_name: str,
    relationship_type: Optional[str] = None,
):
    """
    Get layers related to the specified layer.
    """
    pass


# ============================================================================
# Query Generation
# ============================================================================

@router.post("/query/generate")
async def generate_query(
    natural_language: str,
    layer_name: Optional[str] = None,
):
    """
    Generate a structured query from natural language.

    Request:
    {
        "natural_language": "Show areas with rent above $2000",
        "layer_name": "dwellsyiq_median_rent"  // optional
    }

    Response:
    {
        "queries": [...],
        "reasoning": "...",
        "suggested_layers": [...]
    }
    """
    pass


@router.post("/query/execute")
async def execute_query(
    structured_query: dict,
):
    """
    Execute a structured query against the layer.
    """
    pass


# ============================================================================
# Layer Management (Admin)
# ============================================================================

@router.post("/sync")
async def sync_layers_from_group(
    group_id: str,
    force_refresh: bool = False,
):
    """
    Sync all layers from an ArcGIS group.

    This will:
    1. Fetch all items from the group
    2. Extract layer metadata
    3. Generate semantic descriptions
    4. Create embeddings
    5. Build knowledge graph relationships
    """
    pass


@router.post("/{layer_name}/sync")
async def sync_single_layer(layer_name: str):
    """
    Re-sync a single layer's metadata.
    """
    pass


@router.get("/categories")
async def list_categories():
    """
    List all layer categories.
    """
    pass


# ============================================================================
# Knowledge Graph
# ============================================================================

@router.get("/graph/relationships")
async def get_layer_relationships(
    layer_name: Optional[str] = None,
    depth: int = Query(2, ge=1, le=4),
):
    """
    Get knowledge graph relationships for visualization.
    """
    pass


@router.post("/graph/reasoning")
async def multi_hop_reasoning(
    query: str,
    start_concept: Optional[str] = None,
):
    """
    Perform multi-hop reasoning through the knowledge graph.
    """
    pass
```

---

## 7. Integration with Existing System

### 7.1 GIS Agent Updates

```python
# Updates to /agents/gis_agent.py

# BEFORE: Hardcoded knowledge
GIS_AGENT_INSTRUCTION = """
You are an expert at working with ArcGIS layers.

## Known Layers:
- Tapestry Segmentation: Fields TSEGCODE, TSEGNAME, THHBASE
...
"""

# AFTER: Dynamic discovery
GIS_AGENT_INSTRUCTION = """
You are an expert at working with ArcGIS layers.

## IMPORTANT: Layer Discovery
You do NOT have hardcoded knowledge of available layers.
ALWAYS use the `discover_layers` tool first to find relevant layers.

## Available Tools:
- discover_layers(query): Find layers relevant to user's question
- get_layer_schema(layer_name): Get field information for a layer
- query_layer(layer_name, query): Query a layer with natural language
- cross_layer_analysis(query): Perform analysis across multiple layers

## Workflow:
1. ALWAYS call discover_layers() first
2. Use returned layer information to plan your approach
3. Query layers using natural language (tool handles conversion)
4. Synthesize results for the user
"""
```

### 7.2 New GIS Tools

```python
# /tools/layer_intelligence_tools.py

from google.adk.tools import FunctionTool


async def discover_layers(query: str) -> str:
    """
    Discover layers relevant to a natural language query.

    Args:
        query: What you're looking for (e.g., "rental prices", "population data")

    Returns:
        List of relevant layers with their fields and descriptions.

    Example:
        result = await discover_layers("median rent for apartments")
        # Returns: DwellsyIQ layer with F2bed_apt_median field
    """
    from services.layer_catalog import get_layer_catalog_service

    catalog = get_layer_catalog_service()
    results = await catalog.search_layers(query, limit=5)

    # Format for agent consumption
    output = "## Discovered Layers\n\n"
    for r in results:
        output += f"### {r.layer.display_name}\n"
        output += f"- **Name:** `{r.layer.name}`\n"
        output += f"- **Category:** {r.layer.category}\n"
        output += f"- **Relevance:** {r.similarity_score:.2f}\n"
        output += f"- **Suggested Fields:** {', '.join(r.suggested_fields)}\n"
        output += f"- **Description:** {r.layer.description}\n\n"

    return output


async def get_layer_schema(layer_name: str) -> str:
    """
    Get detailed schema information for a layer.

    Args:
        layer_name: The layer's unique name (from discover_layers)

    Returns:
        Complete field information including types and descriptions.
    """
    pass


async def query_layer_natural(
    layer_name: str,
    query: str,
    limit: int = 100
) -> str:
    """
    Query a layer using natural language.

    The query will be automatically converted to the correct SQL/filter syntax.

    Args:
        layer_name: Layer to query (from discover_layers)
        query: Natural language query (e.g., "rent above $2000 in Texas")
        limit: Maximum results to return

    Returns:
        Query results formatted as a table.

    Example:
        result = await query_layer_natural(
            "dwellsyiq_median_rent",
            "areas where 2BR apartment rent is above $2000"
        )
    """
    pass


async def cross_layer_analysis(
    query: str,
    layers: list[str] = None
) -> str:
    """
    Perform analysis across multiple layers.

    Args:
        query: What you want to analyze
        layers: Optional list of specific layers to use

    Returns:
        Combined analysis results.

    Example:
        result = await cross_layer_analysis(
            "Compare rental affordability with consumer segments in Dallas"
        )
    """
    pass


# Register tools
discover_layers_tool = FunctionTool(discover_layers)
get_layer_schema_tool = FunctionTool(get_layer_schema)
query_layer_natural_tool = FunctionTool(query_layer_natural)
cross_layer_analysis_tool = FunctionTool(cross_layer_analysis)

LAYER_INTELLIGENCE_TOOLS = [
    discover_layers_tool,
    get_layer_schema_tool,
    query_layer_natural_tool,
    cross_layer_analysis_tool,
]
```

---

## 8. Testing Strategy

### 8.1 Test Scenarios

```python
# /tests/test_layer_intelligence.py

import pytest

class TestLayerDiscovery:
    """Test layer discovery and semantic search"""

    async def test_discover_dwellsyiq_by_rental_query(self, catalog):
        """Should find DwellsyIQ when searching for rental data"""
        results = await catalog.search_layers("median rent prices")

        assert len(results) > 0
        assert any(r.layer.name == "dwellsyiq_median_rent" for r in results)

    async def test_discover_tapestry_by_consumer_query(self, catalog):
        """Should find Tapestry when searching for consumer segments"""
        results = await catalog.search_layers("consumer lifestyle segments")

        assert len(results) > 0
        assert any("tapestry" in r.layer.name.lower() for r in results)

    async def test_cross_layer_suggestion(self, catalog, graph):
        """Should suggest multiple layers for complex queries"""
        results = await graph.suggest_analysis_layers(
            "Compare rental affordability with income levels"
        )

        layer_names = [r[0].name for r in results]
        assert "dwellsyiq_median_rent" in layer_names
        assert "usa_demographics" in layer_names


class TestSelfQuery:
    """Test natural language to structured query conversion"""

    async def test_simple_filter_generation(self, self_query):
        """Should generate correct filter for simple constraint"""
        plan = await self_query.parse_query(
            "Show areas with rent above $2000"
        )

        assert len(plan.queries) > 0
        query = plan.queries[0]
        assert ">" in query.where_clause
        assert "2000" in query.where_clause

    async def test_location_filter_generation(self, self_query):
        """Should generate LIKE filter for location names"""
        plan = await self_query.parse_query(
            "Median rent in Texas"
        )

        query = plan.queries[0]
        assert "LIKE" in query.where_clause or "Texas" in query.where_clause

    async def test_multi_layer_plan(self, self_query):
        """Should generate multi-layer plan for complex queries"""
        plan = await self_query.parse_query(
            "Compare rental trends with consumer segments in Dallas"
        )

        assert len(plan.queries) >= 2
        assert plan.requires_post_processing


class TestAgenticOrchestrator:
    """Test end-to-end agent execution"""

    async def test_simple_query_execution(self, orchestrator):
        """Should successfully execute a simple query"""
        response = await orchestrator.process(
            "What is the median rent for 2BR apartments in Dallas?"
        )

        assert response.answer
        assert len(response.layers_used) > 0
        assert response.confidence > 0.5

    async def test_cross_layer_analysis(self, orchestrator):
        """Should perform cross-layer analysis"""
        response = await orchestrator.process(
            "Which consumer segments live in areas with high rental costs?"
        )

        assert len(response.layers_used) >= 2
        assert "dwellsyiq" in str(response.layers_used).lower()
        assert "tapestry" in str(response.layers_used).lower()

    async def test_self_correction(self, orchestrator):
        """Should self-correct on errors"""
        # Query with intentional ambiguity
        response = await orchestrator.process(
            "Show me the rent data"  # Ambiguous - which rent field?
        )

        # Should still succeed after clarification
        assert response.answer
        assert response.confidence > 0.3
```

---

## 9. Performance Considerations

### 9.1 Caching Strategy

```python
# /services/cache.py

from functools import lru_cache
from datetime import timedelta
import redis


class LayerIntelligenceCache:
    """
    Multi-level caching for layer intelligence.

    Levels:
    1. In-memory (LRU) - Hot queries
    2. Redis - Warm queries
    3. Database - Cold storage
    """

    def __init__(self, redis_client):
        self.redis = redis_client

        # Cache TTLs
        self.LAYER_METADATA_TTL = timedelta(hours=24)
        self.SEARCH_RESULTS_TTL = timedelta(minutes=30)
        self.QUERY_PLAN_TTL = timedelta(minutes=5)

    @lru_cache(maxsize=100)
    def get_layer_metadata_memory(self, layer_name: str):
        """In-memory cache for frequently accessed layers"""
        pass

    async def get_layer_metadata(self, layer_name: str):
        """Get layer metadata with multi-level caching"""
        # Check memory
        cached = self.get_layer_metadata_memory(layer_name)
        if cached:
            return cached

        # Check Redis
        redis_key = f"layer:{layer_name}:metadata"
        cached = await self.redis.get(redis_key)
        if cached:
            return cached

        # Fall back to database
        return None

    async def cache_search_results(
        self,
        query_hash: str,
        results: list
    ):
        """Cache search results for similar queries"""
        pass
```

### 9.2 Batch Operations

```python
# Batch embedding generation for layer sync
async def batch_generate_embeddings(
    texts: list[str],
    batch_size: int = 96
) -> list[list[float]]:
    """Generate embeddings in batches to avoid rate limits"""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await embedding_service.embed(batch)
        embeddings.extend(batch_embeddings)

        # Rate limit handling
        await asyncio.sleep(0.1)

    return embeddings
```

---

## 10. Future Enhancements

### 10.1 Planned Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Real-time sync** | WebSocket-based layer updates | Medium |
| **User feedback loop** | Learn from user corrections | High |
| **Custom layer upload** | Users can add their own layers | Medium |
| **Query history** | Track and replay past queries | Low |
| **Visualization** | Graph visualization of layer relationships | Low |
| **Multi-tenant** | Organization-specific layer catalogs | High |

### 10.2 Scalability Path

```
Current (MVP)           →  Growth            →  Enterprise
─────────────────────────────────────────────────────────────
Qdrant (Docker)         →  Qdrant Cloud      →  Qdrant Cluster
FalkorDB (Docker)       →  FalkorDB Cloud    →  Neo4j Enterprise
Single region           →  Multi-region      →  Global
~100 layers             →  ~1000 layers      →  10K+ layers
```

---

## Summary

This plan provides a comprehensive roadmap to build a **Dynamic Layer Intelligence System** that:

1. **Auto-discovers** layers from ArcGIS without manual configuration
2. **Understands** layer schemas through semantic embeddings
3. **Reasons** across layers using knowledge graphs
4. **Translates** natural language to structured queries
5. **Learns** from usage to improve over time

**Key Technology Choices:**
- **Qdrant** for vector search (scalable, fast filtering)
- **FalkorDB** for knowledge graph (LLM-optimized, low latency)
- **Cohere embed-v4** for embeddings (best cost/performance)
- **Google ADK** for agents (existing, proven)

**Timeline:** 17 days to production-ready MVP

---

## References

- [Agentic RAG Survey - arXiv](https://arxiv.org/abs/2501.09136)
- [RAG Architecture Guide 2025 - orq.ai](https://orq.ai/blog/rag-architecture)
- [FalkorDB vs Neo4j - FalkorDB](https://www.falkordb.com/blog/falkordb-vs-neo4j-for-ai-applications/)
- [Graphiti Framework - GitHub](https://github.com/getzep/graphiti)
- [Self-Query Retriever - LangChain](https://python.langchain.com/docs/how_to/self_query/)
- [Vector Database Comparison 2025](https://sysdebug.com/posts/vector-database-comparison-guide-2025/)
- [Embedding Models Comparison - Elephas](https://elephas.app/blog/best-embedding-models)
- [Spatial Text-to-SQL - arXiv](https://arxiv.org/abs/2510.21045)
- [Production Agentic RAG - DecodingML](https://decodingml.substack.com/p/llmops-for-production-agentic-rag)
