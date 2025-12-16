"""
Data models for Layer Intelligence System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================

class LayerType(str, Enum):
    """Types of ArcGIS layers."""
    FEATURE = "feature"
    MAP_IMAGE = "map_image"
    TILE = "tile"
    GROUP = "group"
    VECTOR_TILE = "vector_tile"
    SCENE = "scene"


class GeometryType(str, Enum):
    """Geometry types for feature layers."""
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"
    ENVELOPE = "envelope"
    NONE = "none"


class QueryOperation(str, Enum):
    """Types of query operations."""
    QUERY = "query"
    STATISTICS = "statistics"
    SPATIAL_QUERY = "spatial_query"
    COMPARE = "compare"
    TREND = "trend"
    AGGREGATE = "aggregate"


class FilterOperator(str, Enum):
    """SQL-like filter operators."""
    EQUALS = "="
    NOT_EQUALS = "<>"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    LIKE = "LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


class RelationshipType(str, Enum):
    """Types of relationships in knowledge graph."""
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
    SAME_AS = "same_as"
    COMBINES_WITH = "combines_with"


# =============================================================================
# Field Models
# =============================================================================

@dataclass
class FieldMetadata:
    """Rich metadata about a layer field."""

    # Identity
    name: str
    alias: str
    field_type: str  # esriFieldTypeString, esriFieldTypeInteger, etc.

    # AI-generated semantic information
    semantic_description: str = ""
    sample_values: list[Any] = field(default_factory=list)
    value_range: Optional[tuple] = None  # For numeric: (min, max)

    # Query assistance
    is_filterable: bool = True
    is_numeric: bool = False
    is_date: bool = False
    common_operators: list[str] = field(default_factory=list)

    # Relationships
    related_concepts: list[str] = field(default_factory=list)

    # Embedding
    embedding_id: Optional[str] = None

    def __post_init__(self):
        """Set derived properties based on field type."""
        numeric_types = [
            "esriFieldTypeInteger",
            "esriFieldTypeSmallInteger",
            "esriFieldTypeDouble",
            "esriFieldTypeSingle",
            "esriFieldTypeOID",
        ]
        date_types = ["esriFieldTypeDate"]

        self.is_numeric = self.field_type in numeric_types
        self.is_date = self.field_type in date_types

        if not self.common_operators:
            if self.is_numeric:
                self.common_operators = ["=", "<>", ">", "<", ">=", "<=", "BETWEEN"]
            elif self.is_date:
                self.common_operators = ["=", "<>", ">", "<", ">=", "<=", "BETWEEN"]
            else:
                self.common_operators = ["=", "<>", "LIKE", "IN"]


# =============================================================================
# Layer Models
# =============================================================================

@dataclass
class LayerMetadata:
    """Complete metadata for an ArcGIS layer."""

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    arcgis_item_id: str = ""
    name: str = ""  # Unique: "dwellsyiq_median_rent"
    display_name: str = ""  # Human: "DwellsyIQ Median Rent (MSA)"

    # Source
    layer_url: str = ""
    portal_url: str = ""
    owner: str = ""

    # Type information
    layer_type: LayerType = LayerType.FEATURE
    geometry_type: Optional[GeometryType] = None

    # Schema
    fields: list[FieldMetadata] = field(default_factory=list)

    # AI-generated content
    description: str = ""
    semantic_tags: list[str] = field(default_factory=list)
    category: str = ""  # demographics, residential, commercial, etc.

    # Query assistance
    common_queries: list[str] = field(default_factory=list)
    query_templates: dict[str, str] = field(default_factory=dict)

    # Relationships
    related_layers: list[str] = field(default_factory=list)
    complements: list[str] = field(default_factory=list)

    # Metadata
    last_synced: Optional[datetime] = None
    record_count: Optional[int] = None
    extent: Optional[dict] = None

    # Access control
    requires_authentication: bool = True
    organization_id: Optional[str] = None

    # Embedding
    embedding_id: Optional[str] = None

    def get_field(self, field_name: str) -> Optional[FieldMetadata]:
        """Get a field by name."""
        for f in self.fields:
            if f.name.lower() == field_name.lower():
                return f
        return None

    def get_numeric_fields(self) -> list[FieldMetadata]:
        """Get all numeric fields."""
        return [f for f in self.fields if f.is_numeric]

    def get_text_fields(self) -> list[FieldMetadata]:
        """Get all text fields."""
        return [f for f in self.fields if not f.is_numeric and not f.is_date]

    def to_context_string(self) -> str:
        """Generate context string for AI prompts."""
        parts = [f"**{self.display_name}** (`{self.name}`)"]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.category:
            parts.append(f"Category: {self.category}")

        if self.fields:
            field_strs = []
            for f in self.fields[:10]:  # Limit to 10 fields
                if f.name.lower() not in ["objectid", "shape", "shape_length", "shape_area"]:
                    desc = f" - {f.semantic_description}" if f.semantic_description else ""
                    field_strs.append(f"  - {f.alias or f.name} (`{f.name}`){desc}")
            if field_strs:
                parts.append("Fields:\n" + "\n".join(field_strs))

        if self.common_queries:
            parts.append("Example queries:\n" + "\n".join(f"  - {q}" for q in self.common_queries[:3]))

        return "\n".join(parts)


@dataclass
class LayerSearchResult:
    """Result from semantic layer search."""
    layer: LayerMetadata
    similarity_score: float
    match_reason: str = ""
    suggested_fields: list[str] = field(default_factory=list)

    def __lt__(self, other: "LayerSearchResult") -> bool:
        return self.similarity_score < other.similarity_score


# =============================================================================
# Query Models
# =============================================================================

@dataclass
class QueryFilter:
    """A filter condition for the query."""
    field: str
    operator: FilterOperator
    value: Any

    def to_where_clause(self) -> str:
        """Convert to SQL WHERE clause fragment."""
        if self.operator == FilterOperator.IN:
            if isinstance(self.value, (list, tuple)):
                values = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in self.value
                )
            else:
                values = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
            return f"{self.field} IN ({values})"

        elif self.operator == FilterOperator.NOT_IN:
            if isinstance(self.value, (list, tuple)):
                values = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in self.value
                )
            else:
                values = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
            return f"{self.field} NOT IN ({values})"

        elif self.operator == FilterOperator.LIKE:
            return f"{self.field} LIKE '%{self.value}%'"

        elif self.operator == FilterOperator.BETWEEN:
            if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                return f"{self.field} BETWEEN {self.value[0]} AND {self.value[1]}"
            return f"{self.field} = {self.value}"

        elif self.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return f"{self.field} {self.operator.value}"

        else:
            if isinstance(self.value, str):
                return f"{self.field} {self.operator.value} '{self.value}'"
            return f"{self.field} {self.operator.value} {self.value}"


@dataclass
class StructuredQuery:
    """A structured query ready for execution."""

    # Target
    layer_name: str
    layer_url: str = ""

    # Operation
    operation: QueryOperation = QueryOperation.QUERY

    # Fields to return
    out_fields: list[str] = field(default_factory=lambda: ["*"])

    # Filters
    where_clause: Optional[str] = None
    filters: list[QueryFilter] = field(default_factory=list)

    # Spatial
    geometry: Optional[dict] = None
    geometry_type: str = "esriGeometryEnvelope"
    spatial_relationship: str = "esriSpatialRelIntersects"
    in_sr: int = 4326
    out_sr: int = 4326

    # Statistics
    statistics_fields: Optional[list[dict]] = None
    group_by: Optional[list[str]] = None

    # Pagination
    result_offset: int = 0
    result_record_count: int = 1000

    # Ordering
    order_by: Optional[str] = None

    # Return options
    return_geometry: bool = True
    return_count_only: bool = False

    def build_where_clause(self) -> str:
        """Build WHERE clause from filters."""
        if self.where_clause:
            return self.where_clause

        if not self.filters:
            return "1=1"

        clauses = [f.to_where_clause() for f in self.filters]
        return " AND ".join(clauses)

    def to_arcgis_params(self) -> dict:
        """Convert to ArcGIS REST API query parameters."""
        params = {
            "where": self.build_where_clause(),
            "outFields": ",".join(self.out_fields),
            "returnGeometry": str(self.return_geometry).lower(),
            "f": "json",
            "resultOffset": self.result_offset,
            "resultRecordCount": self.result_record_count,
            "outSR": self.out_sr,
        }

        if self.geometry:
            params["geometry"] = self.geometry
            params["geometryType"] = self.geometry_type
            params["spatialRel"] = self.spatial_relationship
            params["inSR"] = self.in_sr

        if self.statistics_fields:
            params["outStatistics"] = self.statistics_fields
            params["returnGeometry"] = "false"
            if self.group_by:
                params["groupByFieldsForStatistics"] = ",".join(self.group_by)

        if self.order_by:
            params["orderByFields"] = self.order_by

        if self.return_count_only:
            params["returnCountOnly"] = "true"

        return params


@dataclass
class QueryPlan:
    """A plan for executing one or more queries."""
    queries: list[StructuredQuery] = field(default_factory=list)
    reasoning: str = ""
    requires_post_processing: bool = False
    post_processing_instructions: Optional[str] = None

    def is_multi_layer(self) -> bool:
        """Check if plan involves multiple layers."""
        if len(self.queries) <= 1:
            return False
        layer_names = set(q.layer_name for q in self.queries)
        return len(layer_names) > 1


# =============================================================================
# Knowledge Graph Models
# =============================================================================

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    node_type: str  # "layer", "concept", "field", "metric"
    name: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """A relationship between nodes."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: dict = field(default_factory=dict)


@dataclass
class ReasoningPath:
    """A path through the graph for multi-hop reasoning."""
    nodes: list[GraphNode] = field(default_factory=list)
    relationships: list[GraphRelationship] = field(default_factory=list)
    explanation: str = ""
    score: float = 0.0


# =============================================================================
# Agent Models
# =============================================================================

class AgentAction(str, Enum):
    """Actions the agent can take."""
    DISCOVER_LAYERS = "discover_layers"
    QUERY_LAYER = "query_layer"
    ANALYZE_RESULTS = "analyze_results"
    CROSS_REFERENCE = "cross_reference"
    GENERATE_INSIGHT = "generate_insight"
    STORE_LEARNING = "store_learning"
    ASK_CLARIFICATION = "ask_clarification"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    action: AgentAction
    reasoning: str
    inputs: dict = field(default_factory=dict)
    outputs: Optional[dict] = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    layers_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    execution_time_ms: int = 0


# =============================================================================
# Insight Models
# =============================================================================

@dataclass
class QueryInsight:
    """An insight learned from a query execution."""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    intent: str = ""
    layers_used: list[str] = field(default_factory=list)
    filters_used: list[dict] = field(default_factory=list)
    was_successful: bool = True
    user_feedback: Optional[str] = None  # positive, negative, neutral
    created_at: datetime = field(default_factory=datetime.utcnow)
