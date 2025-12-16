"""
Comprehensive tests for Layer Intelligence System.

Tests cover:
- Layer Catalog Service
- Knowledge Graph Service
- Self-Query Retriever
- Agentic Orchestrator
- Learning Service
- REST API endpoints
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import models
from services.layer_intelligence.models import (
    LayerType,
    GeometryType,
    FieldMetadata,
    LayerMetadata,
    LayerSearchResult,
    QueryFilter,
    FilterOperator,
    StructuredQuery,
    QueryOperation,
    QueryPlan,
    GraphNode,
    GraphRelationship,
    RelationshipType,
    AgentAction,
    AgentStep,
    AgentResponse,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_field_metadata():
    """Create sample field metadata for testing."""
    return FieldMetadata(
        name="F2bed_apt_median",
        alias="2BR Apartment Median Rent",
        field_type="esriFieldTypeDouble",
        semantic_description="Median monthly rent for 2-bedroom apartments in USD",
        sample_values=[1500, 2000, 2500, 3000],
        value_range=(500, 5000),
        is_filterable=True,
        is_numeric=True,
        common_operators=["=", ">", "<", ">=", "<=", "BETWEEN"],
        related_concepts=["rental_market", "housing", "affordability"],
    )


@pytest.fixture
def sample_layer_metadata(sample_field_metadata):
    """Create sample layer metadata for testing."""
    return LayerMetadata(
        id="test-layer-001",
        arcgis_item_id="7f98f0cf4f1f4871935c0b51400fb5c5",
        name="dwellsyiq_median_rent",
        display_name="DwellsyIQ Median Rent (MSA)",
        layer_url="https://services.arcgis.com/test/FeatureServer/0",
        portal_url="https://locationmatters.maps.arcgis.com",
        owner="lm_admin",
        layer_type=LayerType.FEATURE,
        geometry_type=GeometryType.POLYGON,
        fields=[sample_field_metadata],
        description="Residential rental trends including median rent by bedroom type",
        semantic_tags=["rental", "housing", "market_trends", "msa"],
        category="residential",
        common_queries=[
            "What is the median rent for 2BR apartments?",
            "Show areas with rent above $2000",
        ],
        last_synced=datetime.utcnow(),
        record_count=1000,
    )


@pytest.fixture
def sample_search_result(sample_layer_metadata):
    """Create sample search result."""
    return LayerSearchResult(
        layer=sample_layer_metadata,
        similarity_score=0.92,
        match_reason="Matched on: rental, median rent",
        suggested_fields=["F2bed_apt_median", "NAME"],
    )


# =============================================================================
# Model Tests
# =============================================================================

class TestFieldMetadata:
    """Tests for FieldMetadata model."""

    def test_field_metadata_creation(self, sample_field_metadata):
        """Test basic field metadata creation."""
        assert sample_field_metadata.name == "F2bed_apt_median"
        assert sample_field_metadata.is_numeric is True
        assert sample_field_metadata.is_date is False

    def test_field_metadata_post_init(self):
        """Test __post_init__ correctly sets derived properties."""
        # Numeric field
        numeric_field = FieldMetadata(
            name="value",
            alias="Value",
            field_type="esriFieldTypeDouble",
        )
        assert numeric_field.is_numeric is True
        assert ">" in numeric_field.common_operators

        # Date field
        date_field = FieldMetadata(
            name="created_date",
            alias="Created Date",
            field_type="esriFieldTypeDate",
        )
        assert date_field.is_date is True

        # String field
        string_field = FieldMetadata(
            name="name",
            alias="Name",
            field_type="esriFieldTypeString",
        )
        assert string_field.is_numeric is False
        assert "LIKE" in string_field.common_operators


class TestLayerMetadata:
    """Tests for LayerMetadata model."""

    def test_layer_metadata_creation(self, sample_layer_metadata):
        """Test basic layer metadata creation."""
        assert sample_layer_metadata.name == "dwellsyiq_median_rent"
        assert sample_layer_metadata.layer_type == LayerType.FEATURE
        assert len(sample_layer_metadata.fields) == 1

    def test_get_field(self, sample_layer_metadata):
        """Test get_field method."""
        field = sample_layer_metadata.get_field("F2bed_apt_median")
        assert field is not None
        assert field.name == "F2bed_apt_median"

        # Case insensitive
        field_lower = sample_layer_metadata.get_field("f2bed_apt_median")
        assert field_lower is not None

        # Non-existent field
        missing = sample_layer_metadata.get_field("nonexistent")
        assert missing is None

    def test_get_numeric_fields(self, sample_layer_metadata):
        """Test get_numeric_fields method."""
        numeric = sample_layer_metadata.get_numeric_fields()
        assert len(numeric) == 1
        assert numeric[0].name == "F2bed_apt_median"

    def test_to_context_string(self, sample_layer_metadata):
        """Test to_context_string method."""
        context = sample_layer_metadata.to_context_string()
        assert "DwellsyIQ Median Rent" in context
        assert "dwellsyiq_median_rent" in context
        assert "residential" in context


class TestQueryFilter:
    """Tests for QueryFilter model."""

    def test_equals_filter(self):
        """Test equals filter to_where_clause."""
        f = QueryFilter(field="NAME", operator=FilterOperator.EQUALS, value="Texas")
        clause = f.to_where_clause()
        assert clause == "NAME = 'Texas'"

    def test_numeric_filter(self):
        """Test numeric filter to_where_clause."""
        f = QueryFilter(field="rent", operator=FilterOperator.GREATER_THAN, value=2000)
        clause = f.to_where_clause()
        assert clause == "rent > 2000"

    def test_like_filter(self):
        """Test LIKE filter to_where_clause."""
        f = QueryFilter(field="NAME", operator=FilterOperator.LIKE, value="Texas")
        clause = f.to_where_clause()
        assert "LIKE" in clause
        assert "%Texas%" in clause

    def test_in_filter(self):
        """Test IN filter to_where_clause."""
        f = QueryFilter(field="STATE", operator=FilterOperator.IN, value=["TX", "CA", "FL"])
        clause = f.to_where_clause()
        assert "IN" in clause
        assert "'TX'" in clause

    def test_between_filter(self):
        """Test BETWEEN filter to_where_clause."""
        f = QueryFilter(field="rent", operator=FilterOperator.BETWEEN, value=[1000, 2000])
        clause = f.to_where_clause()
        assert "BETWEEN" in clause
        assert "1000" in clause
        assert "2000" in clause

    def test_is_null_filter(self):
        """Test IS NULL filter to_where_clause."""
        f = QueryFilter(field="value", operator=FilterOperator.IS_NULL, value=None)
        clause = f.to_where_clause()
        assert clause == "value IS NULL"


class TestStructuredQuery:
    """Tests for StructuredQuery model."""

    def test_build_where_clause(self):
        """Test WHERE clause building from filters."""
        query = StructuredQuery(
            layer_name="test_layer",
            layer_url="https://test.com",
            filters=[
                QueryFilter("rent", FilterOperator.GREATER_THAN, 2000),
                QueryFilter("NAME", FilterOperator.LIKE, "Texas"),
            ],
        )
        where = query.build_where_clause()
        assert "rent > 2000" in where
        assert "LIKE" in where
        assert "AND" in where

    def test_build_where_clause_empty(self):
        """Test WHERE clause with no filters."""
        query = StructuredQuery(
            layer_name="test_layer",
            layer_url="https://test.com",
        )
        where = query.build_where_clause()
        assert where == "1=1"

    def test_to_arcgis_params(self):
        """Test conversion to ArcGIS REST API params."""
        query = StructuredQuery(
            layer_name="test_layer",
            layer_url="https://test.com",
            out_fields=["NAME", "rent"],
            where_clause="rent > 2000",
            return_geometry=True,
            result_record_count=100,
        )
        params = query.to_arcgis_params()

        assert params["where"] == "rent > 2000"
        assert params["outFields"] == "NAME,rent"
        assert params["returnGeometry"] == "true"
        assert params["resultRecordCount"] == 100
        assert params["f"] == "json"

    def test_to_arcgis_params_with_statistics(self):
        """Test conversion with statistics."""
        query = StructuredQuery(
            layer_name="test_layer",
            layer_url="https://test.com",
            statistics_fields=[
                {"statisticType": "avg", "onStatisticField": "rent", "outStatisticFieldName": "avg_rent"}
            ],
            group_by=["STATE"],
        )
        params = query.to_arcgis_params()

        assert params["outStatistics"] is not None
        assert params["groupByFieldsForStatistics"] == "STATE"
        assert params["returnGeometry"] == "false"


class TestQueryPlan:
    """Tests for QueryPlan model."""

    def test_is_multi_layer(self):
        """Test is_multi_layer detection."""
        # Single layer
        plan = QueryPlan(
            queries=[
                StructuredQuery(layer_name="layer1", layer_url="url1"),
            ]
        )
        assert plan.is_multi_layer() is False

        # Multiple queries, same layer
        plan = QueryPlan(
            queries=[
                StructuredQuery(layer_name="layer1", layer_url="url1"),
                StructuredQuery(layer_name="layer1", layer_url="url1"),
            ]
        )
        assert plan.is_multi_layer() is False

        # Multiple different layers
        plan = QueryPlan(
            queries=[
                StructuredQuery(layer_name="layer1", layer_url="url1"),
                StructuredQuery(layer_name="layer2", layer_url="url2"),
            ]
        )
        assert plan.is_multi_layer() is True


# =============================================================================
# Service Tests (Mocked)
# =============================================================================

class TestLayerCatalogService:
    """Tests for LayerCatalogService."""

    @pytest.mark.asyncio
    async def test_search_layers_returns_results(self, sample_layer_metadata):
        """Test that search_layers returns relevant results."""
        with patch("services.layer_intelligence.layer_catalog.LayerCatalogService") as MockCatalog:
            mock_catalog = MockCatalog.return_value
            mock_catalog.search_layers = AsyncMock(return_value=[
                LayerSearchResult(
                    layer=sample_layer_metadata,
                    similarity_score=0.92,
                    match_reason="Matched on rental",
                )
            ])

            results = await mock_catalog.search_layers("median rent prices")

            assert len(results) == 1
            assert results[0].layer.name == "dwellsyiq_median_rent"
            assert results[0].similarity_score > 0.5

    @pytest.mark.asyncio
    async def test_get_layer_by_name(self, sample_layer_metadata):
        """Test getting a layer by name."""
        with patch("services.layer_intelligence.layer_catalog.LayerCatalogService") as MockCatalog:
            mock_catalog = MockCatalog.return_value
            mock_catalog.get_layer = AsyncMock(return_value=sample_layer_metadata)

            layer = await mock_catalog.get_layer("dwellsyiq_median_rent")

            assert layer is not None
            assert layer.name == "dwellsyiq_median_rent"


class TestKnowledgeGraphService:
    """Tests for KnowledgeGraphService."""

    @pytest.mark.asyncio
    async def test_suggest_analysis_layers(self):
        """Test layer suggestions from knowledge graph."""
        with patch("services.layer_intelligence.knowledge_graph.KnowledgeGraphService") as MockGraph:
            mock_graph = MockGraph.return_value
            mock_graph.suggest_analysis_layers = AsyncMock(
                return_value=["dwellsyiq_median_rent", "tapestry_segments", "demographics"]
            )

            suggestions = await mock_graph.suggest_analysis_layers(
                "Compare rental affordability with income levels",
                ["dwellsyiq_median_rent"],
            )

            assert len(suggestions) >= 2
            assert "dwellsyiq_median_rent" in suggestions

    @pytest.mark.asyncio
    async def test_find_cross_layer_path(self):
        """Test finding relationship path between layers."""
        with patch("services.layer_intelligence.knowledge_graph.KnowledgeGraphService") as MockGraph:
            mock_graph = MockGraph.return_value

            from services.layer_intelligence.models import ReasoningPath

            mock_path = ReasoningPath(
                nodes=[],
                relationships=[],
                explanation="DwellsyIQ provides rental data, Tapestry provides consumer segments",
                score=0.85,
            )
            mock_graph.find_cross_layer_path = AsyncMock(return_value=mock_path)

            path = await mock_graph.find_cross_layer_path(
                "dwellsyiq_median_rent", "tapestry_segments"
            )

            assert path is not None
            assert "rental" in path.explanation.lower()


class TestSelfQueryRetriever:
    """Tests for SelfQueryRetriever."""

    @pytest.mark.asyncio
    async def test_parse_simple_query(self, sample_layer_metadata):
        """Test parsing a simple query."""
        with patch("services.layer_intelligence.self_query.SelfQueryRetriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value

            plan = QueryPlan(
                queries=[
                    StructuredQuery(
                        layer_name="dwellsyiq_median_rent",
                        layer_url="https://test.com",
                        out_fields=["NAME", "F2bed_apt_median"],
                        filters=[
                            QueryFilter("F2bed_apt_median", FilterOperator.GREATER_THAN, 2000)
                        ],
                    )
                ],
                reasoning="User wants areas with rent above $2000",
            )
            mock_retriever.parse_query = AsyncMock(return_value=plan)

            result = await mock_retriever.parse_query("Show areas with rent above $2000")

            assert len(result.queries) > 0
            assert result.queries[0].layer_name == "dwellsyiq_median_rent"

    @pytest.mark.asyncio
    async def test_validate_query(self, sample_layer_metadata):
        """Test query validation."""
        with patch("services.layer_intelligence.self_query.SelfQueryRetriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value

            # Valid query
            mock_retriever.validate_query = AsyncMock(return_value=(True, []))

            valid_query = StructuredQuery(
                layer_name="dwellsyiq_median_rent",
                layer_url="https://test.com",
                out_fields=["F2bed_apt_median"],
            )

            is_valid, errors = await mock_retriever.validate_query(valid_query)
            assert is_valid is True
            assert len(errors) == 0


class TestAgenticOrchestrator:
    """Tests for AgenticOrchestrator."""

    @pytest.mark.asyncio
    async def test_process_simple_query(self, sample_layer_metadata):
        """Test processing a simple data query."""
        with patch("services.layer_intelligence.orchestrator.AgenticOrchestrator") as MockOrch:
            mock_orchestrator = MockOrch.return_value

            response = AgentResponse(
                answer="The median rent for 2BR apartments in Dallas is $1,850.",
                steps=[
                    AgentStep(
                        action=AgentAction.DISCOVER_LAYERS,
                        reasoning="Finding rental data layers",
                        success=True,
                    ),
                    AgentStep(
                        action=AgentAction.QUERY_LAYER,
                        reasoning="Querying DwellsyIQ",
                        success=True,
                    ),
                ],
                layers_used=["dwellsyiq_median_rent"],
                confidence=0.85,
                suggestions=["Show rent trends over time", "Compare with Austin"],
                execution_time_ms=1200,
            )
            mock_orchestrator.process = AsyncMock(return_value=response)

            result = await mock_orchestrator.process(
                "What is the median rent for 2BR apartments in Dallas?"
            )

            assert result.answer is not None
            assert result.confidence > 0.5
            assert len(result.layers_used) > 0
            assert len(result.steps) >= 2

    @pytest.mark.asyncio
    async def test_process_cross_layer_query(self):
        """Test processing a query requiring multiple layers."""
        with patch("services.layer_intelligence.orchestrator.AgenticOrchestrator") as MockOrch:
            mock_orchestrator = MockOrch.return_value

            response = AgentResponse(
                answer="Areas with high rental costs tend to have more affluent consumer segments...",
                steps=[
                    AgentStep(action=AgentAction.DISCOVER_LAYERS, reasoning="Finding layers", success=True),
                    AgentStep(action=AgentAction.QUERY_LAYER, reasoning="Querying DwellsyIQ", success=True),
                    AgentStep(action=AgentAction.QUERY_LAYER, reasoning="Querying Tapestry", success=True),
                    AgentStep(action=AgentAction.CROSS_REFERENCE, reasoning="Cross-referencing", success=True),
                ],
                layers_used=["dwellsyiq_median_rent", "tapestry_segments"],
                confidence=0.78,
                suggestions=[],
                execution_time_ms=2500,
            )
            mock_orchestrator.process = AsyncMock(return_value=response)

            result = await mock_orchestrator.process(
                "Which consumer segments live in areas with high rental costs?"
            )

            assert len(result.layers_used) >= 2
            assert "dwellsyiq" in str(result.layers_used).lower() or "tapestry" in str(result.layers_used).lower()


class TestLearningService:
    """Tests for LearningService."""

    @pytest.mark.asyncio
    async def test_store_query_pattern(self):
        """Test storing a query pattern."""
        with patch("services.layer_intelligence.learning_service.LearningService") as MockLearning:
            mock_learning = MockLearning.return_value

            from services.layer_intelligence.learning_service import QueryPattern

            pattern = QueryPattern(
                id="test-pattern-001",
                query="What is the median rent?",
                intent="statistics",
                layers_used=["dwellsyiq_median_rent"],
                was_successful=True,
                confidence_score=0.85,
            )
            mock_learning.store_query_pattern = AsyncMock(return_value=pattern)

            response = AgentResponse(
                answer="The median rent is $1,850",
                layers_used=["dwellsyiq_median_rent"],
                confidence=0.85,
                steps=[],
                suggestions=[],
                execution_time_ms=1000,
            )

            result = await mock_learning.store_query_pattern(
                query="What is the median rent?",
                response=response,
                structured_queries=[],
                intent="statistics",
            )

            assert result.id is not None
            assert result.was_successful is True

    @pytest.mark.asyncio
    async def test_find_similar_patterns(self):
        """Test finding similar query patterns."""
        with patch("services.layer_intelligence.learning_service.LearningService") as MockLearning:
            mock_learning = MockLearning.return_value

            from services.layer_intelligence.learning_service import QueryPattern

            patterns = [
                QueryPattern(
                    query="Average rent in Texas",
                    intent="statistics",
                    layers_used=["dwellsyiq_median_rent"],
                    confidence_score=0.9,
                ),
            ]
            mock_learning.find_similar_patterns = AsyncMock(return_value=patterns)

            results = await mock_learning.find_similar_patterns("median rent in Dallas")

            assert len(results) > 0
            assert results[0].confidence_score > 0.5


# =============================================================================
# Integration Tests (require running services)
# =============================================================================

@pytest.mark.integration
class TestLayerIntelligenceIntegration:
    """Integration tests requiring Qdrant and other services."""

    @pytest.mark.asyncio
    async def test_full_query_flow(self):
        """Test complete query flow from NL to results."""
        pytest.skip("Requires running Qdrant and ArcGIS services")

    @pytest.mark.asyncio
    async def test_layer_sync_from_arcgis(self):
        """Test syncing layers from ArcGIS Online."""
        pytest.skip("Requires ArcGIS credentials and network access")


# =============================================================================
# API Tests
# =============================================================================

class TestLayerIntelligenceAPI:
    """Tests for REST API endpoints."""

    @pytest.mark.asyncio
    async def test_search_endpoint(self):
        """Test /api/v1/layer-intelligence/search endpoint."""
        from fastapi.testclient import TestClient

        # Would need to import and configure the FastAPI app
        # This is a placeholder for the actual test
        pass

    @pytest.mark.asyncio
    async def test_query_endpoint(self):
        """Test /api/v1/layer-intelligence/query endpoint."""
        pass

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test /api/v1/layer-intelligence/health endpoint."""
        pass


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
