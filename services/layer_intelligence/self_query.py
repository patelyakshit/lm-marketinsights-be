"""
Self-Query Retriever for Layer Intelligence System.

Converts natural language queries into structured ArcGIS queries
using semantic understanding of layers and fields.
"""

import json
import logging
import re
from typing import Optional

import google.generativeai as genai
from decouple import config as env_config

from .models import (
    FilterOperator,
    LayerMetadata,
    LayerSearchResult,
    QueryFilter,
    QueryOperation,
    QueryPlan,
    StructuredQuery,
)
from .layer_catalog import LayerCatalogService, get_layer_catalog_service_sync
from .knowledge_graph import KnowledgeGraphService, get_knowledge_graph_service_sync

logger = logging.getLogger(__name__)


class SelfQueryRetriever:
    """
    Self-Query Retriever that converts natural language to structured queries.

    This component:
    1. Understands user intent from natural language
    2. Identifies relevant layers using semantic search
    3. Extracts filter conditions and field selections
    4. Generates valid ArcGIS REST API query parameters
    """

    def __init__(
        self,
        catalog_service: Optional[LayerCatalogService] = None,
        graph_service: Optional[KnowledgeGraphService] = None,
        llm_model: str = "gemini-2.0-flash",
    ):
        self.catalog = catalog_service or get_layer_catalog_service_sync()
        self.graph = graph_service or get_knowledge_graph_service_sync()
        self.llm_model = llm_model

        # Initialize Gemini
        api_key = env_config("GOOGLE_API_KEY", default="")
        if api_key:
            genai.configure(api_key=api_key)

        self._model = None

    @property
    def model(self):
        """Lazy load the Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(self.llm_model)
        return self._model

    async def parse_query(
        self,
        natural_query: str,
        context: Optional[dict] = None,
    ) -> QueryPlan:
        """
        Parse a natural language query into a structured query plan.

        Args:
            natural_query: Natural language question
            context: Optional context (location, previous queries, etc.)

        Returns:
            QueryPlan with one or more structured queries
        """
        # Step 1: Identify relevant layers
        layer_results = await self.catalog.search_layers(
            query=natural_query,
            limit=5,
        )

        if not layer_results:
            return QueryPlan(
                queries=[],
                reasoning="No relevant layers found for this query.",
            )

        # Step 2: Use LLM to understand intent and extract structure
        query_structure = await self._extract_query_structure(
            natural_query,
            layer_results,
            context,
        )

        # Step 3: Build structured queries
        queries = self._build_structured_queries(
            query_structure,
            layer_results,
        )

        # Step 4: Check if cross-layer analysis is needed
        if len(queries) > 1:
            # Use knowledge graph to find connections
            layer_names = [q.layer_name for q in queries]
            for i, name1 in enumerate(layer_names):
                for name2 in layer_names[i+1:]:
                    path = await self.graph.find_cross_layer_path(name1, name2)
                    if path:
                        query_structure["cross_layer_reasoning"] = path.explanation

        return QueryPlan(
            queries=queries,
            reasoning=query_structure.get("reasoning", ""),
            requires_post_processing=query_structure.get("needs_aggregation", False),
            post_processing_instructions=query_structure.get("post_processing", None),
        )

    async def _extract_query_structure(
        self,
        natural_query: str,
        layer_results: list[LayerSearchResult],
        context: Optional[dict] = None,
    ) -> dict:
        """Use LLM to extract query structure from natural language."""

        # Build layer context for the LLM
        layer_context = []
        for result in layer_results[:3]:  # Top 3 layers
            layer = result.layer
            fields_info = []
            for f in layer.fields[:15]:  # Limit fields
                if f.name.lower() not in ["objectid", "shape", "shape_length", "shape_area", "globalid"]:
                    field_desc = {
                        "name": f.name,
                        "alias": f.alias,
                        "type": "numeric" if f.is_numeric else ("date" if f.is_date else "text"),
                    }
                    if f.semantic_description:
                        field_desc["description"] = f.semantic_description
                    if f.sample_values:
                        field_desc["examples"] = f.sample_values[:3]
                    fields_info.append(field_desc)

            layer_context.append({
                "name": layer.name,
                "display_name": layer.display_name,
                "description": layer.description,
                "category": layer.category,
                "fields": fields_info,
                "relevance_score": result.similarity_score,
            })

        prompt = f"""You are a GIS query parser. Convert the natural language query into a structured query plan.

USER QUERY: {natural_query}

AVAILABLE LAYERS:
{json.dumps(layer_context, indent=2)}

{f"ADDITIONAL CONTEXT: {json.dumps(context)}" if context else ""}

Analyze the query and return a JSON object with:
{{
    "intent": "query|statistics|comparison|trend|spatial",
    "reasoning": "Brief explanation of how you interpreted the query",
    "primary_layer": "layer_name to query",
    "secondary_layers": ["optional additional layers for comparison"],
    "fields_needed": ["field1", "field2"],
    "filters": [
        {{"field": "field_name", "operator": "=|>|<|>=|<=|LIKE|IN|BETWEEN", "value": "value or [list]"}}
    ],
    "statistics": [
        {{"field": "field_name", "stat_type": "AVG|SUM|COUNT|MIN|MAX"}}
    ],
    "group_by": ["field for grouping"],
    "order_by": "field ASC|DESC",
    "needs_aggregation": true/false,
    "post_processing": "instructions for combining results if needed",
    "spatial_filter": {{
        "type": "point|polygon|envelope",
        "coordinates": [...]
    }}
}}

Important rules:
1. Use exact field names from the layer schema
2. For text filters with partial matching, use LIKE operator
3. For numeric comparisons, ensure the value is a number
4. For IN operator, provide a list of values
5. For BETWEEN operator, provide [min, max] as value
6. Only include statistics if the query asks for aggregations like "average", "total", "count"
7. Include group_by when statistics are needed per category

Return ONLY valid JSON, no markdown or explanation."""

        try:
            response = await self.model.generate_content_async(prompt)
            response_text = response.text.strip()

            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"```json?\n?", "", response_text)
                response_text = response_text.rstrip("`").strip()

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Error extracting query structure: {e}")
            # Fallback to basic query
            if layer_results:
                return {
                    "intent": "query",
                    "reasoning": "Fallback to basic query due to parsing error",
                    "primary_layer": layer_results[0].layer.name,
                    "fields_needed": ["*"],
                    "filters": [],
                }
            return {"intent": "query", "reasoning": "No layers matched", "filters": []}

    def _build_structured_queries(
        self,
        structure: dict,
        layer_results: list[LayerSearchResult],
    ) -> list[StructuredQuery]:
        """Build StructuredQuery objects from parsed structure."""
        queries = []

        # Get layer lookup
        layer_map = {r.layer.name: r.layer for r in layer_results}

        # Primary query
        primary_layer_name = structure.get("primary_layer")
        if primary_layer_name and primary_layer_name in layer_map:
            primary_layer = layer_map[primary_layer_name]

            # Determine operation type
            intent = structure.get("intent", "query")
            operation = self._intent_to_operation(intent)

            # Build filters
            filters = []
            for f in structure.get("filters", []):
                try:
                    operator = self._parse_operator(f.get("operator", "="))
                    filters.append(QueryFilter(
                        field=f["field"],
                        operator=operator,
                        value=f["value"],
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid filter: {f}, error: {e}")

            # Build statistics
            statistics = None
            if structure.get("statistics"):
                statistics = [
                    {
                        "statisticType": s["stat_type"].lower(),
                        "onStatisticField": s["field"],
                        "outStatisticFieldName": f"{s['stat_type'].lower()}_{s['field']}",
                    }
                    for s in structure["statistics"]
                ]

            # Determine fields
            out_fields = structure.get("fields_needed", ["*"])
            if out_fields == ["*"] and not statistics:
                # Get meaningful fields
                out_fields = self._get_meaningful_fields(primary_layer)

            query = StructuredQuery(
                layer_name=primary_layer.name,
                layer_url=primary_layer.layer_url,
                operation=operation,
                out_fields=out_fields,
                filters=filters,
                statistics_fields=statistics,
                group_by=structure.get("group_by"),
                order_by=structure.get("order_by"),
                return_geometry=intent not in ["statistics", "trend"],
            )

            # Handle spatial filter
            spatial = structure.get("spatial_filter")
            if spatial and spatial.get("coordinates"):
                query.geometry = self._build_geometry(spatial)

            queries.append(query)

        # Secondary queries for comparison
        for secondary_name in structure.get("secondary_layers", []):
            if secondary_name in layer_map:
                secondary_layer = layer_map[secondary_name]
                queries.append(StructuredQuery(
                    layer_name=secondary_layer.name,
                    layer_url=secondary_layer.layer_url,
                    operation=QueryOperation.QUERY,
                    out_fields=self._get_meaningful_fields(secondary_layer),
                    return_geometry=False,
                ))

        return queries

    def _intent_to_operation(self, intent: str) -> QueryOperation:
        """Convert intent string to QueryOperation enum."""
        mapping = {
            "query": QueryOperation.QUERY,
            "statistics": QueryOperation.STATISTICS,
            "comparison": QueryOperation.COMPARE,
            "trend": QueryOperation.TREND,
            "spatial": QueryOperation.SPATIAL_QUERY,
            "aggregate": QueryOperation.AGGREGATE,
        }
        return mapping.get(intent, QueryOperation.QUERY)

    def _parse_operator(self, op_str: str) -> FilterOperator:
        """Parse operator string to FilterOperator enum."""
        mapping = {
            "=": FilterOperator.EQUALS,
            "==": FilterOperator.EQUALS,
            "!=": FilterOperator.NOT_EQUALS,
            "<>": FilterOperator.NOT_EQUALS,
            ">": FilterOperator.GREATER_THAN,
            "<": FilterOperator.LESS_THAN,
            ">=": FilterOperator.GREATER_EQUAL,
            "<=": FilterOperator.LESS_EQUAL,
            "LIKE": FilterOperator.LIKE,
            "IN": FilterOperator.IN,
            "NOT IN": FilterOperator.NOT_IN,
            "BETWEEN": FilterOperator.BETWEEN,
            "IS NULL": FilterOperator.IS_NULL,
            "IS NOT NULL": FilterOperator.IS_NOT_NULL,
        }
        return mapping.get(op_str.upper(), FilterOperator.EQUALS)

    def _get_meaningful_fields(self, layer: LayerMetadata) -> list[str]:
        """Get meaningful fields from a layer, excluding system fields."""
        exclude = {"objectid", "shape", "shape_length", "shape_area", "globalid", "fid"}
        fields = []
        for f in layer.fields:
            if f.name.lower() not in exclude:
                fields.append(f.name)
        return fields[:20] if fields else ["*"]  # Limit to 20 fields

    def _build_geometry(self, spatial: dict) -> dict:
        """Build ArcGIS geometry object from spatial filter."""
        geo_type = spatial.get("type", "envelope")
        coords = spatial.get("coordinates", [])

        if geo_type == "point" and len(coords) >= 2:
            return {
                "x": coords[0],
                "y": coords[1],
                "spatialReference": {"wkid": 4326},
            }
        elif geo_type == "envelope" and len(coords) >= 4:
            return {
                "xmin": coords[0],
                "ymin": coords[1],
                "xmax": coords[2],
                "ymax": coords[3],
                "spatialReference": {"wkid": 4326},
            }
        elif geo_type == "polygon" and coords:
            return {
                "rings": [coords],
                "spatialReference": {"wkid": 4326},
            }

        return {}

    async def suggest_queries(
        self,
        layer_name: str,
        limit: int = 5,
    ) -> list[str]:
        """
        Suggest natural language queries for a specific layer.

        Args:
            layer_name: Name of the layer
            limit: Number of suggestions

        Returns:
            List of suggested query strings
        """
        layer = await self.catalog.get_layer(layer_name)
        if not layer:
            return []

        # Return cached suggestions if available
        if layer.common_queries:
            return layer.common_queries[:limit]

        # Generate suggestions using LLM
        fields_desc = [
            f"{f.alias or f.name}: {f.semantic_description}"
            for f in layer.fields[:10]
            if f.name.lower() not in ["objectid", "shape", "shape_length", "shape_area"]
        ]

        prompt = f"""Generate {limit} natural language questions that users might ask about this GIS layer.

Layer: {layer.display_name}
Description: {layer.description}
Category: {layer.category}
Fields:
{chr(10).join(fields_desc)}

Generate diverse questions covering:
1. Basic data retrieval (e.g., "Show me all...")
2. Filtering by specific values
3. Statistical analysis (averages, totals)
4. Geographic queries (near, within)
5. Comparisons and trends

Return ONLY a JSON array of question strings, no explanation."""

        try:
            response = await self.model.generate_content_async(prompt)
            response_text = response.text.strip()

            # Clean up response
            if response_text.startswith("```"):
                response_text = re.sub(r"```json?\n?", "", response_text)
                response_text = response_text.rstrip("`").strip()

            suggestions = json.loads(response_text)
            return suggestions[:limit] if isinstance(suggestions, list) else []

        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []

    async def validate_query(
        self,
        query: StructuredQuery,
    ) -> tuple[bool, list[str]]:
        """
        Validate a structured query before execution.

        Args:
            query: StructuredQuery to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Get layer metadata
        layer = await self.catalog.get_layer(query.layer_name)
        if not layer:
            errors.append(f"Layer '{query.layer_name}' not found in catalog")
            return False, errors

        # Validate fields
        layer_field_names = {f.name.lower() for f in layer.fields}

        for field_name in query.out_fields:
            if field_name != "*" and field_name.lower() not in layer_field_names:
                errors.append(f"Field '{field_name}' not found in layer")

        # Validate filters
        for f in query.filters:
            if f.field.lower() not in layer_field_names:
                errors.append(f"Filter field '{f.field}' not found in layer")
            else:
                # Check operator compatibility with field type
                field_meta = layer.get_field(f.field)
                if field_meta:
                    if f.operator in [FilterOperator.LIKE] and field_meta.is_numeric:
                        errors.append(f"LIKE operator not valid for numeric field '{f.field}'")

        # Validate statistics
        if query.statistics_fields:
            for stat in query.statistics_fields:
                stat_field = stat.get("onStatisticField", "")
                if stat_field.lower() not in layer_field_names:
                    errors.append(f"Statistics field '{stat_field}' not found in layer")

        # Validate group_by
        if query.group_by:
            for group_field in query.group_by:
                if group_field.lower() not in layer_field_names:
                    errors.append(f"Group by field '{group_field}' not found in layer")

        return len(errors) == 0, errors

    async def explain_query(
        self,
        query: StructuredQuery,
    ) -> str:
        """
        Generate a human-readable explanation of what a query does.

        Args:
            query: StructuredQuery to explain

        Returns:
            Natural language explanation
        """
        layer = await self.catalog.get_layer(query.layer_name)
        layer_display = layer.display_name if layer else query.layer_name

        parts = [f"Query the **{layer_display}** layer"]

        # Describe fields
        if query.out_fields and query.out_fields != ["*"]:
            parts.append(f"returning fields: {', '.join(query.out_fields)}")

        # Describe filters
        if query.filters:
            filter_desc = []
            for f in query.filters:
                filter_desc.append(f"{f.field} {f.operator.value} {f.value}")
            parts.append(f"where {' AND '.join(filter_desc)}")

        # Describe statistics
        if query.statistics_fields:
            stats_desc = []
            for s in query.statistics_fields:
                stats_desc.append(f"{s['statisticType']}({s['onStatisticField']})")
            parts.append(f"calculating: {', '.join(stats_desc)}")

        # Describe grouping
        if query.group_by:
            parts.append(f"grouped by: {', '.join(query.group_by)}")

        # Describe ordering
        if query.order_by:
            parts.append(f"ordered by: {query.order_by}")

        # Describe spatial
        if query.geometry:
            parts.append("within the specified geographic area")

        return " ".join(parts)


# =============================================================================
# Query Execution Helper
# =============================================================================

class QueryExecutor:
    """
    Executes structured queries against ArcGIS REST API.

    This is a helper class used by the orchestrator to actually
    run queries and return results.
    """

    def __init__(self, arcgis_token: Optional[str] = None):
        self.token = arcgis_token
        self._session = None

    async def execute(self, query: StructuredQuery) -> dict:
        """
        Execute a structured query and return results.

        Args:
            query: StructuredQuery to execute

        Returns:
            Dict with features and metadata
        """
        import aiohttp

        if not query.layer_url:
            raise ValueError(f"No layer URL for query on {query.layer_name}")

        # Build query URL
        query_url = f"{query.layer_url}/query"
        params = query.to_arcgis_params()

        if self.token:
            params["token"] = self.token

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(query_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Query failed: {error_text}")

                    result = await response.json()

                    if "error" in result:
                        raise Exception(f"ArcGIS error: {result['error']}")

                    return {
                        "features": result.get("features", []),
                        "fields": result.get("fields", []),
                        "geometryType": result.get("geometryType"),
                        "spatialReference": result.get("spatialReference"),
                        "exceededTransferLimit": result.get("exceededTransferLimit", False),
                    }

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    async def execute_plan(self, plan: QueryPlan) -> list[dict]:
        """
        Execute all queries in a plan.

        Args:
            plan: QueryPlan with one or more queries

        Returns:
            List of results for each query
        """
        import asyncio

        results = []

        # Execute queries (could parallelize independent ones)
        for query in plan.queries:
            try:
                result = await self.execute(query)
                results.append({
                    "layer_name": query.layer_name,
                    "success": True,
                    "data": result,
                })
            except Exception as e:
                results.append({
                    "layer_name": query.layer_name,
                    "success": False,
                    "error": str(e),
                })

        return results


# =============================================================================
# Factory Functions
# =============================================================================

_self_query_retriever: Optional[SelfQueryRetriever] = None


def get_self_query_retriever(
    force_new: bool = False,
) -> SelfQueryRetriever:
    """Get or create the global self-query retriever."""
    global _self_query_retriever

    if _self_query_retriever is None or force_new:
        _self_query_retriever = SelfQueryRetriever()

    return _self_query_retriever


def set_self_query_retriever(retriever: SelfQueryRetriever):
    """Set the global self-query retriever."""
    global _self_query_retriever
    _self_query_retriever = retriever
