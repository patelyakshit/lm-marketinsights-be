"""
Layer Intelligence ADK Tools - Dynamic layer discovery and querying.

These tools enable the GIS agent to:
1. Automatically discover relevant layers based on user queries
2. Query layers using natural language
3. Get detailed layer information and schema
4. Perform cross-layer analysis
"""

import json
import logging
from typing import Optional

from google.adk.tools import FunctionTool, ToolContext

logger = logging.getLogger(__name__)


async def discover_data_layers(
    query: str,
    limit: int = 5,
) -> str:
    """
    Discover relevant GIS data layers based on a natural language query.

    Use this tool FIRST when you need to find which data layers are available
    for answering a user's question about geographic or market data.

    This searches through all indexed layers in the catalog using semantic
    similarity to find the most relevant matches.

    Args:
        query: Natural language description of what data you're looking for
               (e.g., "median rent prices", "demographics", "commercial properties")
        limit: Maximum number of layers to return (default: 5)

    Returns:
        JSON string with matching layers including:
        - name: Unique layer identifier
        - display_name: Human-readable name
        - description: What the layer contains
        - category: Data category (demographics, residential, commercial, etc.)
        - relevance_score: How well this matches the query (0-1)
        - url: Layer service URL
        - fields: Available data fields

    Examples:
        discover_data_layers("rental market data")
        discover_data_layers("population demographics by ZIP code")
        discover_data_layers("commercial real estate inventory")
    """
    try:
        from services.layer_intelligence import LayerCatalogService
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()
        results = await catalog.search_layers(query, limit=limit)

        if not results:
            return json.dumps({
                "success": False,
                "message": "No matching layers found. Try different search terms.",
                "layers": [],
            })

        layers_data = []
        for r in results:
            layer = r.layer
            layers_data.append({
                "name": layer.name,
                "display_name": layer.display_name,
                "description": layer.description,
                "category": layer.category,
                "relevance_score": round(r.similarity_score, 3),
                "url": layer.layer_url,
                "geometry_type": layer.geometry_type.value if layer.geometry_type else None,
                "fields": [
                    {
                        "name": f.name,
                        "alias": f.alias or f.name,
                        "type": "numeric" if f.is_numeric else ("date" if f.is_date else "text"),
                        "description": f.semantic_description,
                    }
                    for f in layer.fields[:15]  # Limit to key fields
                    if f.name.lower() not in ["objectid", "shape", "shape_length", "shape_area", "globalid", "fid"]
                ],
                "suggested_queries": layer.common_queries[:3] if layer.common_queries else [],
            })

        return json.dumps({
            "success": True,
            "total_found": len(results),
            "layers": layers_data,
        }, indent=2)

    except ImportError as e:
        logger.error(f"Layer Intelligence module not available: {e}")
        return json.dumps({
            "success": False,
            "error": "Layer Intelligence system not initialized. Please run layer sync first.",
        })
    except Exception as e:
        logger.error(f"Error discovering layers: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def query_layer_with_natural_language(
    query: str,
    layer_name: Optional[str] = None,
) -> str:
    """
    Query a GIS layer using natural language and get analyzed results.

    This tool automatically:
    1. Identifies the most relevant layer if not specified
    2. Converts your question to a structured ArcGIS query
    3. Executes the query and analyzes results
    4. Returns insights in natural language

    Use this for complex questions that require understanding the data
    and providing analysis, not just raw data retrieval.

    Args:
        query: Natural language question about geographic/market data
               (e.g., "What's the average rent in Miami?",
                "Show areas with population over 50,000")
        layer_name: Optional specific layer to query. If not provided,
                   the system will automatically find the best match.

    Returns:
        JSON string with:
        - answer: Natural language response to the query
        - layers_used: Which layers were queried
        - confidence: How confident the system is in the answer (0-1)
        - suggestions: Follow-up queries you might ask
        - data_summary: Key statistics from the results

    Examples:
        query_layer_with_natural_language("What is the median rent in Texas MSAs?")
        query_layer_with_natural_language("Compare population density across counties")
        query_layer_with_natural_language("Show rent trends over $2000", "dwellsyiq_median_rent")
    """
    try:
        from services.layer_intelligence.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        response = await orchestrator.process(query)

        # Format response for agent consumption
        result = {
            "success": True,
            "answer": response.answer,
            "layers_used": response.layers_used,
            "confidence": round(response.confidence, 2),
            "suggestions": response.suggestions,
            "execution_time_ms": response.execution_time_ms,
        }

        # Add step summary for transparency
        if response.steps:
            result["steps_taken"] = [
                {
                    "action": step.action.value,
                    "success": step.success,
                }
                for step in response.steps
            ]

        return json.dumps(result, indent=2)

    except ImportError as e:
        logger.error(f"Layer Intelligence module not available: {e}")
        return json.dumps({
            "success": False,
            "error": "Layer Intelligence system not initialized.",
        })
    except Exception as e:
        logger.error(f"Error in natural language query: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def get_layer_details(layer_name: str) -> str:
    """
    Get detailed information about a specific data layer.

    Use this to understand:
    - What data a layer contains
    - What fields are available for filtering
    - What questions you can answer with this layer
    - Related layers for cross-analysis

    Args:
        layer_name: The unique layer identifier (from discover_data_layers results)

    Returns:
        JSON string with complete layer metadata including:
        - Full description and category
        - All available fields with types and descriptions
        - Common query patterns
        - Related/complementary layers
        - Data freshness and record count

    Examples:
        get_layer_details("dwellsyiq_median_rent")
        get_layer_details("census_demographics_2023")
    """
    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()
        layer = await catalog.get_layer(layer_name)

        if not layer:
            return json.dumps({
                "success": False,
                "error": f"Layer '{layer_name}' not found in catalog",
            })

        return json.dumps({
            "success": True,
            "layer": {
                "name": layer.name,
                "display_name": layer.display_name,
                "description": layer.description,
                "category": layer.category,
                "layer_url": layer.layer_url,
                "geometry_type": layer.geometry_type.value if layer.geometry_type else None,
                "record_count": layer.record_count,
                "last_synced": layer.last_synced.isoformat() if layer.last_synced else None,
                "fields": [
                    {
                        "name": f.name,
                        "alias": f.alias,
                        "type": f.field_type,
                        "description": f.semantic_description,
                        "is_filterable": f.is_filterable,
                        "is_numeric": f.is_numeric,
                        "is_date": f.is_date,
                        "sample_values": f.sample_values[:5] if f.sample_values else [],
                        "value_range": f.value_range,
                        "common_operators": f.common_operators,
                    }
                    for f in layer.fields
                ],
                "common_queries": layer.common_queries,
                "related_layers": layer.related_layers,
                "complements": layer.complements,
                "semantic_tags": layer.semantic_tags,
            },
        }, indent=2)

    except ImportError:
        return json.dumps({
            "success": False,
            "error": "Layer Intelligence system not initialized.",
        })
    except Exception as e:
        logger.error(f"Error getting layer details: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def find_related_layers(
    layer_name: str,
    query_context: Optional[str] = None,
) -> str:
    """
    Find layers that are related to or complement a given layer.

    Use this to discover additional data sources that could enhance
    your analysis by providing complementary information.

    Args:
        layer_name: The layer to find relationships for
        query_context: Optional context about what you're analyzing
                      (helps find more relevant related layers)

    Returns:
        JSON string with related layers and relationship types:
        - complements: Layers with complementary data
        - correlates: Layers with correlated metrics
        - derived_from: Parent/source layers
        - contains: Sublayers or grouped data

    Examples:
        find_related_layers("dwellsyiq_median_rent", "housing affordability")
        find_related_layers("census_population", "demographic analysis")
    """
    try:
        from services.layer_intelligence.knowledge_graph import get_knowledge_graph_service_sync

        graph = get_knowledge_graph_service_sync()

        # Get suggested layers from knowledge graph
        suggested = await graph.suggest_analysis_layers(
            query_context or f"analysis related to {layer_name}",
            [layer_name],
        )

        # Get relationship explanations
        relationships = []
        for related in suggested:
            if related != layer_name:
                path = await graph.find_cross_layer_path(layer_name, related)
                relationships.append({
                    "layer": related,
                    "relationship": path.explanation if path else "May provide complementary data",
                })

        return json.dumps({
            "success": True,
            "source_layer": layer_name,
            "related_layers": relationships,
            "total_found": len(relationships),
        }, indent=2)

    except ImportError:
        return json.dumps({
            "success": False,
            "error": "Layer Intelligence system not initialized.",
        })
    except Exception as e:
        logger.error(f"Error finding related layers: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def suggest_queries_for_layer(
    layer_name: str,
    limit: int = 5,
) -> str:
    """
    Get suggested natural language queries for a specific layer.

    Use this to understand what questions can be answered with a layer
    and to provide suggestions to users about available analysis options.

    Args:
        layer_name: The layer to generate suggestions for
        limit: Maximum number of suggestions (default: 5)

    Returns:
        JSON string with suggested natural language queries that
        can be answered using this layer's data.

    Examples:
        suggest_queries_for_layer("dwellsyiq_median_rent")
        suggest_queries_for_layer("tapestry_segments", 3)
    """
    try:
        from services.layer_intelligence.self_query import get_self_query_retriever

        retriever = get_self_query_retriever()
        suggestions = await retriever.suggest_queries(layer_name, limit)

        return json.dumps({
            "success": True,
            "layer": layer_name,
            "suggested_queries": suggestions,
        }, indent=2)

    except ImportError:
        return json.dumps({
            "success": False,
            "error": "Layer Intelligence system not initialized.",
        })
    except Exception as e:
        logger.error(f"Error generating query suggestions: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def execute_structured_query(
    layer_name: str,
    layer_url: str,
    where_clause: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = True,
    result_limit: int = 100,
) -> str:
    """
    Execute a structured ArcGIS query directly on a layer.

    Use this when you have specific filter criteria and know the exact
    layer and fields you want to query. For natural language queries,
    use query_layer_with_natural_language instead.

    Args:
        layer_name: Name of the layer for reference
        layer_url: Full URL to the layer's REST endpoint
        where_clause: SQL WHERE clause (e.g., "MEDIAN_RENT > 2000")
        out_fields: Comma-separated field names or "*" for all
        return_geometry: Whether to return feature geometries
        result_limit: Maximum records to return (default: 100)

    Returns:
        JSON string with query results including features and statistics

    Examples:
        execute_structured_query(
            "dwellsyiq_median_rent",
            "https://services.arcgis.com/.../FeatureServer/0",
            "F2bed_apt_median > 2000",
            "NAME,F2bed_apt_median",
            False,
            50
        )
    """
    try:
        from services.layer_intelligence.self_query import QueryExecutor
        from services.layer_intelligence.models import StructuredQuery, QueryOperation
        from decouple import config as env_config

        # Build structured query
        query = StructuredQuery(
            layer_name=layer_name,
            layer_url=layer_url,
            operation=QueryOperation.QUERY,
            out_fields=out_fields.split(",") if out_fields != "*" else ["*"],
            where_clause=where_clause,
            return_geometry=return_geometry,
            result_record_count=result_limit,
        )

        # Execute
        token = env_config("ARCGIS_API_KEY", default="")
        executor = QueryExecutor(arcgis_token=token if token else None)
        result = await executor.execute(query)

        return json.dumps({
            "success": True,
            "layer": layer_name,
            "feature_count": len(result.get("features", [])),
            "exceeded_limit": result.get("exceededTransferLimit", False),
            "features": result.get("features", [])[:50],  # Limit response size
            "fields": [f.get("name") for f in result.get("fields", [])],
        }, indent=2)

    except Exception as e:
        logger.error(f"Error executing structured query: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
        })


# =============================================================================
# Create ADK FunctionTool wrappers
# =============================================================================

discover_data_layers_tool = FunctionTool(discover_data_layers)
query_layer_natural_language_tool = FunctionTool(query_layer_with_natural_language)
get_layer_details_tool = FunctionTool(get_layer_details)
find_related_layers_tool = FunctionTool(find_related_layers)
suggest_queries_tool = FunctionTool(suggest_queries_for_layer)
execute_structured_query_tool = FunctionTool(execute_structured_query)


# Export list for easy integration
LAYER_INTELLIGENCE_TOOLS = [
    discover_data_layers_tool,
    query_layer_natural_language_tool,
    get_layer_details_tool,
    find_related_layers_tool,
    suggest_queries_tool,
    execute_structured_query_tool,
]
