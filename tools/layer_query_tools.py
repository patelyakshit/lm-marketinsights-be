"""
Layer Query Tools

Tools for querying feature data from layers on the map.
These tools communicate with the frontend to query actual layer data.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# Store pending layer query requests
_pending_queries: Dict[str, asyncio.Future] = {}

# WebSocket manager reference (set during initialization)
_ws_manager = None
_connection_id = None


def set_ws_context(manager, connection_id: str):
    """Set WebSocket context for layer queries."""
    global _ws_manager, _connection_id
    _ws_manager = manager
    _connection_id = connection_id


@dataclass
class LayerQueryResult:
    """Result from a layer query."""
    success: bool
    features: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    error: Optional[str] = None
    layer_name: Optional[str] = None


async def _send_layer_query_request(
    request_type: str,
    payload: Dict[str, Any],
    timeout: float = 30.0
) -> LayerQueryResult:
    """
    Send a layer query request to the frontend and wait for response.

    Args:
        request_type: Type of query request
        payload: Query parameters
        timeout: Timeout in seconds

    Returns:
        LayerQueryResult with features or error
    """
    global _ws_manager, _connection_id, _pending_queries

    if not _ws_manager or not _connection_id:
        return LayerQueryResult(
            success=False,
            error="WebSocket context not set"
        )

    request_id = str(uuid.uuid4())

    # Create future to wait for response
    future = asyncio.get_event_loop().create_future()
    _pending_queries[request_id] = future

    try:
        # Send request to frontend
        await _ws_manager.send_message(_connection_id, {
            "type": request_type,
            "payload": {
                "request_id": request_id,
                **payload
            }
        })

        # Wait for response with timeout
        response = await asyncio.wait_for(future, timeout=timeout)

        return LayerQueryResult(
            success=True,
            features=response.get("features", []),
            total_count=response.get("total_count", len(response.get("features", []))),
            layer_name=payload.get("layer_name")
        )

    except asyncio.TimeoutError:
        return LayerQueryResult(
            success=False,
            error=f"Layer query timed out after {timeout} seconds"
        )
    except Exception as e:
        logger.error(f"Layer query error: {e}")
        return LayerQueryResult(
            success=False,
            error=str(e)
        )
    finally:
        _pending_queries.pop(request_id, None)


def handle_layer_query_response(response: Dict[str, Any]):
    """
    Handle layer query response from frontend.
    Called when LAYER/QUERY_RESPONSE message is received.
    """
    request_id = response.get("request_id")
    if request_id and request_id in _pending_queries:
        future = _pending_queries[request_id]
        if not future.done():
            future.set_result(response)


# =============================================================================
# LAYER QUERY TOOLS
# =============================================================================

async def query_layer_features(
    layer_name: str,
    geometry: Optional[Dict] = None,
    where_clause: Optional[str] = None,
    out_fields: Optional[List[str]] = None,
    return_geometry: bool = False,
    max_features: int = 1000
) -> str:
    """
    Query features from a layer on the map.

    This tool queries actual feature data from layers currently on the map,
    enabling analysis based on real GIS data.

    Args:
        layer_name: Name of the layer to query (e.g., "Tapestry Segmentation 2025")
        geometry: Optional GeoJSON geometry for spatial query
        where_clause: Optional SQL where clause for attribute filtering
        out_fields: List of fields to return (default: all)
        return_geometry: Whether to return feature geometries
        max_features: Maximum number of features to return

    Returns:
        JSON string with query results

    Example:
        # Query all Tapestry segments in current view
        result = await query_layer_features("Tapestry Segmentation 2025")

        # Query with spatial filter
        result = await query_layer_features(
            "Tapestry Segmentation 2025",
            geometry={"type": "Polygon", "coordinates": [...]},
            out_fields=["TSEGCODE", "TSEGNAME", "THHBASE"]
        )
    """
    result = await _send_layer_query_request(
        "LAYER/QUERY_REQUEST",
        {
            "layer_name": layer_name,
            "geometry": geometry,
            "where_clause": where_clause,
            "out_fields": out_fields or ["*"],
            "return_geometry": return_geometry,
            "max_features": max_features
        }
    )

    if not result.success:
        return f"Error querying layer: {result.error}"

    if not result.features:
        return f"No features found in layer '{layer_name}'"

    # Format response
    import json
    return json.dumps({
        "layer_name": layer_name,
        "feature_count": result.total_count,
        "features": result.features[:50]  # Limit for response size
    }, indent=2)


async def query_layer_in_extent(
    layer_name: str,
    out_fields: Optional[List[str]] = None
) -> str:
    """
    Query all features from a layer within the current map extent.

    Args:
        layer_name: Name of the layer to query
        out_fields: List of fields to return

    Returns:
        JSON string with features in current extent

    Example:
        result = await query_layer_in_extent("Tapestry Segmentation 2025")
    """
    result = await _send_layer_query_request(
        "LAYER/QUERY_EXTENT",
        {
            "layer_name": layer_name,
            "out_fields": out_fields or ["*"]
        }
    )

    if not result.success:
        return f"Error querying layer in extent: {result.error}"

    import json
    return json.dumps({
        "layer_name": layer_name,
        "feature_count": result.total_count,
        "features": result.features
    }, indent=2)


async def get_layer_statistics(
    layer_name: str,
    stat_field: str,
    group_by: Optional[str] = None,
    geometry: Optional[Dict] = None
) -> str:
    """
    Get aggregated statistics from a layer.

    Args:
        layer_name: Name of the layer
        stat_field: Field to calculate statistics on
        group_by: Optional field to group results by
        geometry: Optional geometry for spatial filtering

    Returns:
        JSON string with statistics

    Example:
        # Get total households by segment
        result = await get_layer_statistics(
            "Tapestry Segmentation 2025",
            stat_field="THHBASE",
            group_by="TSEGCODE"
        )
    """
    result = await _send_layer_query_request(
        "LAYER/STATISTICS_REQUEST",
        {
            "layer_name": layer_name,
            "stat_field": stat_field,
            "group_by": group_by,
            "geometry": geometry
        }
    )

    if not result.success:
        return f"Error getting statistics: {result.error}"

    import json
    return json.dumps(result.features, indent=2)


async def ensure_layer_visible(
    layer_name: str,
    visible: bool = True
) -> str:
    """
    Ensure a layer is visible (or hidden) on the map.

    Args:
        layer_name: Name of the layer
        visible: Whether the layer should be visible

    Returns:
        Status message

    Example:
        await ensure_layer_visible("Tapestry Segmentation 2025", True)
    """
    global _ws_manager, _connection_id

    if not _ws_manager or not _connection_id:
        return "WebSocket context not set"

    try:
        # Note: layer_id is not available here, so we pass layer_name as layerId
        # The frontend will try to match by title if ID lookup fails
        await _ws_manager.send_message(_connection_id, {
            "type": "CHAT/OPERATION_DATA",
            "payload": {
                "operations": [{
                    "type": "TOGGLE_LAYER_VISIBILITY",
                    "payload": {
                        "layerId": layer_name,  # Use layer_name as fallback ID
                        "layerName": layer_name,
                        "visible": visible
                    }
                }]
            }
        })

        action = "shown" if visible else "hidden"
        return f"Layer '{layer_name}' has been {action}"

    except Exception as e:
        return f"Error toggling layer visibility: {e}"


async def get_visible_layers() -> str:
    """
    Get list of all layers currently on the map with their visibility status.

    Returns:
        JSON string with layer information
    """
    result = await _send_layer_query_request(
        "LAYER/LIST_REQUEST",
        {}
    )

    if not result.success:
        return f"Error getting layers: {result.error}"

    import json
    return json.dumps(result.features, indent=2)


# =============================================================================
# SPATIAL ANALYTICS TOOLS
# =============================================================================

def aggregate_segment_data(
    features: List[Dict],
    group_field: str = "TSEGCODE",
    value_field: str = "THHBASE",
    include_names: bool = True
) -> Dict[str, Any]:
    """
    Aggregate segment data from features.

    Args:
        features: List of feature dictionaries
        group_field: Field to group by (default: TSEGCODE)
        value_field: Field to sum (default: THHBASE for households)
        include_names: Include segment names in output

    Returns:
        Dictionary with aggregated statistics
    """
    aggregated = {}
    names = {}

    for feature in features:
        attrs = feature.get("attributes", feature)
        group_key = attrs.get(group_field)

        if group_key:
            value = attrs.get(value_field, 0) or 0
            aggregated[group_key] = aggregated.get(group_key, 0) + value

            if include_names:
                name_field = group_field.replace("CODE", "NAME")
                if name_field in attrs:
                    names[group_key] = attrs[name_field]

    total = sum(aggregated.values())

    # Calculate percentages
    results = []
    for code, value in aggregated.items():
        pct = (value / total * 100) if total > 0 else 0
        result = {
            "code": code,
            "value": value,
            "percentage": round(pct, 2)
        }
        if code in names:
            result["name"] = names[code]
        results.append(result)

    # Sort by value descending
    results.sort(key=lambda x: x["value"], reverse=True)

    return {
        "total": total,
        "segments": results
    }


def get_top_segments(
    aggregated_data: Dict[str, Any],
    n: int = 5
) -> List[Dict]:
    """
    Get top N segments from aggregated data.

    Args:
        aggregated_data: Output from aggregate_segment_data
        n: Number of top segments to return

    Returns:
        List of top N segments
    """
    segments = aggregated_data.get("segments", [])
    return segments[:n]


def calculate_gap_analysis(
    market_data: Dict[str, Any],
    customer_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate gap analysis between market and customer segments.

    Args:
        market_data: Market segment distribution
        customer_data: Customer segment distribution

    Returns:
        Gap analysis results
    """
    market_segments = {s["code"]: s for s in market_data.get("segments", [])}
    customer_segments = {s["code"]: s for s in customer_data.get("segments", [])}

    all_codes = set(market_segments.keys()) | set(customer_segments.keys())

    results = []
    for code in all_codes:
        market = market_segments.get(code, {})
        customer = customer_segments.get(code, {})

        market_pct = market.get("percentage", 0)
        customer_pct = customer.get("percentage", 0)

        gap = market_pct - customer_pct
        index = (customer_pct / market_pct * 100) if market_pct > 0 else 0

        results.append({
            "code": code,
            "name": market.get("name") or customer.get("name", "Unknown"),
            "market_pct": market_pct,
            "customer_pct": customer_pct,
            "gap": round(gap, 2),
            "index": round(index, 1),
            "opportunity": "Under-penetrated" if gap > 5 else (
                "Over-penetrated" if gap < -5 else "Balanced"
            )
        })

    # Sort by absolute gap
    results.sort(key=lambda x: abs(x["gap"]), reverse=True)

    return {
        "analysis": results,
        "top_opportunities": [r for r in results if r["gap"] > 5][:5],
        "top_saturated": [r for r in results if r["gap"] < -5][:5]
    }


# =============================================================================
# ADK FUNCTION TOOLS
# =============================================================================

query_layer_features_tool = FunctionTool(query_layer_features)
query_layer_in_extent_tool = FunctionTool(query_layer_in_extent)
get_layer_statistics_tool = FunctionTool(get_layer_statistics)
ensure_layer_visible_tool = FunctionTool(ensure_layer_visible)
get_visible_layers_tool = FunctionTool(get_visible_layers)

LAYER_QUERY_TOOLS = [
    query_layer_features_tool,
    query_layer_in_extent_tool,
    get_layer_statistics_tool,
    ensure_layer_visible_tool,
    get_visible_layers_tool,
]
