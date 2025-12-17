"""
GIS ADK Tools - Google ADK FunctionTool wrappers for GIS operations

These tools wrap the core GIS executor methods and return LangGraph-compatible
operation payloads that the frontend can interpret.
"""

import json
import logging
from typing import List, Optional

from google.adk.tools import ToolContext
from tools.gis_tools import gis_executor
from tools.spatial_rag_tools import (
    get_location_intelligence_tool as spatial_rag_location_tool,
    analyze_location_for_marketing_tool,
    compare_locations_tool,
)
from tools.trade_area_tools import TRADE_AREA_TOOLS
from tools.layer_intelligence_tools import LAYER_INTELLIGENCE_TOOLS
from tools.lifestyle_analysis_tools import get_lifestyle_analysis_tool
from utils.semantic_cache import semantic_cache_get, semantic_cache_set


logger = logging.getLogger(__name__)


async def toggle_layer_visibility(layer_id: str, layer_name: str, visible: bool) -> str:
    """
    Show or hide a map layer.

    Args:
        layer_id: Layer identifier from layer_data state
        layer_name: Human-readable layer name
        visible: True to show layer, False to hide

    Returns:
        JSON string with operation payload

    Examples:
        toggle_layer_visibility("layer123", "Population", True)
        toggle_layer_visibility("layer456", "Counties", False)
    """
    try:
        result = await gis_executor.toggle_layer_visibility(
            layer_id, layer_name, visible
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in toggle_layer_visibility: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def toggle_sublayer_visibility(
    layer_id: str, sublayer_id: str, visible: bool
) -> str:
    """
    Show or hide a sublayer within a parent layer.

    Args:
        layer_id: Parent layer identifier
        sublayer_id: Sublayer identifier
        visible: True to show, False to hide

    Returns:
        JSON string with operation payload
    """
    try:
        result = await gis_executor.toggle_sublayer_visibility(
            layer_id, sublayer_id, visible
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in toggle_sublayer_visibility: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def apply_layer_filter(
    layer_id: str, where_clause: str, spatial_lock: bool = False
) -> str:
    """
    Apply attribute filter to a map layer using SQL WHERE clause.

    Args:
        layer_id: Layer identifier
        where_clause: SQL WHERE clause (e.g., "POPULATION > 1500", "city = 'Jacksonville'")
        spatial_lock: If True, lock filter to current map extent

    Returns:
        JSON string with operation payload

    Examples:
        apply_layer_filter("layer123", "POPULATION > 1500")
        apply_layer_filter("layer456", "Segment_ID IN ('7A', '5C')")
        apply_layer_filter("layer789", "city = 'Jacksonville' AND state = 'FL'")
    """
    try:
        result = await gis_executor.apply_filter(layer_id, where_clause, spatial_lock)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in apply_layer_filter: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def remove_layer_filter(layer_id: str) -> str:
    """
    Remove filter from layer to show all features.

    Args:
        layer_id: Layer identifier

    Returns:
        JSON string with operation payload (resets WHERE clause to "1=1")
    """
    try:
        result = await gis_executor.remove_filter(layer_id)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in remove_layer_filter: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def zoom_map(zoom_action: str, zoom_percentage: int = 50) -> str:
    """
    Zoom the map in or out.

    Args:
        zoom_action: "zoom_in" or "zoom_out"
        zoom_percentage: Zoom amount (0-100). Typical values:
            - 10: Small zoom (e.g., "zoom in a little")
            - 50: Medium zoom (default)
            - 100: Large zoom (e.g., "zoom in a lot")

    Returns:
        JSON string with operation payload

    Examples:
        zoom_map("zoom_in", 10)  # Zoom in a little
        zoom_map("zoom_out", 50)  # Zoom out medium amount
    """
    try:
        result = await gis_executor.zoom_map(zoom_action, zoom_percentage)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in zoom_map: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})



async def zoom_to_location(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    wkid: int = 4326
) -> str:
    """
    Zoom map to specific extent with spatial reference

    Args:
        xmin: Minimum x coordinate (longitude) of bounding box
        ymin: Minimum y coordinate (latitude) of bounding box
        xmax: Maximum x coordinate (longitude) of bounding box
        ymax: Maximum y coordinate (latitude) of bounding box
        wkid: Spatial reference well-known ID from geocode results (default: 4326 for WGS84)

    Returns:
        JSON string with operation payload including extent and spatialReference

    Examples:
        zoom_to_location(-124.48, 32.53, -114.13, 42.01)  # California with default wkid
        zoom_to_location(-124.48, 32.53, -114.13, 42.01, 4326)  # With explicit wkid
    """
    try:
        targetExtent = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        spatialReference = {
            "wkid": wkid
        }
        result = await gis_executor.zoom_to_location(targetExtent, spatialReference)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in zoom_to_location: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def pan_map(direction: str, distance: int = 20) -> str:
    """
    Pan the map in a specific direction.

    Args:
        direction: Pan direction - "north", "south", "east", or "west"
                  (aliases: "up", "down", "left", "right")
        distance: Pan distance as percentage of map width/height (0-100)

    Returns:
        JSON string with operation payload

    Examples:
        pan_map("north", 20)  # Pan up
        pan_map("west", 50)   # Pan left 50%
        pan_map("left", 30)   # Alias for west
    """
    try:
        result = await gis_executor.pan_map(direction, distance)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in pan_map: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def add_map_pin(
    address: str, latitude: float, longitude: float, note: str = ""
) -> str:
    """
    Add a pin marker to the map at specific coordinates.

    Note: You should geocode the address first to get coordinates, then call this tool
    with the geocoding results as suggestions.

    Args:
        address: Address string
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        note: Optional note/description for the pin

    Returns:
        JSON string with pin suggestions (SUGGEST_PIN operation)

    Usage:
        1. Call geocode_address to get coordinate suggestions
        2. Parse the top result(s)
        3. Call add_map_pin with each suggestion
    """
    try:
        # Create pin suggestion with UUID
        import uuid

        pins = [
            {
                "id": str(uuid.uuid4()),
                "address": address,
                "latitude": latitude,
                "longitude": longitude,
                "score": 100,
                "note": note,
            }
        ]
        result = await gis_executor.suggest_pin(pins)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in add_map_pin: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def remove_map_pins(pin_ids: List[str]) -> str:
    """
    Remove one or more pin markers from the map.

    Args:
        pin_ids: List of pin IDs to remove
                 Use ["all"] to remove all pins

    Returns:
        JSON string with operation payload

    Examples:
        remove_map_pins(["uuid-1", "uuid-2"])  # Remove specific pins
        remove_map_pins(["all"])                # Remove all pins
    """
    try:
        result = await gis_executor.remove_pin(pin_ids)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in remove_map_pins: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def toggle_layer_labels(
    layer_id: str, enabled: bool, label_field: Optional[str] = None
) -> str:
    """
    Enable or disable labels for a map layer.

    Args:
        layer_id: Layer identifier
        enabled: True to show labels, False to hide
        label_field: Field name to use for labels (e.g., "Name", "City", "Title")
                    If None, the layer's default label field is used

    Returns:
        JSON string with operation payload

    Examples:
        toggle_layer_labels("layer123", True, "City")  # Show city name labels
        toggle_layer_labels("layer456", True, "Name")  # Show name labels
        toggle_layer_labels("layer789", False)         # Hide labels
    """
    try:
        result = await gis_executor.toggle_labels(layer_id, enabled, label_field)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in toggle_layer_labels: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def geocode_address(address: str, max_candidates: int = 5) -> str:
    """
    Convert an address string to geographic coordinates.

    Returns multiple suggestions if address is ambiguous. The results include
    coordinates, match score, and formatted address.

    Args:
        address: Address string to geocode (e.g., "123 Main St, San Francisco, CA")
        max_candidates: Maximum number of location suggestions to return

    Returns:
        JSON string with geocoding suggestions (SUGGEST_PIN format with coordinates)

    Examples:
        geocode_address("1600 Pennsylvania Ave, Washington DC")
        geocode_address("Times Square, New York")
        geocode_address("Golden Gate Bridge")
    """
    try:
        # Check semantic cache for similar address queries
        cache_key = f"geocode:{address.lower().strip()}"
        cached_result, similarity = await semantic_cache_get(cache_key)
        if cached_result and similarity >= 0.92:
            logger.info(f"Semantic cache hit for geocode: {address[:30]}... (similarity: {similarity:.2f})")
            return cached_result

        suggestions = await gis_executor.get_coordinates_suggestions(address, max_candidates)

        if not suggestions:
            return json.dumps(
                {
                    "type": "ERROR",
                    "payload": {"error": "No locations found for address"},
                }
            )

        # Return as SUGGEST_PIN payload
        # result = await gis_executor.suggest_pin(suggestions[:max_candidates])

        result = json.dumps(suggestions, indent=0)

        # Cache the result for similar future queries
        await semantic_cache_set(cache_key, result, {"address": address})

        return result
    except Exception as e:
        logger.error(f"Error in geocode_address: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def reverse_geocode_location(latitude: float, longitude: float) -> str:
    """
    Convert geographic coordinates to a readable address.

    Args:
        latitude: Latitude coordinate (-90 to 90)
        longitude: Longitude coordinate (-180 to 180)

    Returns:
        JSON string with address information

    Examples:
        reverse_geocode_location(34.0522, -118.2437)  # Returns LA address
        reverse_geocode_location(40.7128, -74.0060)   # Returns NYC address
    """
    try:
        result = await gis_executor.reverse_geocode_coordinates(latitude, longitude)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in reverse_geocode_location: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def get_layer_statistics(
    layer_id: str, field_names: List[str], tool_context: ToolContext
) -> str:
    """
    Calculate statistics for numeric fields in a layer.
    Returns count, sum, min, max, avg, and stddev for each field.

    Args:
        layer_id: Layer identifier
        field_names: List of numeric field names to calculate stats for, Exclude database's internal fields such as OBJECTID etc.

    Returns:
        JSON string with statistics data

    Examples:
        get_layer_statistics(
            "layer123",
            ["POPULATION", "INCOME"]
        )
    """

    session = tool_context.session
    map_context = session.state.get("map_context", {})

    layers = map_context.get("layers", [])
    layer = gis_executor.filter_layer_by_id(layers, layer_id)
    if not layer:
        return json.dumps({
            "success": False,
            "error": "Unable to find layer for given name."
        })
    elif not layer.get("visible", False):
        return json.dumps({
            "success": False,
            "error": "Layer is not currently visible on map. Please enable it first"
        })

    current_filter = layer.get("current_filter")

    try:
        statistics = gis_executor.get_fields_statistics(layer, field_names, current_filter, limit=5)
        return json.dumps(statistics, indent=4)
    except Exception as e:
        logger.error(f"Error in get_layer_statistics: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def get_poi_location_intelligence(latitude: float, longitude: float) -> str:
    """
    Fetch Point of Interest items list for given the location(latitude & longitude)

    Returns Top 10 Places details that user's might find helpful.

    Args:
        latitude: Latitude coordinate of the address
        longitude: Longitude coordinate of the address

    Returns:
        POI data list encoded into JSON string.

    Examples:
        get_poi_location_intelligence(34.0522, -118.2437)  # LA area intelligence
        get_poi_location_intelligence(40.7128, -74.0060)   # NYC area intelligence
    """

    try:
        result = await gis_executor.get_esri_poi_data(latitude, longitude)
        result_items = result.get("results", [])
        return json.dumps({
            "latitude": latitude,
            "longitude": longitude,
            "poi_items": result_items
        }, indent=0)
    except Exception as e:
        logger.error(f"Error in get_poi_location_intelligence: {e}")
        return json.dumps({
            "type": "ERROR",
            "payload": {
                "error": str(e)
            }
        })


async def get_tapestry_segmentation_intelligence(latitude: float, longitude: float):
    """
    Fetch Tapestry segmentation details such as lifestyles and consumer behavior for given the location (latitude, longitude).

    Args:
        latitude: Latitude coordinate of the address
        longitude: Longitude coordinate of the address

    Returns:
        Tapestry segmentation JSON string containing fields and feature which can be summarised.

    Examples:
        get_tapestry_segmentation_intelligence(34.0522, -118.2437)  # LA area intelligence
        get_tapestry_segmentation_intelligence(40.7128, -74.0060)   # NYC area intelligence
    """

    try:
        result = await gis_executor.get_esri_geoenrich_data_tapestry(latitude, longitude)
        return json.dumps(result, indent=4)
    except Exception as e:
        logger.error(f"Error in get_location_intelligence: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def get_demographics_location_intelligence(latitude: float, longitude: float):
    """
    Fetch demographics details such as food spending, population total etc. for given the location (latitude, longitude).

    Args:
        latitude: Latitude coordinate of the address
        longitude: Longitude coordinate of the address

    Returns:
        JSON string of fields and features(values) for demographic point such as food spending, population

    Examples:
        get_demographics_location_intelligence(34.0522, -118.2437)  # LA area intelligence
        get_demographics_location_intelligence(40.7128, -74.0060)   # NYC area intelligence
    """

    try:
        result = await gis_executor.get_esri_geoenrich_data_general_info(latitude, longitude)
        return json.dumps(result, indent=4)
    except Exception as e:
        logger.error(f"Error in get_location_intelligence: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def get_location_intelligence(latitude: float, longitude: float) -> str:
    """
    Get demographic data and points of interest for a location.

    Returns comprehensive intelligence including:
    - Points of Interest (restaurants, schools, healthcare, etc.)
    - Demographics (population, income, household data)
    - Tapestry segmentation (lifestyle and consumer behavior)

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        JSON string with POI data and demographic information

    Examples:
        get_location_intelligence(34.0522, -118.2437)  # LA area intelligence
        get_location_intelligence(40.7128, -74.0060)   # NYC area intelligence
    """
    try:
        result = await gis_executor.get_address_intelligence(latitude, longitude)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_location_intelligence: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})


async def identify_map_location(latitude: float, longitude: float) -> str:
    """
    Identify what's at a specific map location (reverse geocode + intelligence).

    Combines reverse geocoding with address intelligence to provide complete
    location information.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        JSON string with address and location intelligence

    Examples:
        identify_map_location(34.0522, -118.2437)  # Identify LA location
    """
    try:
        # Get address
        address_result = await gis_executor.reverse_geocode_coordinates(
            latitude, longitude
        )

        # Get intelligence
        intel_result = await gis_executor.get_address_intelligence(latitude, longitude)

        # Combine results
        combined = {
            "address": address_result.get("current_address", ""),
            "is_valid": address_result.get("is_valid", False),
            "intelligence": intel_result,
        }

        return json.dumps(combined, indent=0)
    except Exception as e:
        logger.error(f"Error in identify_map_location: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})



# Using raw async functions instead of FunctionTool wrappers to fix AFC compatibility
zoom_map_tool = zoom_map
remove_layer_filter_tool = remove_layer_filter
zoom_to_location_tool = zoom_to_location
toggle_sublayer_visibility_tool = toggle_sublayer_visibility
toggle_layer_visibility_tool = toggle_layer_visibility
apply_layer_filter_tool = apply_layer_filter
pan_map_tool = pan_map
add_map_pin_tool = add_map_pin
remove_map_pins_tool = remove_map_pins
toggle_layer_labels_tool = toggle_layer_labels
geocode_address_tool = geocode_address
reverse_geocode_tool = reverse_geocode_location
get_layer_statistics_tool = get_layer_statistics
get_poi_location_intelligence_tool = get_poi_location_intelligence
get_tapestry_segmentation_intelligence_tool = get_tapestry_segmentation_intelligence
get_demographics_location_intelligence_tool = get_demographics_location_intelligence
get_location_intelligence_tool = get_location_intelligence
identify_map_location_tool = identify_map_location


# Using raw async functions for AFC compatibility (no FunctionTool wrappers)
GIS_ADK_TOOLS = [
    # Layer operations
    toggle_layer_visibility_tool,
    toggle_sublayer_visibility_tool,
    # Filter operations
    apply_layer_filter_tool,
    remove_layer_filter_tool,
    # Zoom/Pan operations
    zoom_map_tool,
    zoom_to_location_tool,
    pan_map_tool,
    # Pin operations
    add_map_pin_tool,
    remove_map_pins_tool,
    # Label operations
    toggle_layer_labels_tool,
    # Geocoding operations
    geocode_address_tool,
    reverse_geocode_tool,
    # Statistics operations
    get_layer_statistics_tool,
    # Intelligence operations
    get_location_intelligence_tool,
    get_poi_location_intelligence_tool,
    get_tapestry_segmentation_intelligence_tool,
    get_demographics_location_intelligence_tool,
    identify_map_location_tool,
    # Spatial-RAG hybrid search tools
    analyze_location_for_marketing_tool,
    compare_locations_tool,
    # Trade area tools (drive-time polygons, radius buffers)
    *TRADE_AREA_TOOLS,
    # Layer Intelligence tools (dynamic layer discovery and querying)
    *LAYER_INTELLIGENCE_TOOLS,
    # Comprehensive lifestyle analysis (top 5 segments with insights)
    get_lifestyle_analysis_tool,
]
