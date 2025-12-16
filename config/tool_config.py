"""
Dynamic tool filtering configuration.

Provides context-aware control over which tool results are sent to the frontend.

Modes:
- "always": Always send to FE (important intermediate results user should see)
- "never": Never send to FE (internal helper tools, purely for agent logic)
- "auto": Send only if it's the LAST tool in the turn (default behavior)

This allows the same tool to have different behaviors based on context:
- geocode_address might show suggestions (always) OR be used internally (would need override)
- Most actionable tools use "auto" to only send the final result
"""

import logging

logger = logging.getLogger(__name__)

# GIS Tool Configuration
GIS_TOOL_CONFIG = {
    "get_location_intelligence": {
        "send_mode": "always",  # Rich demographic/POI data user requested
        "type": "info",
        "description": "Location intelligence data always shown"
    },
    "identify_map_location": {
        "send_mode": "always",  # Location details user might want
        "type": "info",
        "description": "Map location identification results"
    },

    # ========== NEVER SEND ==========
    # Pure internal helpers that shouldn't clutter the UI

    "reverse_geocode_location": {
        "send_mode": "never",  # Internal address lookup for coordinates
        "type": "intermediate",
        "description": "Internal reverse geocoding, not shown"
    },

    "geocode_address": {
        "send_mode": "auto",  # Only show suggestions if it's the final action
        "type": "intermediate",
        "description": "Geocoding, sent only if user explicitly asked for it"
    },
    "zoom_to_location": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Zoom operation, sent only if final action"
    },
    "pan_map": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Pan operation, sent only if final action"
    },
    "zoom_map": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Zoom in/out, sent only if final action"
    },
    "toggle_layer_visibility": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Layer visibility toggle"
    },
    "toggle_sublayer_visibility": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Sublayer visibility toggle"
    },
    "apply_layer_filter": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Apply attribute filter"
    },
    "remove_layer_filter": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Remove attribute filter"
    },
    "add_map_pin": {
        "send_mode": "always",
        "type": "actionable",
        "description": "Add pin to map"
    },
    "remove_map_pins": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Remove pins from map"
    },
    "toggle_layer_labels": {
        "send_mode": "auto",
        "type": "actionable",
        "description": "Toggle layer labels"
    },
    "get_layer_statistics": {
        "send_mode": "auto",
        "type": "info",
        "description": "Layer statistics, sent only if final"
    },

    # ========== TRADE AREA TOOLS ==========
    # Always send to display on map

    "create_drive_time_polygon": {
        "send_mode": "always",
        "type": "actionable",
        "description": "Drive-time polygon, always displayed on map"
    },
    "create_radius_buffer": {
        "send_mode": "always",
        "type": "actionable",
        "description": "Radius buffer, always displayed on map"
    },
    "create_multiple_drive_time_rings": {
        "send_mode": "always",
        "type": "actionable",
        "description": "Multiple drive-time rings, always displayed on map"
    },

    # ========== TAPESTRY/INTELLIGENCE TOOLS ==========
    "get_tapestry_segmentation_intelligence": {
        "send_mode": "always",
        "type": "info",
        "description": "Tapestry segmentation data, always shown"
    },
    "get_demographics_location_intelligence": {
        "send_mode": "always",
        "type": "info",
        "description": "Demographics data, always shown"
    },
    "get_poi_location_intelligence": {
        "send_mode": "always",
        "type": "info",
        "description": "POI data, always shown"
    },
}


def should_send_tool_to_frontend(
    tool_name: str,
    is_last_tool: bool,
    default_mode: str = "auto"
) -> bool:
    """
    Determine if tool result should be sent to frontend based on context.

    Decision matrix:
    - Mode "always" → Always send (important intermediate results)
    - Mode "never" → Never send (internal helpers)
    - Mode "auto" → Send only if last tool (default behavior)

    Args:
        tool_name: Name of the tool executed
        is_last_tool: Is this the last tool in the current turn?
        default_mode: Default mode for tools not in registry ("auto")

    Returns:
        True if tool result should be sent to FE, False otherwise

    Examples:
        # geocode_address with "always" mode
        should_send_tool_to_frontend("geocode_address", False) → True
        should_send_tool_to_frontend("geocode_address", True) → True

        # zoom_to_location with "auto" mode
        should_send_tool_to_frontend("zoom_to_location", False) → False
        should_send_tool_to_frontend("zoom_to_location", True) → True

        # reverse_geocode_location with "never" mode
        should_send_tool_to_frontend("reverse_geocode_location", False) → False
        should_send_tool_to_frontend("reverse_geocode_location", True) → False
    """
    config = GIS_TOOL_CONFIG.get(tool_name)

    if config is None:
        # Unknown tool - use default mode for safety
        mode = default_mode
        logger.debug(
            f"Tool '{tool_name}' not in config, using default mode: {default_mode}"
        )
    else:
        mode = config.get("send_mode", default_mode)

    # Decision logic based on mode
    if mode == "always":
        return True
    elif mode == "never":
        return False
    elif mode == "auto":
        return is_last_tool  # Only send if this is the final tool in turn
    else:
        # Unknown mode, fallback to auto behavior
        logger.warning(f"Unknown send_mode '{mode}' for tool '{tool_name}', using auto")
        return is_last_tool


# Optional: Global debug override
# Set to True to send ALL tool results to frontend (useful for debugging)
DEBUG_SEND_ALL_TOOLS = False

if DEBUG_SEND_ALL_TOOLS:
    logger.warning("DEBUG_SEND_ALL_TOOLS is enabled - all tool results will be sent to FE")
