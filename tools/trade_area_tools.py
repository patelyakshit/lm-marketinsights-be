"""
Trade Area Tools

Tools for creating drive-time polygons, radius buffers, and trade area analysis.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ArcGIS service URLs
ARCGIS_ROUTING_URL = "https://route-api.arcgis.com/arcgis/rest/services/World/ServiceAreas/NAServer/ServiceArea_World"
ARCGIS_GEOMETRY_URL = "https://utility.arcgisonline.com/ArcGIS/rest/services/Geometry/GeometryServer"


@dataclass
class TradeAreaResult:
    """Result from trade area creation."""
    success: bool
    geometry: Optional[Dict] = None
    area_sq_miles: float = 0.0
    travel_time_minutes: float = 0.0
    travel_distance_miles: float = 0.0
    error: Optional[str] = None


async def create_drive_time_polygon(
    latitude: float,
    longitude: float,
    time_minutes: int = 15,
    travel_mode: str = "driving",
    direction: str = "from_facility"
) -> str:
    """
    Create a drive-time polygon (service area) from a location.

    This tool generates a polygon representing the area reachable within
    a specified drive time from a location.

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        time_minutes: Drive time in minutes (default: 15)
        travel_mode: Travel mode - "driving" or "walking" (default: driving)
        direction: Direction of travel - "from_facility" or "to_facility"

    Returns:
        JSON string with the polygon geometry and metadata

    Example:
        result = await create_drive_time_polygon(
            latitude=33.0198,
            longitude=-96.6989,
            time_minutes=15
        )
    """
    import aiohttp
    import json
    import os

    api_key = os.getenv("ARCGIS_API_KEY")
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "ARCGIS_API_KEY not configured"
        })

    try:
        # Prepare the service area request
        # Note: Using simple parameters without complex travelMode to ensure API compatibility
        params = {
            "f": "json",
            "token": api_key,
            "facilities": json.dumps({
                "features": [{
                    "geometry": {
                        "x": longitude,
                        "y": latitude,
                        "spatialReference": {"wkid": 4326}
                    }
                }]
            }),
            "defaultBreaks": str(time_minutes),
            "travelDirection": "esriNATravelDirectionFromFacility" if direction == "from_facility" else "esriNATravelDirectionToFacility",
            "outSR": 4326
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ARCGIS_ROUTING_URL}/solveServiceArea",
                data=params
            ) as response:
                result = await response.json()

        if "error" in result:
            return json.dumps({
                "success": False,
                "error": result["error"].get("message", "Unknown error")
            })

        # Extract polygon from response
        polygons = result.get("saPolygons", {}).get("features", [])
        if not polygons:
            return json.dumps({
                "success": False,
                "error": "No service area polygon generated"
            })

        polygon = polygons[0]
        geometry = polygon.get("geometry", {})
        attributes = polygon.get("attributes", {})
        geojson = _arcgis_to_geojson(geometry)

        # Return in operations format for frontend to display on map
        # Note: Only include operations for frontend, exclude raw geometry to reduce context size
        return json.dumps({
            "operations": [
                {
                    "type": "PLOT_GEOJSON",
                    "payload": {
                        "geojson": geojson,
                        "style": {
                            "fillColor": [59, 130, 246, 0.2],
                            "strokeColor": [37, 99, 235, 1],
                            "strokeWidth": 2
                        },
                        "label": f"{time_minutes} min drive time",
                        "id": f"trade-area-{time_minutes}min"
                    }
                }
            ],
            "success": True,
            "message": f"Created {time_minutes}-minute drive time polygon centered at ({latitude}, {longitude})",
            "travel_time_minutes": time_minutes,
            "center": {"latitude": latitude, "longitude": longitude}
        })

    except Exception as e:
        logger.error(f"Error creating drive-time polygon: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


async def create_radius_buffer(
    latitude: float,
    longitude: float,
    radius_miles: float = 5.0
) -> str:
    """
    Create a circular buffer around a point.

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        radius_miles: Radius in miles (default: 5)

    Returns:
        JSON string with the buffer geometry

    Example:
        result = await create_radius_buffer(33.0198, -96.6989, 10)
    """
    import aiohttp
    import json
    import os
    import math

    api_key = os.getenv("ARCGIS_API_KEY")

    try:
        # Convert miles to meters for the buffer operation
        radius_meters = radius_miles * 1609.34

        params = {
            "f": "json",
            "token": api_key,
            "geometries": json.dumps({
                "geometryType": "esriGeometryPoint",
                "geometries": [{
                    "x": longitude,
                    "y": latitude,
                    "spatialReference": {"wkid": 4326}
                }]
            }),
            "inSR": 4326,
            "outSR": 4326,
            "bufferSR": 102100,  # Web Mercator for accurate buffering
            "distances": str(radius_meters),
            "unit": 9001,  # Meters
            "unionResults": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ARCGIS_GEOMETRY_URL}/buffer",
                data=params
            ) as response:
                result = await response.json()

        if "error" in result:
            return json.dumps({
                "success": False,
                "error": result["error"].get("message", "Unknown error")
            })

        geometries = result.get("geometries", [])
        if not geometries:
            return json.dumps({
                "success": False,
                "error": "No buffer generated"
            })

        geometry = geometries[0]

        return json.dumps({
            "success": True,
            "geometry": geometry,
            "geojson": _arcgis_to_geojson(geometry),
            "radius_miles": radius_miles,
            "center": {"latitude": latitude, "longitude": longitude},
            "area_sq_miles": math.pi * radius_miles ** 2
        }, indent=2)

    except Exception as e:
        logger.error(f"Error creating radius buffer: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


async def create_multiple_drive_time_rings(
    latitude: float,
    longitude: float,
    break_values: List[int] = None
) -> str:
    """
    Create multiple drive-time rings (e.g., 5, 10, 15 minutes).

    Args:
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        break_values: List of drive times in minutes (default: [5, 10, 15])

    Returns:
        JSON string with multiple ring geometries

    Example:
        result = await create_multiple_drive_time_rings(
            33.0198, -96.6989,
            break_values=[5, 10, 15, 30]
        )
    """
    import aiohttp
    import json
    import os

    if break_values is None:
        break_values = [5, 10, 15]

    api_key = os.getenv("ARCGIS_API_KEY")
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "ARCGIS_API_KEY not configured"
        })

    try:
        params = {
            "f": "json",
            "token": api_key,
            "facilities": json.dumps({
                "features": [{
                    "geometry": {
                        "x": longitude,
                        "y": latitude,
                        "spatialReference": {"wkid": 4326}
                    }
                }]
            }),
            "defaultBreaks": ",".join(str(b) for b in break_values),
            "travelMode": json.dumps({
                "type": "automobile",
                "name": "Driving Time",
                "impedanceAttributeName": "TravelTime",
                "timeAttributeName": "TravelTime",
                "distanceAttributeName": "Miles"
            }),
            "travelDirection": "esriNATravelDirectionFromFacility",
            "polygonType": "esriNATriangulatedPolygon",
            "outputPolygons": "esriNAOutputPolygonSimplified",
            "trimOuterPolygon": True,
            "splitPolygonsAtBreaks": True,
            "returnFacilities": False,
            "outSR": 4326
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ARCGIS_ROUTING_URL}/solveServiceArea",
                data=params
            ) as response:
                result = await response.json()

        if "error" in result:
            return json.dumps({
                "success": False,
                "error": result["error"].get("message", "Unknown error")
            })

        polygons = result.get("saPolygons", {}).get("features", [])

        rings = []
        for polygon in polygons:
            geometry = polygon.get("geometry", {})
            attributes = polygon.get("attributes", {})

            rings.append({
                "from_minutes": attributes.get("FromBreak", 0),
                "to_minutes": attributes.get("ToBreak"),
                "geometry": geometry,
                "geojson": _arcgis_to_geojson(geometry)
            })

        return json.dumps({
            "success": True,
            "rings": rings,
            "break_values": break_values,
            "center": {"latitude": latitude, "longitude": longitude}
        }, indent=2)

    except Exception as e:
        logger.error(f"Error creating drive-time rings: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def _arcgis_to_geojson(arcgis_geometry: Dict) -> Dict:
    """Convert ArcGIS geometry to GeoJSON format."""
    if "rings" in arcgis_geometry:
        # Polygon
        return {
            "type": "Polygon",
            "coordinates": arcgis_geometry["rings"]
        }
    elif "paths" in arcgis_geometry:
        # Polyline
        return {
            "type": "MultiLineString",
            "coordinates": arcgis_geometry["paths"]
        }
    elif "x" in arcgis_geometry and "y" in arcgis_geometry:
        # Point
        return {
            "type": "Point",
            "coordinates": [arcgis_geometry["x"], arcgis_geometry["y"]]
        }
    return arcgis_geometry


# =============================================================================
# TRADE AREA ANALYSIS FUNCTIONS
# =============================================================================

async def analyze_trade_area_segments(
    trade_area_geometry: Dict,
    tapestry_layer_name: str = "Tapestry Segmentation 2025"
) -> str:
    """
    Analyze Tapestry segments within a trade area.

    This is a high-level function that:
    1. Queries the Tapestry layer with the trade area geometry
    2. Aggregates segment data
    3. Returns top segments with statistics

    Args:
        trade_area_geometry: GeoJSON or ArcGIS geometry of the trade area
        tapestry_layer_name: Name of the Tapestry layer on the map

    Returns:
        JSON string with segment analysis
    """
    from tools.layer_query_tools import (
        query_layer_features,
        aggregate_segment_data,
        get_top_segments
    )
    import json

    # Query segments in trade area
    query_result = await query_layer_features(
        layer_name=tapestry_layer_name,
        geometry=trade_area_geometry,
        out_fields=["TSEGCODE", "TSEGNAME", "TLIFENAME", "THHBASE"]
    )

    try:
        result_data = json.loads(query_result)
    except:
        return json.dumps({
            "success": False,
            "error": "Failed to parse query result"
        })

    if "error" in result_data:
        return query_result

    features = result_data.get("features", [])
    if not features:
        return json.dumps({
            "success": False,
            "error": "No segments found in trade area"
        })

    # Aggregate segment data
    aggregated = aggregate_segment_data(
        features,
        group_field="TSEGCODE",
        value_field="THHBASE"
    )

    # Get top 5 segments
    top_segments = get_top_segments(aggregated, n=5)

    return json.dumps({
        "success": True,
        "total_households": aggregated["total"],
        "segment_count": len(aggregated["segments"]),
        "top_segments": top_segments,
        "all_segments": aggregated["segments"]
    }, indent=2)


# =============================================================================
# ADK FUNCTION TOOLS
# =============================================================================

create_drive_time_polygon_tool = FunctionTool(create_drive_time_polygon)
create_radius_buffer_tool = FunctionTool(create_radius_buffer)
create_multiple_drive_time_rings_tool = FunctionTool(create_multiple_drive_time_rings)
analyze_trade_area_segments_tool = FunctionTool(analyze_trade_area_segments)

TRADE_AREA_TOOLS = [
    create_drive_time_polygon_tool,
    create_radius_buffer_tool,
    create_multiple_drive_time_rings_tool,
    analyze_trade_area_segments_tool,
]
