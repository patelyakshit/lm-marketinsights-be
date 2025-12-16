"""
GIS Tool Executor - Core operations with LangGraph-compatible response format

Handles all GIS operations including:
- Layer visibility management
- Filtering and spatial queries
- Zoom/Pan operations
- Pin management
- Label operations
- Geocoding and reverse geocoding
- Statistics and address intelligence
"""

import copy
import logging
import uuid
from typing import Dict, List, Any, Optional

import aiohttp
from arcgis.features import FeatureLayer

from utils.arcgis_auth import ArcGISAuthManager
from utils.cache import (
    get_cached_geocode,
    cache_geocode,
    get_cached_reverse_geocode,
    cache_reverse_geocode,
    get_cached_demographics,
    cache_demographics,
)
from utils.semantic_cache import semantic_cache_get, semantic_cache_set

from config.config import (
    GEOLOCATION_API_KEY,
    ARCGIS_GEOCODE_URL,
    ARCGIS_REVERSE_GEOCODE_URL,
    ARCGIS_PLACES_URL,
    ARCGIS_ENRICHMENT_URL,
    ARCGIS_API_KEY,
)

logger = logging.getLogger(__name__)


class GISToolExecutor:
    """Executes GIS operations with LangGraph-compatible response format"""

    def __init__(
        self,
        map_context: Optional[Dict] = None,
        layer_data: Optional[List] = None,
        pin_locations: Optional[List] = None,
    ):
        """
        Initialize GIS executor with state data.

        Args:
            map_context: Current map view state (center, zoom, extent)
            layer_data: Available layers with schemas and metadata
            pin_locations: Current map pins
        """
        self.map_context = map_context or {}
        self.layer_data = layer_data or []
        self.pin_locations = pin_locations or []

        # API endpoints and keys
        self.geocode_url = ARCGIS_GEOCODE_URL
        self.reverse_geocode_url = ARCGIS_REVERSE_GEOCODE_URL
        self.places_url = ARCGIS_PLACES_URL
        self.geoenrich_url = ARCGIS_ENRICHMENT_URL
        self.geolocation_api_key = GEOLOCATION_API_KEY

        self.gis_auth_manager = ArcGISAuthManager()

        logger.info("GISToolExecutor initialized")

    def find_layer_by_name(self, layer_name: str) -> Optional[Dict]:
        """Find layer in state by title"""
        for layer in self.layer_data:
            if layer.get("title") == layer_name:
                return layer
            # Check sublayers
            if "sublayers" in layer:
                for sublayer in layer["sublayers"]:
                    if sublayer.get("title") == layer_name:
                        return sublayer
        return None

    def find_layer_by_id(self, layer_id: str) -> Optional[Dict]:
        """Find layer in state by id"""
        for layer in self.layer_data:
            if layer.get("id") == layer_id:
                return layer
            # Check sublayers
            if "sublayers" in layer:
                for sublayer in layer["sublayers"]:
                    if sublayer.get("id") == layer_id:
                        return sublayer
        return None

    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinate ranges"""
        return -90 <= latitude <= 90 and -180 <= longitude <= 180

    async def toggle_layer_visibility(
        self, layer_id: str, layer_name: str, visible: bool
    ) -> Dict[str, Any]:
        """
        Toggle layer visibility on map.

        Returns LangGraph format:
        {"type": "TOGGLE_LAYER_VISIBILITY", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "TOGGLE_LAYER_VISIBILITY",
                        "payload": {
                            "layerId": layer_id,
                            "layerName": layer_name,
                            "visible": visible,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error toggling layer visibility: {e}")
            return {"type": "ERROR", "payload": {"error": str(e), "layerId": layer_id}}

    async def toggle_sublayer_visibility(
        self, layer_id: str, sublayer_id: str, visible: bool
    ) -> Dict[str, Any]:
        """
        Toggle sublayer visibility.

        Returns LangGraph format:
        {"type": "TOGGLE_SUBLAYER_VISIBILITY", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "TOGGLE_SUBLAYER_VISIBILITY",
                        "payload": {
                            "layerId": layer_id,
                            "sublayerId": sublayer_id,
                            "visible": visible,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error toggling sublayer visibility: {e}")
            return {
                "type": "ERROR",
                "payload": {
                    "error": str(e),
                    "layerId": layer_id,
                    "sublayerId": sublayer_id,
                },
            }

    async def suggest_layers(self, layers: List[Dict]) -> Dict[str, Any]:
        """
        Suggest multiple matching layers.

        Returns LangGraph format:
        {"type": "SUGGEST_LAYERS", "payload": {"layers": [...]}}
        """
        try:
            suggest_layers = []
            for layer in layers:
                suggest_layers.append(
                    {
                        "title": layer.get("title", ""),
                        "visible": bool(layer.get("visible", False)),
                        "layer_id": layer.get("id", ""),
                        "sublayers": layer.get("sublayers", []),
                    }
                )

            return {
                "operations": [
                    {"type": "SUGGEST_LAYERS", "payload": {"layers": suggest_layers}}
                ]
            }
        except Exception as e:
            logger.error(f"Error suggesting layers: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def apply_filter(
        self, layer_id: str, where_clause: str, spatial_lock: bool = False
    ) -> Dict[str, Any]:
        """
        Apply attribute filter to layer.

        Args:
            layer_id: Layer identifier
            where_clause: SQL WHERE clause (e.g., "POPULATION > 1500")
            spatial_lock: Lock filter to current map extent

        Returns LangGraph format:
        {"type": "APPLY_FILTER", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "APPLY_FILTER",
                        "payload": {
                            "layerId": layer_id,
                            "whereClause": where_clause,
                            "spatialLock": spatial_lock,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return {"type": "ERROR", "payload": {"error": str(e), "layerId": layer_id}}

    async def remove_filter(self, layer_id: str) -> Dict[str, Any]:
        """
        Remove filter from layer (reset to show all).

        Returns LangGraph format:
        {"type": "APPLY_FILTER", "payload": {"whereClause": "1=1"}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "APPLY_FILTER",
                        "payload": {
                            "layerId": layer_id,
                            "whereClause": "1=1",
                            "spatialLock": False,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error removing filter: {e}")
            return {"type": "ERROR", "payload": {"error": str(e), "layerId": layer_id}}

    async def zoom_map(self, zoom_action: str, zoom_percentage: int) -> Dict[str, Any]:
        """
        Zoom map in or out.

        Args:
            zoom_action: "zoom_in" or "zoom_out"
            zoom_percentage: Zoom percentage (0-100)

        Returns LangGraph format:
        {"type": "ZOOM_MAP", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "ZOOM_MAP",
                        "payload": {
                            "zoom_action": zoom_action,
                            "zoom_percentage": zoom_percentage,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error zooming map: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def zoom_to_location(
        self,
        extent: Dict[str, Any],
        spatial_reference: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Zoom to location using extent for accurate zooming.

        Args:
            extent: Bounding box {xmin, ymin, xmax, ymax}
            spatial_reference: Spatial reference {wkid: 4326} from geocode results

        Returns LangGraph format:
        {"type": "ZOOM_TO_LOCATION", "payload": {...}}
        """
        try:
            payload = {}

            if extent:
                payload["extent"] = extent

            if spatial_reference:
                payload["spatialReference"] = spatial_reference

            return {
                "operations": [
                    {
                        "type": "ZOOM_TO_LOCATION",
                        "payload": payload,
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error zooming to location: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def zoom_to_features(
        self, layer_id: str, layer_name: str, target: str = "FULL_EXTENT"
    ) -> Dict[str, Any]:
        """
        Zoom to layer features.

        Args:
            layer_id: Layer identifier
            layer_name: Layer name
            target: "FULL_EXTENT" or "FILTERED"

        Returns LangGraph format:
        {"type": "ZOOM_TO_FEATURES", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "ZOOM_TO_FEATURES",
                        "payload": {
                            "layerId": layer_id,
                            "layerName": layer_name,
                            "target": target,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error zooming to features: {e}")
            return {"type": "ERROR", "payload": {"error": str(e), "layerId": layer_id}}

    async def pan_map(self, direction: str, distance: int) -> Dict[str, Any]:
        """
        Pan map in direction.

        Args:
            direction: "north", "south", "east", "west"
            distance: Pan distance as percentage (0-100)

        Returns LangGraph format:
        {"type": "PAN_MAP", "payload": {...}}
        """
        try:
            # Normalize direction aliases
            direction_map = {
                "up": "north",
                "down": "south",
                "left": "west",
                "right": "east",
            }
            normalized_direction = direction_map.get(
                direction.lower(), direction.lower()
            )

            return {
                "operations": [
                    {
                        "type": "PAN_MAP",
                        "payload": {
                            "distance": distance,
                            "direction": normalized_direction,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error panning map: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def suggest_pin(self, pins: List[Dict]) -> Dict[str, Any]:
        """
        Suggest pin locations (from geocoding results).

        Args:
            pins: List of pin suggestions with address, lat, lng, score

        Returns LangGraph format:
        {"type": "SUGGEST_PIN", "payload": {"pins": [...]}}
        """
        try:
            return {"operations": [{"type": "SUGGEST_PIN", "payload": {"pins": pins}}]}
        except Exception as e:
            logger.error(f"Error suggesting pins: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def remove_pin(self, pin_ids: List[str]) -> Dict[str, Any]:
        """
        Remove pins from map.

        Args:
            pin_ids: List of pin IDs to remove, or ["all"] for all pins

        Returns LangGraph format:
        {"type": "REMOVE_PIN", "payload": {"pinIds": [...]}}
        """
        try:
            return {
                "operations": [{"type": "REMOVE_PIN", "payload": {"pinIds": pin_ids}}]
            }
        except Exception as e:
            logger.error(f"Error removing pins: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def toggle_labels(
        self, layer_id: str, enabled: bool, label_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Toggle layer labels.

        Args:
            layer_id: Layer identifier
            enabled: True to show labels, False to hide
            label_field: Field name for labels (e.g., "Name", "City")

        Returns LangGraph format:
        {"type": "TOGGLE_LABELS", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "TOGGLE_LABELS",
                        "payload": {
                            "layerId": layer_id,
                            "enabled": enabled,
                            "labelField": label_field,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error toggling labels: {e}")
            return {"type": "ERROR", "payload": {"error": str(e), "layerId": layer_id}}

    async def toggle_sublayer_labels(
        self,
        layer_id: str,
        sublayer_id: str,
        enabled: bool,
        label_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Toggle sublayer labels.

        Returns LangGraph format:
        {"type": "TOGGLE_SUBLAYER_LABELS", "payload": {...}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "TOGGLE_SUBLAYER_LABELS",
                        "payload": {
                            "layerId": layer_id,
                            "sublayerId": sublayer_id,
                            "enabled": enabled,
                            "labelField": label_field,
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error toggling sublayer labels: {e}")
            return {
                "type": "ERROR",
                "payload": {
                    "error": str(e),
                    "layerId": layer_id,
                    "sublayerId": sublayer_id,
                },
            }

    async def suggest_labels(self, labels: List[str]) -> Dict[str, Any]:
        """
        Suggest label field options.

        Returns LangGraph format:
        {"type": "SUGGEST_LABELS", "payload": {"labels": [...]}}
        """
        try:
            return {
                "operations": [
                    {"type": "SUGGEST_LABELS", "payload": {"labels": labels}}
                ]
            }
        except Exception as e:
            logger.error(f"Error suggesting labels: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def suggest_location(
        self, longitude: float, latitude: float, address: str, score: int
    ) -> Dict[str, Any]:
        """
        Suggest location (from geocoding).

        Returns LangGraph format:
        {"type": "SUGGEST_LOCATION", "payload": {"location": {...}}}
        """
        try:
            return {
                "operations": [
                    {
                        "type": "SUGGEST_LOCATION",
                        "payload": {
                            "location": {
                                "longitude": longitude,
                                "latitude": latitude,
                                "address": address,
                                "score": score,
                            }
                        },
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error suggesting location: {e}")
            return {"type": "ERROR", "payload": {"error": str(e)}}

    async def get_coordinates_for_address(
        self, address: str, max_locations: int = 5
    ) -> Dict[str, Any]:
        """
        Geocode address to coordinates using ArcGIS API.
        Results are cached for 1 hour to reduce API calls.

        Args:
            address: Address string to geocode
            max_locations: Maximum number of candidates

        Returns:
            {"data": {...}} or {"error": "..."}
        """
        # Check cache first (only for single location requests for consistency)
        if max_locations == 1:
            cached = await get_cached_geocode(address)
            if cached:
                logger.info(f"Geocode cache hit for: {address[:50]}")
                return cached

        params = {
            "address": address,
            "outFields": "*",
            "f": "json",
            "token": self.geolocation_api_key,
            "maxLocations": max_locations,
            "searchExtent": "-171.791110603, 18.91619, -66.96466, 71.3577635769",  # USA bounds
            "location": "-98.5795,39.8283",  # USA center
            "searchExtent_sr": "4326",
            "countryCode": "USA",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.geocode_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    result = {"data": data}

                    # Cache successful results (only single location for consistency)
                    if max_locations == 1 and "error" not in result:
                        await cache_geocode(address, result)

                    return result
        except Exception as e:
            error_msg = f"Error geocoding address: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def get_coordinates_suggestions(self, address: str, max_candidates: int) -> List[Dict]:
        """
        Get geocoding suggestions with coordinates.

        Returns list of suggestions:
        [{"address": "...", "score": 100, "latitude": 34.05, "longitude": -118.25, "id": "uuid"}, ...]
        """
        response = await self.get_coordinates_for_address(address, max_locations=max_candidates)
        if "error" in response:
            return []

        data = response.get("data", {})
        suggestions = []
        for candidate in data.get("candidates", []):
            attrs = candidate.get("attributes", {})
            extent = candidate.get("extent", {})

            suggestions.append(
                {
                    "address": candidate.get("address"),
                    "score": candidate.get("score"),
                    "longitude": candidate["location"].get("x"),
                    "latitude": candidate["location"].get("y"),
                    "attributes": attrs,
                    "extent": extent,
                    "id": str(uuid.uuid4()),
                }
            )

        # Sort by score descending
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions

    async def reverse_geocode_coordinates(
        self, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        """
        Reverse geocode coordinates to address using ArcGIS API.
        Results are cached for 1 hour to reduce API calls.

        Returns:
            {"current_address": "...", "is_valid": True} or {"current_address": "", "is_valid": False}
        """
        if not self.validate_coordinates(latitude, longitude):
            return {"current_address": "", "is_valid": False}

        # Check cache first
        cached = await get_cached_reverse_geocode(latitude, longitude)
        if cached:
            logger.info(f"Reverse geocode cache hit for: ({latitude}, {longitude})")
            return cached

        params = {
            "f": "json",
            "token": self.geolocation_api_key,
            "location": f"{longitude},{latitude}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.reverse_geocode_url, params=params
                ) as response:
                    if response.status != 200:
                        return {"current_address": "", "is_valid": False}

                    data = await response.json()
                    result = {
                        "current_address": data["address"]["LongLabel"],
                        "is_valid": True,
                    }

                    # Cache successful result
                    await cache_reverse_geocode(latitude, longitude, result)
                    return result
        except Exception as e:
            logger.error(f"Error reverse geocoding: {str(e)}")
            return {"current_address": "", "is_valid": False}

    async def get_field_statistics(
        self, layer_id: str, field_names: List[str], service_url: str, layer_index: int
    ) -> Dict[str, Any]:
        """
        Get statistics for numeric fields in a layer.

        Args:
            layer_id: Layer identifier
            field_names: List of field names to calculate stats for
            service_url: ArcGIS service URL
            layer_index: Layer index in service

        Returns:
            Statistics data with count, sum, min, max, avg, stddev
        """
        try:
            # Build statistics request
            out_statistics = []
            for field_name in field_names:
                for stat_type in ["count", "sum", "min", "max", "avg", "stddev"]:
                    out_statistics.append(
                        {
                            "onStatisticField": field_name,
                            "outStatisticFieldName": f"{field_name}_{stat_type}",
                            "statisticType": stat_type,
                        }
                    )

            # Query statistics from ArcGIS REST API
            query_url = f"{service_url}/{layer_index}/query"
            params = {
                "where": "1=1",
                "outStatistics": str(out_statistics),
                "f": "json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(query_url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Statistics query failed: {response.status}")

                    data = await response.json()
                    features = data.get("features", [])

                    if features:
                        return {
                            "success": True,
                            "statistics": features[0]["attributes"],
                        }
                    return {"success": False, "error": "No statistics data"}

        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
            return {"success": False, "error": str(e)}

    async def get_esri_poi_data(
        self, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        """
        Get Points of Interest data from ArcGIS Places API.

        Returns POI data for location (restaurants, schools, etc.)
        """
        category_ids = [
            "4d4b7104d754a06370d81259",  # Arts & Entertainment
            "4d4b7105d754a06375d81259",  # Food
            "63be6904847c3692a84b9b9a",  # Healthcare
            "63be6904847c3692a84b9bb5",  # Landmarks
            "63be6904847c3692a84b9bb9",  # Shopping
            "4d4b7105d754a06377d81259",  # Professional Services
            "4d4b7105d754a06378d81259",  # Recreation
            "4f4528bc4b90abdf24c9de85",  # Travel & Transport
            "4d4b7105d754a06379d81259",  # Residence
        ]

        params = {
            "x": longitude,
            "y": latitude,
            "categoryIds": ",".join(set(category_ids)),
            "token": self.geolocation_api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.places_url, params=params, timeout=60
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error fetching POI data: {str(e)}")
            return {}

    async def get_esri_geoenrich_data_general_info(
        self,
        latitude: float,
        longitude: float
    ):
        """Get demographics information for given location (lat and long)."""
        import json as json_lib

        # General demographic variables (using correct collection names)
        analysis_vars = [
            "KeyGlobalFacts.TOTPOP",      # Total population
            "KeyGlobalFacts.TOTHH",       # Total households
            "KeyUSFacts.MEDHINC_CY",      # Median household income
            "KeyUSFacts.MEDAGE_CY",       # Median age
            "KeyUSFacts.PCI_CY",          # Per capita income
        ]

        try:
            # Convert geometry to ArcGIS JSON format
            arcgis_geometry = json_lib.dumps({
                "x": longitude,
                "y": latitude,
                "spatialReference": {"wkid": 4326}
            })

            # Use POST with form data - MUST use ARCGIS_API_KEY
            params = {
                "f": "json",
                "token": ARCGIS_API_KEY,
                "studyAreas": f'[{{"geometry":{arcgis_geometry}}}]',
                "analysisVariables": json_lib.dumps(analysis_vars),
                "returnGeometry": "false",
            }

            async with aiohttp.ClientSession() as session:
                # Use POST method with form data
                async with session.post(
                    self.geoenrich_url,
                    data=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as general_resp:
                    general_data = await general_resp.json()

                # Check for API error
                if "error" in general_data:
                    logger.error(f"GeoEnrich API error: {general_data['error']}")
                    return {"fields": [], "features": [], "error": general_data["error"]}

                # Extract features and fields
                general_features = (
                    general_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("features", [])
                )
                general_fields = (
                    general_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("fields", [])
                )
                return {"fields": general_fields, "features": general_features}

        except Exception as e:
            logger.error(f"Error fetching GeoEnrich data: {str(e)}")
            return {"fields": [], "features": []}

    async def get_esri_geoenrich_data_tapestry(
        self,
        latitude: float,
        longitude: float
    ):
        """Get tapestry segmentation detail for given lat & long"""
        import json as json_lib

        # Tapestry segmentation variables (correct 2025 format)
        tapestry_vars = [
            "Tapestry.TOP1NAME",   # Top 1 Segment Name
            "Tapestry.TOP1CODE",   # Top 1 Segment Code
            "Tapestry.TOP2NAME",   # Top 2 Segment Name
            "Tapestry.TOP2CODE",   # Top 2 Segment Code
            "Tapestry.TOP3NAME",   # Top 3 Segment Name
            "Tapestry.TOTPOP",     # Total Population
        ]

        try:
            # Convert geometry to ArcGIS JSON format
            arcgis_geometry = json_lib.dumps({
                "x": longitude,
                "y": latitude,
                "spatialReference": {"wkid": 4326}
            })

            # Use POST with form data - MUST use ARCGIS_API_KEY (not GEOLOCATION_API_KEY)
            params = {
                "f": "json",
                "token": ARCGIS_API_KEY,
                "studyAreas": f'[{{"geometry":{arcgis_geometry}}}]',
                "analysisVariables": json_lib.dumps(tapestry_vars),
                "returnGeometry": "false",
            }

            async with aiohttp.ClientSession() as session:
                # Use POST method with form data (URL already includes /enrich)
                async with session.post(
                    self.geoenrich_url,
                    data=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as tapestry_resp:
                    tapestry_data = await tapestry_resp.json()

                # Check for API error
                if "error" in tapestry_data:
                    logger.error(f"GeoEnrich API error: {tapestry_data['error']}")
                    return {"fields": [], "features": [], "error": tapestry_data["error"]}

                tapestry_features = (
                    tapestry_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("features", [])
                )
                tapestry_fields = (
                    tapestry_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("fields", [])
                )
                return {
                    "fields": tapestry_fields,
                    "features": tapestry_features,
                }
        except Exception as e:
            logger.error(f"Error fetching GeoEnrich tapestry data: {str(e)}")
            return {"fields": [], "features": []}

    async def get_esri_geoenrich_data(
        self, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        """
        Get demographic data from ArcGIS GeoEnrichment API.
        Results are cached for 30 minutes to reduce API calls.

        Returns demographics and tapestry segmentation data.
        """
        import json as json_lib

        # Check cache first
        cached = await get_cached_demographics(latitude, longitude)
        if cached:
            logger.info(f"Demographics cache hit for: ({latitude}, {longitude})")
            return cached

        # General demographic variables (using correct collection names)
        analysis_vars = [
            "KeyGlobalFacts.TOTPOP",      # Total population
            "KeyGlobalFacts.TOTHH",       # Total households
            "KeyUSFacts.MEDHINC_CY",      # Median household income
            "KeyUSFacts.MEDAGE_CY",       # Median age
            "KeyUSFacts.AVGHHSZ_CY",      # Avg household size
        ]

        # Tapestry segmentation variables (correct 2025 format)
        tapestry_vars = [
            "Tapestry.TOP1NAME",   # Top 1 Segment Name
            "Tapestry.TOP1CODE",   # Top 1 Segment Code
            "Tapestry.TOP2NAME",   # Top 2 Segment Name
            "Tapestry.TOP3NAME",   # Top 3 Segment Name
            "Tapestry.TOTPOP",     # Total Population
        ]

        try:
            # Convert geometry to ArcGIS JSON format
            arcgis_geometry = json_lib.dumps({
                "x": longitude,
                "y": latitude,
                "spatialReference": {"wkid": 4326}
            })

            # Prepare params for POST request - MUST use ARCGIS_API_KEY
            general_params = {
                "f": "json",
                "token": ARCGIS_API_KEY,
                "studyAreas": f'[{{"geometry":{arcgis_geometry}}}]',
                "analysisVariables": json_lib.dumps(analysis_vars),
                "returnGeometry": "false",
            }

            tapestry_params = {
                "f": "json",
                "token": ARCGIS_API_KEY,
                "studyAreas": f'[{{"geometry":{arcgis_geometry}}}]',
                "analysisVariables": json_lib.dumps(tapestry_vars),
                "returnGeometry": "false",
            }

            async with aiohttp.ClientSession() as session:
                # Fetch both in parallel using POST method
                async with session.post(
                    self.geoenrich_url,
                    data=general_params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as general_resp:
                    general_data = await general_resp.json()

                async with session.post(
                    self.geoenrich_url,
                    data=tapestry_params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as tapestry_resp:
                    tapestry_data = await tapestry_resp.json()

                # Check for API errors in response
                if "error" in general_data:
                    logger.error(f"GeoEnrich general API error: {general_data['error']}")
                if "error" in tapestry_data:
                    logger.error(f"GeoEnrich tapestry API error: {tapestry_data['error']}")

                # Extract features and fields
                general_features = (
                    general_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("features", [])
                )
                general_fields = (
                    general_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("fields", [])
                )

                tapestry_features = (
                    tapestry_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("features", [])
                )
                tapestry_fields = (
                    tapestry_data.get("results", [{}])[0]
                    .get("value", {})
                    .get("FeatureSet", [{}])[0]
                    .get("fields", [])
                )

                result = {
                    "general": {"fields": general_fields, "features": general_features},
                    "tapestry": {
                        "fields": tapestry_fields,
                        "features": tapestry_features,
                    },
                }

                # Cache successful result
                await cache_demographics(latitude, longitude, result)
                return result

        except Exception as e:
            logger.error(f"Error fetching GeoEnrich data: {str(e)}")
            return {
                "general": {"fields": [], "features": []},
                "tapestry": {"fields": [], "features": []},
            }

    async def get_address_intelligence(
        self, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        """
        Get complete address intelligence (POI + demographics).

        Returns:
            Complete intelligence data for location
        """
        try:
            if not self.validate_coordinates(latitude, longitude):
                return {"error": "Invalid coordinates"}

            # Fetch POI and demographic data in parallel
            poi_data = await self.get_esri_poi_data(latitude, longitude)
            geoenrich_data = await self.get_esri_geoenrich_data(latitude, longitude)

            return {
                "address": "",
                "latitude": latitude,
                "longitude": longitude,
                "poi_data": poi_data,
                "geoenrich_data": geoenrich_data,
            }

        except Exception as e:
            logger.error(f"Error getting address intelligence: {e}")
            return {"error": str(e)}

    def _get_stats_out_format(self, field_name: str):
        return [
            {"onStatisticField": field_name, "outStatisticFieldName": x, "statisticType": x}
            for x in ("count", "sum", "min", "max", "avg", "stddev")
        ]

    def _build_schema_from_example(self, examples: List) -> Dict:
        if not len(examples):
            return {}
        
        example: Dict = examples[0]

        type_map = {
            int: "number",
            float: "number",
            str: "string",
        }
        schema = {}
        for ek, ev in example.items():
            if type(ev) in type_map:
                schema[ek] = type_map[type(ev)]
            else:
                schema[ek] = "string"
        return schema

    def filter_layer_by_id(self, layers: List[Dict], layer_id: str) -> Dict:
        for layer in layers:
            _layer_id:str = layer.get("id")
            if _layer_id.lower() == layer_id:
                schema = layer.get('schema', {})
                if not len(schema):
                    example = layer.get('example_data', [])
                    schema = self._build_schema_from_example(example)
                
                _layer = copy.deepcopy(layer)
                _layer.update({'schema': schema})
                return _layer

        return {}

    def _get_field_stats(self, service_url: str, field_name: str, filters: str):

        gis_client = self.gis_auth_manager.get_connection()

        try:
            feature_layer = FeatureLayer(service_url, gis=gis_client)
            result = feature_layer.query(
                where=filters,
                out_statistics=self._get_stats_out_format(field_name),
                return_geometry=False,
                spatial_rel="esriSpatialRelIntersects",
            )
            if result.features:
                return result.features[0].attributes

        except Exception as e:
            print(f"Statistic Query Error: {str(e)}")

        return {}

    def get_fields_statistics(
        self, layer: Dict, field_names: List[str], filters: str = "1=1", limit: int = -1
    ) -> Dict:
        service_url = layer['service_url']
        statistics = {}
        schema = layer.get("schema", {})
        type_map = {
            "integer": "number",
            "decimal": "number",
            "float": "number",
            "string": "string",
        }
        # title = layer.get("title", "")

        for fname in field_names:
            ftype = schema[fname].get("type")

            if fname not in schema or type_map.get(ftype, "") != "number":
                continue

            stats = self._get_field_stats(service_url, fname, filters)
            statistics[fname] = stats
            # statistics[fname] = {
            #     "statistic": stats,
            #     "layer_name": title,
            #     "field_name": fname,
            #     "field_descrption": schema[fname].get("description", ""),
            # }

            if limit > 0 and len(statistics) >= limit:
                break

        return list(statistics.values())


# Global instance
gis_executor = GISToolExecutor()
