"""
Lifestyle Analysis Tools

Provides comprehensive tapestry/lifestyle segmentation analysis with:
- Top 5 segments with names, percentages, and household counts
- Enriched segment profiles with demographics
- AI-generated insights per segment
- Business recommendations based on segment mix
"""

import asyncio
import json
import logging
import math
import uuid
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from google.adk.tools import FunctionTool, ToolContext
from google.genai import types as genai_types
from decouple import config

from utils.genai_client_manager import get_genai_client
from knowledge.tapestry_service import get_tapestry_service
from managers.websocket_manager import manager as ws_manager

logger = logging.getLogger(__name__)

# ArcGIS API configuration
ARCGIS_API_KEY = config("ARCGIS_API_KEY", default="")
ARCGIS_USERNAME = config("ARCGIS_USERNAME", default="")
ARCGIS_PASSWORD = config("ARCGIS_PASSWORD", default="")
ARCGIS_PORTAL_URL = config("ARCGIS_PORTAL_URL", default="https://www.arcgis.com")
GEOENRICH_URL = "https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/enrich"

# Tapestry Feature Layer configuration (query directly without GeoEnrichment)
# Set this to your hosted Tapestry Block Group feature layer REST endpoint
TAPESTRY_FEATURE_LAYER_URL = config(
    "TAPESTRY_FEATURE_LAYER_URL",
    default="https://services3.arcgis.com/wbaN2aypfh77OFPp/arcgis/rest/services/Tapestry_Segmentation_Block_Group_2025/FeatureServer/0"
)
# Use feature layer instead of GeoEnrichment (set to "true" to enable)
USE_TAPESTRY_FEATURE_LAYER = config("USE_TAPESTRY_FEATURE_LAYER", default="false").lower() == "true"

# Household Demographics Layer configuration (for accurate segment weighting)
# Esri Updated Demographics Variables 2024 - Census Tract level
HOUSEHOLD_LAYER_URL = config(
    "HOUSEHOLD_LAYER_URL",
    default="https://services8.arcgis.com/peDZJliSvYims39Q/arcgis/rest/services/Esri_Updated_Demographics_Variables_2024/FeatureServer/0"
)
# Enable household data integration (joins with Tapestry for accurate weighting)
USE_HOUSEHOLD_LAYER = config("USE_HOUSEHOLD_LAYER", default="true").lower() == "true"

# Cache for OAuth token
_oauth_token_cache = {"token": None, "expires": 0}


async def get_arcgis_oauth_token() -> str:
    """
    Get an OAuth token for ArcGIS Online using username/password.
    Tokens are cached until near expiry.

    Returns:
        OAuth token string, or falls back to ARCGIS_API_KEY if auth fails
    """
    import time

    # Check cache first
    current_time = time.time()
    if _oauth_token_cache["token"] and _oauth_token_cache["expires"] > current_time + 60:
        return _oauth_token_cache["token"]

    # If no username/password, fall back to API key
    if not ARCGIS_USERNAME or not ARCGIS_PASSWORD:
        logger.info("No ArcGIS username/password configured, using API key")
        return ARCGIS_API_KEY

    # Generate new token
    token_url = f"{ARCGIS_PORTAL_URL}/sharing/rest/generateToken"

    params = {
        "f": "json",
        "username": ARCGIS_USERNAME,
        "password": ARCGIS_PASSWORD,
        "referer": "http://localhost",
        "expiration": 60,  # 60 minutes
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                data = await response.json()

        if "error" in data:
            logger.error(f"Failed to generate OAuth token: {data['error']}")
            return ARCGIS_API_KEY  # Fall back to API key

        token = data.get("token")
        expires = data.get("expires", 0) / 1000  # Convert from ms to seconds

        if token:
            _oauth_token_cache["token"] = token
            _oauth_token_cache["expires"] = expires
            logger.info("Generated new ArcGIS OAuth token")
            return token

        return ARCGIS_API_KEY

    except Exception as e:
        logger.error(f"Error generating OAuth token: {e}")
        return ARCGIS_API_KEY


# Segment colors by LifeMode group (matching MVP)
SEGMENT_COLORS = {
    "1": "#9A5527",  # Affluent Estates - Rust
    "2": "#847A28",  # Upscale Avenues - Brown-Gold
    "3": "#A88704",  # Uptown Individuals - Gold
    "4": "#027373",  # Family Landscapes - Cyan
    "5": "#01715D",  # Gen X Urban - Dark Teal
    "6": "#557332",  # Cozy Country Living - Olive
    "7": "#37733F",  # Ethnic Enclaves - Green
    "8": "#8C5182",  # Middle Ground - Mauve
    "9": "#585182",  # Senior Styles - Purple
    "10": "#036473", # Rustic Outposts - Teal
    "11": "#155E81", # Midtown Singles - Dark Blue
    "12": "#9B660F", # Hometown - Orange-Brown
    "13": "#6B8E23", # Next Wave - Olive Drab
    "14": "#4682B4", # Scholars and Patriots - Steel Blue
    # Fallback for unknown
    "default": "#64748B"
}


@dataclass
class SegmentData:
    """Data structure for a single tapestry segment."""
    code: str
    name: str
    percentage: float
    household_count: int
    lifemode_group: str
    urbanization: str
    median_age: Optional[float] = None
    median_income: Optional[float] = None
    median_net_worth: Optional[float] = None
    homeownership_rate: Optional[float] = None
    description: str = ""
    characteristics: List[str] = None
    color: str = "#64748B"
    insight: str = ""

    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = []


@dataclass
class LifestyleReport:
    """Complete lifestyle analysis report."""
    address: str
    latitude: float
    longitude: float
    total_households: int
    segments: List[SegmentData]
    business_insight: str = ""
    generated_at: str = ""
    buffer_miles: float = None
    drive_time_minutes: int = None


def get_segment_color(code: str) -> str:
    """Get color for a segment based on its LifeMode group."""
    if not code:
        return SEGMENT_COLORS["default"]
    # Extract the number prefix (e.g., "1A" -> "1", "10B" -> "10")
    num_part = ""
    for char in code:
        if char.isdigit():
            num_part += char
        else:
            break
    return SEGMENT_COLORS.get(num_part, SEGMENT_COLORS["default"])


async def fetch_tapestry_data(
    latitude: float,
    longitude: float,
    buffer_miles: float = 1.0,
    drive_time_minutes: int = None
) -> Dict[str, Any]:
    """
    Fetch comprehensive tapestry segmentation data.

    Uses either:
    - Direct feature layer query (if USE_TAPESTRY_FEATURE_LAYER=true) - NO credits for radius, ~0.5 credits for drive-time
    - GeoEnrichment API (default) - consumes credits

    Args:
        latitude: Location latitude
        longitude: Location longitude
        buffer_miles: Radius around point (default 1 mile for trade area)
        drive_time_minutes: Optional drive time in minutes (e.g., 15, 30, 60). If provided, uses drive-time polygon instead of radius.

    Returns top 5 segments with names, codes, and percentages.
    """
    # Check if we should use the feature layer instead of GeoEnrichment
    if USE_TAPESTRY_FEATURE_LAYER and TAPESTRY_FEATURE_LAYER_URL:
        logger.info("Using Tapestry Feature Layer (no GeoEnrichment credits)")
        return await fetch_tapestry_from_feature_layer(latitude, longitude, buffer_miles, drive_time_minutes)

    logger.info("Using GeoEnrichment API for tapestry data")
    return await fetch_tapestry_from_geoenrichment(latitude, longitude, buffer_miles)


async def fetch_tapestry_from_geoenrichment(latitude: float, longitude: float, buffer_miles: float = 1.0) -> Dict[str, Any]:
    """
    Fetch tapestry data from Esri GeoEnrichment API (consumes credits).
    """
    # Use Ring Buffer study area to get more comprehensive segment data
    study_areas = json.dumps([{
        "geometry": {"x": longitude, "y": latitude},
        "areaType": "RingBuffer",
        "bufferUnits": "esriMiles",
        "bufferRadii": [buffer_miles]
    }])

    # Request tapestry data collection which includes all segment rankings
    params = {
        "f": "json",
        "token": ARCGIS_API_KEY,
        "studyAreas": study_areas,
        "dataCollections": json.dumps(["tapestry"]),
        "useData": json.dumps({"sourceCountry": "US"}),
        "returnGeometry": "false",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GEOENRICH_URL,
                data=params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                data = await response.json()

        if "error" in data:
            logger.error(f"GeoEnrich API error: {data['error']}")
            return {"error": data["error"]}

        # Extract features
        features = (
            data.get("results", [{}])[0]
            .get("value", {})
            .get("FeatureSet", [{}])[0]
            .get("features", [])
        )

        if features:
            attrs = features[0].get("attributes", {})
            logger.info(f"GeoEnrich returned {len(attrs)} attributes for tapestry data")
            # Log available TOP fields for debugging
            top_fields = [k for k in attrs.keys() if "TOP" in k.upper()]
            logger.info(f"Available TOP fields: {top_fields[:20]}")  # First 20
            return {"attributes": attrs}

        return {"attributes": {}}

    except Exception as e:
        logger.error(f"Error fetching tapestry data: {e}")
        return {"error": str(e)}


async def generate_drive_time_polygon(gis, latitude: float, longitude: float, minutes: int) -> Optional[Any]:
    """
    Generate a drive-time polygon using ArcGIS Service Area API.

    This consumes approximately 0.5 ArcGIS credits per polygon.

    Args:
        gis: Connected GIS object
        latitude: Center point latitude
        longitude: Center point longitude
        minutes: Drive time in minutes

    Returns:
        Polygon geometry or None if failed
    """
    try:
        from arcgis.network.analysis import generate_service_areas
        from arcgis.features import FeatureSet

        logger.info(f"Generating {minutes}-minute drive-time polygon (uses ~0.5 credits)")

        # Create the facility as a FeatureSet
        facilities_features = [{
            "geometry": {"x": longitude, "y": latitude, "spatialReference": {"wkid": 4326}},
            "attributes": {"ObjectID": 1, "Name": "Location"}
        }]
        facilities = FeatureSet(facilities_features)

        # Generate service area with correct parameters
        # break_values is a space-separated string, e.g., "15" for 15 minutes
        result = generate_service_areas(
            facilities=facilities,
            break_values=str(minutes),  # String format: "15"
            break_units="Minutes",
            travel_direction="Away From Facility",
            time_of_day=None,  # Use typical traffic
            use_hierarchy=True,
            detailed_polygons=False,  # Faster, less detailed
            gis=gis
        )

        # Result is a ToolOutput object with service_areas FeatureSet
        if result and hasattr(result, 'service_areas'):
            service_areas = result.service_areas
            # service_areas is a FeatureSet, get features from it
            if hasattr(service_areas, 'features') and len(service_areas.features) > 0:
                feature = service_areas.features[0]
                polygon = feature.geometry
                logger.info(f"Successfully generated drive-time polygon with {len(polygon.get('rings', []))} rings")
                return polygon

        logger.warning("No service area polygon returned")
        return None

    except Exception as e:
        logger.error(f"Error generating drive-time polygon: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def fetch_household_data(
    token: str,
    latitude: float,
    longitude: float,
    buffer_meters: int,
    polygon_dict: dict = None
) -> Dict[str, int]:
    """
    Fetch household data from Esri Demographics layer for the given area.

    Returns a dictionary mapping Census Tract GEOID to household count.
    This data is used to weight Tapestry segments by actual household counts
    instead of just counting block groups.

    Args:
        token: ArcGIS OAuth token for authentication
        latitude: Center point latitude
        longitude: Center point longitude
        buffer_meters: Buffer radius in meters
        polygon_dict: Optional polygon geometry for drive-time queries

    Returns:
        Dict mapping tract GEOID to household count, e.g., {"48085030101": 1500, ...}
    """
    if not USE_HOUSEHOLD_LAYER:
        logger.info("Household layer integration disabled")
        return {}

    try:
        # Build layer URL
        base_url = HOUSEHOLD_LAYER_URL.rstrip("/")
        if not base_url.split("/")[-1].isdigit():
            base_url = f"{base_url}/12"  # Default to Tract layer
        query_url = f"{base_url}/query"

        logger.info(f"Fetching household data from: {base_url}")

        # Build query params using REST API (more reliable spatial filtering)
        if polygon_dict:
            params = {
                'f': 'json',
                'token': token,
                'where': '1=1',
                'geometry': json.dumps(polygon_dict),
                'geometryType': 'esriGeometryPolygon',
                'spatialRel': 'esriSpatialRelIntersects',
                'outFields': 'ID,TOTHH_CY',
                'returnGeometry': 'false',
                'resultRecordCount': 5000
            }
        else:
            # Use envelope query (Esri Demographics layer doesn't support distance queries)
            # Convert buffer meters to degrees (approximate at mid-latitudes)
            # 1 degree latitude ≈ 111,000 meters, longitude varies by latitude
            buffer_miles = buffer_meters / 1609.34
            lat_offset = buffer_miles / 69.0  # degrees per mile latitude
            lon_offset = buffer_miles / (69.0 * abs(math.cos(math.radians(latitude))))  # adjust for latitude

            envelope = {
                'xmin': longitude - lon_offset,
                'ymin': latitude - lat_offset,
                'xmax': longitude + lon_offset,
                'ymax': latitude + lat_offset,
                'spatialReference': {'wkid': 4326}
            }

            params = {
                'f': 'json',
                'token': token,
                'where': '1=1',
                'geometry': json.dumps(envelope),
                'geometryType': 'esriGeometryEnvelope',
                'spatialRel': 'esriSpatialRelIntersects',
                'outFields': 'ID,HHPOP_CY',  # HHPOP_CY is the household population field
                'returnGeometry': 'false',
                'resultRecordCount': 10000
            }

        # Execute query via REST API
        async with aiohttp.ClientSession() as session:
            if polygon_dict:
                async with session.post(query_url, data=params, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    data = await response.json()
            else:
                async with session.get(query_url, params=params, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    data = await response.json()

        if 'error' in data:
            logger.warning(f"Household query error: {data['error']}")
            return {}

        features = data.get('features', [])
        logger.info(f"Household layer returned {len(features)} tracts")

        if not features:
            return {}

        # Build GEOID -> household count mapping
        # The tract layer uses 'ID' for GEOID and 'HHPOP_CY' (or 'TOTHH_CY') for household count
        tract_households = {}
        total_hh = 0

        for feature in features:
            attrs = feature.get('attributes', {})
            geoid = str(attrs.get('ID', ''))
            # Try both field names (different layers may use different names)
            hh_count = attrs.get('HHPOP_CY') or attrs.get('TOTHH_CY') or 0

            if geoid:
                tract_households[geoid] = hh_count
                total_hh += hh_count

        logger.info(f"Mapped {len(tract_households)} tracts with {total_hh:,} total households")
        return tract_households

    except Exception as e:
        logger.warning(f"Error fetching household data (will use block group count): {e}")
        return {}


async def fetch_tapestry_from_feature_layer(
    latitude: float,
    longitude: float,
    buffer_miles: float = 1.0,
    drive_time_minutes: int = None
) -> Dict[str, Any]:
    """
    Fetch tapestry data by querying hosted Tapestry Block Group feature layer directly.

    This method queries your own hosted Tapestry feature layer with a spatial
    buffer query, then aggregates block group data to calculate top 5 segments.

    Credits:
    - Radius query: FREE (no credits)
    - Drive-time query: ~0.5 credits (for generating the drive-time polygon)

    Args:
        latitude: Location latitude
        longitude: Location longitude
        buffer_miles: Radius around point (default 1 mile for trade area)
        drive_time_minutes: Optional drive time in minutes. If provided, uses drive-time polygon instead of radius.

    Returns:
        Dict with 'attributes' containing aggregated segment data in same format
        as GeoEnrichment response for compatibility.
    """
    try:
        from arcgis.gis import GIS
    except ImportError:
        logger.error("arcgis package not installed. Run: pip install arcgis")
        return {"error": "arcgis package not installed"}

    try:
        # Connect to ArcGIS Online with credentials
        logger.info("Connecting to ArcGIS Online...")

        if ARCGIS_USERNAME and ARCGIS_PASSWORD:
            gis = GIS("https://www.arcgis.com", ARCGIS_USERNAME, ARCGIS_PASSWORD)
            logger.info(f"Connected as: {gis.users.me.username}")
        else:
            logger.warning("No ArcGIS credentials, trying anonymous access")
            gis = GIS()

        # Ensure URL ends with layer ID (e.g., /0)
        base_url = TAPESTRY_FEATURE_LAYER_URL.rstrip("/")
        if not base_url.split("/")[-1].isdigit():
            base_url = f"{base_url}/0"

        logger.info(f"Querying Tapestry feature layer: {base_url}")

        # Get token for direct REST API calls (more reliable spatial filtering)
        token = gis._con.token

        # Build query URL
        query_url = f"{base_url}/query"

        # Determine query method: drive-time polygon or radius
        if drive_time_minutes and drive_time_minutes > 0:
            # Use drive-time polygon (costs ~0.5 credits)
            logger.info(f"Using {drive_time_minutes}-minute drive-time area")
            polygon_dict = await generate_drive_time_polygon(gis, latitude, longitude, drive_time_minutes)

            if polygon_dict:
                # Query with polygon geometry via REST API
                params = {
                    'f': 'json',
                    'token': token,
                    'where': '1=1',
                    'geometry': json.dumps(polygon_dict),
                    'geometryType': 'esriGeometryPolygon',
                    'spatialRel': 'esriSpatialRelIntersects',
                    'outFields': '*',
                    'returnGeometry': 'false',
                    'resultRecordCount': 5000
                }
            else:
                logger.warning("Failed to generate drive-time polygon, falling back to radius")
                # Fall back to radius query
                buffer_meters = int(buffer_miles * 1609.34)
                params = {
                    'f': 'json',
                    'token': token,
                    'where': '1=1',
                    'geometry': json.dumps({'x': longitude, 'y': latitude, 'spatialReference': {'wkid': 4326}}),
                    'geometryType': 'esriGeometryPoint',
                    'spatialRel': 'esriSpatialRelIntersects',
                    'distance': buffer_meters,
                    'units': 'esriSRUnit_Meter',
                    'outFields': '*',
                    'returnGeometry': 'false',
                    'resultRecordCount': 5000
                }
        else:
            # Use radius query (FREE - no credits)
            logger.info(f"Using {buffer_miles}-mile radius (FREE)")
            buffer_meters = int(buffer_miles * 1609.34)

            params = {
                'f': 'json',
                'token': token,
                'where': '1=1',
                'geometry': json.dumps({'x': longitude, 'y': latitude, 'spatialReference': {'wkid': 4326}}),
                'geometryType': 'esriGeometryPoint',
                'spatialRel': 'esriSpatialRelIntersects',
                'distance': buffer_meters,
                'units': 'esriSRUnit_Meter',
                'outFields': '*',
                'returnGeometry': 'false',
                'resultRecordCount': 5000
            }

        # Execute query via REST API
        # Use POST for polygon queries (large geometries exceed URL length limits)
        async with aiohttp.ClientSession() as session:
            if drive_time_minutes and drive_time_minutes > 0 and polygon_dict:
                # POST for large polygon geometries
                async with session.post(query_url, data=params, ssl=False, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    data = await response.json()
            else:
                # GET for simple point queries
                async with session.get(query_url, params=params, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    data = await response.json()

        if 'error' in data:
            logger.error(f"Query error: {data['error']}")
            return {"error": data['error'].get('message', str(data['error']))}

        features = data.get('features', [])
        logger.info(f"Feature layer returned {len(features)} block groups")

        if not features:
            logger.warning("No block groups found in area")
            return {"attributes": {}}

        # Convert to expected format for aggregation
        feature_dicts = [{"attributes": f.get("attributes", {})} for f in features]

        # Fetch household data for accurate weighting
        # Use the same buffer/polygon to get tract-level household counts
        tract_households = await fetch_household_data(
            token=token,
            latitude=latitude,
            longitude=longitude,
            buffer_meters=buffer_meters if not (drive_time_minutes and polygon_dict) else 0,
            polygon_dict=polygon_dict if (drive_time_minutes and polygon_dict) else None
        )

        # Aggregate block group data to get segment distribution (with household weighting)
        return aggregate_tapestry_segments(feature_dicts, tract_households)

    except Exception as e:
        logger.error(f"Error querying Tapestry feature layer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def aggregate_tapestry_segments(features: List[Dict[str, Any]], tract_households: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Aggregate block group Tapestry data to calculate top 5 segments.

    Each block group has a dominant Tapestry segment. This function:
    1. Counts households per segment across all block groups
    2. Calculates percentage distribution (weighted by actual household counts if available)
    3. Returns top 5 segments in GeoEnrichment-compatible format

    Args:
        features: List of block group features from feature layer query
        tract_households: Optional dict mapping tract GEOID to household count for weighting

    Returns:
        Dict with 'attributes' matching GeoEnrichment response format
    """
    from collections import defaultdict

    # Detect field names from first feature
    if not features:
        return {"attributes": {}}

    sample_attrs = features[0].get("attributes", {})
    field_names = list(sample_attrs.keys())

    # Common field name patterns for Tapestry layers
    # Detect segment code field
    segment_code_field = None
    segment_name_field = None
    household_field = None

    # Try to find segment code field
    # 2025 layer uses "TapestryHouseholds_THHSCODE" for segment code
    code_patterns = [
        "TapestryHouseholds_THHSCODE", "THHSCODE",  # 2025 hosted layer format
        "Dominant Tapestry Seg", "Dominant_Tapestry_Seg",
        "TAPSEGCODE", "TSEGCODE", "DOMTAP", "DomTap", "TapCode",
        "TAPESTRY_CODE", "Tapestry_Code", "SEGMENT_CODE", "SegmentCode",
        "DOM_SEG", "DomSeg", "DOMINANT_TAPESTRY"
    ]
    for pattern in code_patterns:
        for field in field_names:
            if pattern.lower() in field.lower():
                segment_code_field = field
                break
        if segment_code_field:
            break

    # Try to find segment name field
    # 2025 layer uses "TGTapestry2025SegmentName" for segment name
    name_patterns = [
        "TGTapestry2025SegmentName", "Tapestry2025SegmentName",  # 2025 hosted layer format
        "Tapestry Segment Name", "Tapestry_Segment_Name",
        "TAPSEGNAM", "TSEGNAM", "TAPNAME", "TapName", "TAPESTRY_NAME",
        "Tapestry_Name", "SEGMENT_NAME", "SegmentName", "DOM_NAME", "DomName"
    ]
    for pattern in name_patterns:
        for field in field_names:
            if pattern.lower() in field.lower():
                segment_name_field = field
                break
        if segment_name_field:
            break

    # Try to find household count field
    hh_patterns = [
        "TOTHH", "TotHH", "HH_TOTAL", "HOUSEHOLDS", "Households",
        "HH_COUNT", "HHCount", "HHCNT", "TOTALHH", "TotalHH",
        "Total_Households", "Household_Count", "TapestryHouseholds"
    ]
    for pattern in hh_patterns:
        for field in field_names:
            if pattern.lower() in field.lower() and "code" not in field.lower():
                household_field = field
                break
        if household_field:
            break

    logger.info(f"Detected fields - Code: {segment_code_field}, Name: {segment_name_field}, HH: {household_field}")
    logger.info(f"Available fields: {field_names[:15]}...")  # Log first 15 fields

    if not segment_code_field:
        logger.error(f"Could not find segment code field. Available fields: {field_names}")
        return {"attributes": {}, "error": "Could not detect Tapestry segment code field"}

    # Try to find GEOID field for household weighting
    geoid_field = None
    for pattern in ['GEOID', 'GeoID', 'FIPS', 'GEOID10', 'GEOID20']:
        for field in field_names:
            if field.upper() == pattern.upper():
                geoid_field = field
                break
        if geoid_field:
            break

    # Determine weighting method
    use_tract_households = tract_households and len(tract_households) > 0 and geoid_field
    using_block_group_count = household_field is None and not use_tract_households

    if use_tract_households:
        logger.info(f"Using tract household data for weighting ({len(tract_households)} tracts)")
        # Count block groups per tract for fair distribution
        tract_bg_counts = defaultdict(int)
        for feature in features:
            attrs = feature.get("attributes", {})
            geoid = str(attrs.get(geoid_field, ''))
            if len(geoid) >= 11:
                tract_geoid = geoid[:11]  # Extract tract portion
                tract_bg_counts[tract_geoid] += 1
    elif household_field:
        logger.info(f"Using household field '{household_field}' for weighting")
    else:
        logger.info("No household data - using block group count as proxy")

    # Aggregate by segment
    segment_counts = defaultdict(float)  # households (can be fractional for distribution)
    segment_names = {}
    total_count = 0

    for feature in features:
        attrs = feature.get("attributes", {})

        seg_code = attrs.get(segment_code_field)
        if not seg_code:
            continue

        # Normalize segment code (e.g., "D2", "L1", etc.)
        seg_code = str(seg_code).strip()

        # Get count value based on weighting method
        count_val = 1.0  # Default: count each block group as 1

        if use_tract_households:
            # Use tract household count divided by block groups in that tract
            geoid = str(attrs.get(geoid_field, ''))
            if len(geoid) >= 11:
                tract_geoid = geoid[:11]
                tract_hh = tract_households.get(tract_geoid, 0)
                bg_count = tract_bg_counts.get(tract_geoid, 1)
                count_val = tract_hh / bg_count if bg_count > 0 else 0
        elif household_field:
            hh_val = attrs.get(household_field)
            if hh_val is not None:
                try:
                    count_val = float(hh_val)
                except (ValueError, TypeError):
                    count_val = 1.0

        segment_counts[seg_code] += count_val
        total_count += count_val

        # Store segment name if available
        if segment_name_field and seg_code not in segment_names:
            seg_name = attrs.get(segment_name_field)
            if seg_name:
                seg_name_str = str(seg_name).strip()
                # Handle format like "D2: Trendsetters" - extract just the name part
                if ": " in seg_name_str:
                    seg_name_str = seg_name_str.split(": ", 1)[1]
                segment_names[seg_code] = seg_name_str

    if use_tract_households:
        count_type = "households (tract-weighted)"
    elif household_field:
        count_type = "households"
    else:
        count_type = "block groups"
    logger.info(f"Aggregated {len(segment_counts)} unique segments, {total_count:,.0f} total {count_type}")

    if not segment_counts:
        return {"attributes": {}}

    # Sort segments by count (descending) and get top 5
    sorted_segments = sorted(
        segment_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Build attributes dict in GeoEnrichment-compatible format
    # Use total_count as TOTHH for compatibility (represents households or block groups)
    attributes = {
        "TOTHH": int(round(total_count)),  # Round to nearest integer for display
    }

    # Get segment profiles from knowledge base for names
    service = get_tapestry_service()

    for i, (seg_code, count) in enumerate(sorted_segments, 1):
        pct = (count / total_count * 100) if total_count > 0 else 0

        # Get segment name - try from data first, then knowledge base
        seg_name = segment_names.get(seg_code, "")
        if not seg_name:
            profile = service.get_segment(seg_code)
            if profile:
                seg_name = profile.get("name", f"Segment {seg_code}")
            else:
                seg_name = f"Segment {seg_code}"

        # Set fields in format compatible with existing parsing logic
        attributes[f"TSEGCODE{i}"] = seg_code
        attributes[f"TSEGNAM{i}"] = seg_name
        attributes[f"TSEGPCT{i}"] = round(pct, 2)
        attributes[f"TSEGHH{i}"] = int(count)  # Household count for this segment

        logger.info(f"Top {i}: {seg_code} - {seg_name} ({pct:.1f}%, {count:,.0f} {count_type})")

    return {"attributes": attributes}


def _parse_money_string(value: str) -> float | None:
    """Parse money string like '$45,000' to float."""
    if not value or value == "N/A":
        return None
    try:
        # Remove $, commas, and other non-numeric characters except decimal
        cleaned = ''.join(c for c in str(value) if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _parse_number_string(value: str) -> float | None:
    """Parse number string to float."""
    if not value or value == "N/A":
        return None
    try:
        cleaned = ''.join(c for c in str(value) if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _parse_percentage_string(value: str) -> float | None:
    """Parse percentage string like '65%' to decimal (0.65)."""
    if not value or value == "N/A":
        return None
    try:
        cleaned = ''.join(c for c in str(value) if c.isdigit() or c == '.')
        if cleaned:
            pct = float(cleaned)
            return pct / 100.0 if pct > 1 else pct
        return None
    except (ValueError, TypeError):
        return None


def enrich_segment_with_profile(segment: SegmentData) -> SegmentData:
    """Enrich segment with data from local knowledge base."""
    service = get_tapestry_service()
    profile = service.get_segment(segment.code)

    if profile:
        segment.description = profile.get("overview", profile.get("description", ""))
        segment.lifemode_group = profile.get("lifemode_name", profile.get("lifemode_group", ""))
        segment.urbanization = profile.get("urbanization", "")
        segment.characteristics = profile.get("characteristics", [])[:5]

        # Demographics - check both nested structure and top-level fields
        demographics = profile.get("demographics", {})
        if demographics:
            # Nested structure
            segment.median_age = demographics.get("median_age")
            segment.median_income = demographics.get("median_household_income")
            segment.median_net_worth = demographics.get("median_net_worth")
            segment.homeownership_rate = demographics.get("homeownership_rate")
        else:
            # Top-level fields (Tapestry 2025 JSON format) - parse string values
            segment.median_age = _parse_number_string(profile.get("median_age"))
            segment.median_income = _parse_money_string(profile.get("median_income"))
            segment.median_net_worth = _parse_money_string(profile.get("median_net_worth"))
            segment.homeownership_rate = _parse_percentage_string(profile.get("homeownership_rate"))

    # Set color
    segment.color = get_segment_color(segment.code)

    return segment


async def generate_segment_insight(segment: SegmentData, business_type: str = "") -> str:
    """Generate a 50-word actionable insight for a specific segment."""
    try:
        client = get_genai_client()

        business_context = f" for a {business_type}" if business_type else ""

        prompt = f"""Generate exactly 50 words of actionable marketing advice{business_context} targeting the "{segment.name}" ({segment.code}) tapestry segment.

Segment Profile:
- LifeMode: {segment.lifemode_group}
- Median Age: {segment.median_age or 'N/A'}
- Median Income: {f'${segment.median_income:,.0f}' if segment.median_income else 'N/A'}
- Homeownership: {f'{segment.homeownership_rate * 100:.0f}%' if segment.homeownership_rate else 'N/A'}
- Key traits: {', '.join(segment.characteristics[:3]) if segment.characteristics else 'N/A'}

Be specific and practical. Include one concrete channel or tactic recommendation.
Return ONLY the 50-word insight, no introduction or labels."""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # Fast model for quick insights
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
        )

        text = ""
        for c in response.candidates or []:
            if c.content:
                for p in c.content.parts or []:
                    if hasattr(p, "text"):
                        text += p.text

        return text.strip()

    except Exception as e:
        logger.error(f"Error generating segment insight: {e}")
        return f"Target {segment.name} customers with messaging that resonates with their lifestyle and preferences."


async def generate_business_insight(segments: List[SegmentData], business_type: str = "") -> str:
    """Generate a ~120 word business insight based on top segments."""
    try:
        client = get_genai_client()

        business_context = f" for a {business_type}" if business_type else ""

        # Format segment summary
        segment_summary = "\n".join([
            f"- {s.name} ({s.code}): {s.percentage:.1f}% - {s.lifemode_group}"
            for s in segments[:5]
        ])

        prompt = f"""Based on the top 5 lifestyle segments{business_context}, write a cohesive 120-word paragraph with specific, actionable marketing tactics.

TOP 5 LIFESTYLE SEGMENTS:
{segment_summary}

Cover these areas in your recommendations:
1. Marketing channels (be specific - which platforms, media types)
2. Messaging and tone recommendations
3. Product/service focus areas
4. Community engagement tactics
5. Digital strategy suggestions

Write as flowing prose, not bullet points. Wrap 2-3 key phrases in <strong> tags for emphasis.
Return ONLY the paragraph, no introduction or labels."""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # Fast model for quick insights
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
        )

        text = ""
        for c in response.candidates or []:
            if c.content:
                for p in c.content.parts or []:
                    if hasattr(p, "text"):
                        text += p.text

        return text.strip()

    except Exception as e:
        logger.error(f"Error generating business insight: {e}")
        return "Analyze the segment mix to develop targeted marketing strategies that resonate with your local customer base."


async def get_lifestyle_analysis(
    latitude: float,
    longitude: float,
    address: str = "",
    business_type: str = "",
    buffer_miles: float = 1.0,
    drive_time_minutes: int = None,
    tool_context: ToolContext = None,
) -> str:
    """
    Get comprehensive lifestyle/tapestry analysis for a location.

    Returns top 5 segments with:
    - Segment name, code, and percentage
    - Demographics (age, income, net worth, homeownership)
    - AI-generated insights per segment
    - Overall business recommendations

    Args:
        latitude: Location latitude
        longitude: Location longitude
        address: Optional address string for display
        business_type: Optional business type for tailored insights
        buffer_miles: Radius in miles (default 1.0). Used if drive_time_minutes is not provided.
        drive_time_minutes: Optional drive time in minutes (e.g., 15, 30, 60). If provided, uses drive-time polygon instead of radius. Costs ~0.5 credits.
        tool_context: ADK tool context

    Returns:
        Formatted lifestyle analysis report
    """
    from datetime import datetime

    area_desc = f"{drive_time_minutes}-minute drive time" if drive_time_minutes else f"{buffer_miles}-mile radius"
    logger.info(f"Starting lifestyle analysis for ({latitude}, {longitude}) - {area_desc}")

    # Step 1: Fetch tapestry data
    tapestry_data = await fetch_tapestry_data(latitude, longitude, buffer_miles, drive_time_minutes)

    if "error" in tapestry_data:
        return json.dumps({
            "status": "error",
            "message": f"Failed to fetch tapestry data: {tapestry_data['error']}"
        })

    attributes = tapestry_data.get("attributes", {})

    if not attributes:
        return json.dumps({
            "status": "error",
            "message": "No tapestry data available for this location"
        })

    # Step 2: Parse top 5 segments
    segments: List[SegmentData] = []

    # Helper to get attribute with multiple possible names
    def get_attr(*keys: str):
        for key in keys:
            val = attributes.get(key)
            if val is not None:
                return val
        return None

    total_households = get_attr("TOTHH", "Tapestry.TOTHH", "TOTHH_CY") or 0

    # Parse segments - try multiple naming conventions
    # Esri tapestry data collection uses different field names depending on version
    for i in range(1, 6):
        # Try multiple field name formats for each segment
        name = get_attr(
            f"TSEGNAM{i}" if i > 0 else "TSEGNAM",  # TSEGNAM1, TSEGNAM2, etc.
            f"TOP{i}NAME",                          # TOP1NAME, TOP2NAME, etc.
            f"Tapestry.TOP{i}NAME",
            f"TAPSEGNAM{i}" if i > 0 else "TAPSEGNAM",
        )
        code = get_attr(
            f"TSEGCODE{i}" if i > 0 else "TSEGCODE", # TSEGCODE1, TSEGCODE2, etc.
            f"TOP{i}CODE",                           # TOP1CODE, TOP2CODE, etc.
            f"Tapestry.TOP{i}CODE",
            f"TAPSEGCODE{i}" if i > 0 else "TAPSEGCODE",
        )
        pct = get_attr(
            f"TSEGPCT{i}" if i > 0 else "TSEGPCT",  # TSEGPCT1, TSEGPCT2, etc.
            f"TOP{i}PCT",                           # TOP1PCT, TOP2PCT, etc.
            f"Tapestry.TOP{i}PCT",
        ) or 0
        # Get actual household count if available (from feature layer aggregation)
        hh_count_attr = get_attr(
            f"TSEGHH{i}" if i > 0 else "TSEGHH",   # TSEGHH1, TSEGHH2, etc.
            f"TOP{i}HH",                           # TOP1HH, TOP2HH, etc.
        )

        # For segment 1, also try dominant segment fields if not found
        if i == 1 and not name:
            name = get_attr("TSEGNAME", "TAPSEGNAM", "DOMTAP_NAME", "Tapestry.TAPSEGNAM")
            code = get_attr("TSEGCODE", "TAPSEGCODE", "DOMTAP_CODE", "Tapestry.TAPSEGCODE")
            # Dominant segment often has 100% or unknown percentage
            pct = pct or 100.0

        if name and code:
            # Use actual household count if available, otherwise calculate from percentage
            if hh_count_attr is not None:
                hh_count = int(hh_count_attr)
            else:
                hh_count = int((float(pct) / 100) * total_households) if total_households else 0

            segment = SegmentData(
                code=str(code),
                name=str(name),
                percentage=float(pct),
                household_count=hh_count,
                lifemode_group="",
                urbanization=""
            )

            # Enrich with profile data
            segment = enrich_segment_with_profile(segment)
            segments.append(segment)
            logger.info(f"Parsed segment {i}: {code} - {name} ({pct}%)")

    if not segments:
        return json.dumps({
            "status": "error",
            "message": "Could not parse segment data from response"
        })

    # Step 3: Generate AI insights for all segments IN PARALLEL (major speed improvement)
    logger.info(f"Generating insights for {len(segments)} segments in parallel...")

    # Create tasks for all segment insights + business insight
    segment_tasks = [generate_segment_insight(segment, business_type) for segment in segments]
    business_task = generate_business_insight(segments, business_type)

    # Run all tasks in parallel
    all_results = await asyncio.gather(*segment_tasks, business_task, return_exceptions=True)

    # Assign segment insights (first N results)
    for i, segment in enumerate(segments):
        result = all_results[i]
        if isinstance(result, Exception):
            logger.error(f"Error generating insight for segment {segment.code}: {result}")
            segment.insight = f"Target {segment.name} customers with messaging that resonates with their lifestyle."
        else:
            segment.insight = result

    # Business insight is the last result
    business_result = all_results[-1]
    if isinstance(business_result, Exception):
        logger.error(f"Error generating business insight: {business_result}")
        business_insight = "Analyze the segment mix to develop targeted marketing strategies."
    else:
        business_insight = business_result

    # Step 5: Build report
    report = LifestyleReport(
        address=address or f"{latitude:.6f}, {longitude:.6f}",
        latitude=latitude,
        longitude=longitude,
        total_households=total_households,
        segments=segments,
        business_insight=business_insight,
        generated_at=datetime.now().isoformat(),
        buffer_miles=buffer_miles if not drive_time_minutes else None,
        drive_time_minutes=drive_time_minutes
    )

    # Step 6: Emit operations to frontend (map + report)
    connection_id = tool_context.state.get("connection_id") if tool_context else None

    if connection_id:
        # Convert to dict for JSON serialization
        report_dict = {
            "address": report.address,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "totalHouseholds": report.total_households,
            "businessInsight": report.business_insight,
            "generatedAt": report.generated_at,
            # Include buffer/drive-time info for map visualization
            "bufferMiles": buffer_miles if not drive_time_minutes else None,
            "driveTimeMinutes": drive_time_minutes,
            "segments": [
                {
                    "code": s.code,
                    "name": s.name,
                    "percentage": s.percentage,
                    "householdCount": s.household_count,
                    "lifemodeGroup": s.lifemode_group,
                    "urbanization": s.urbanization,
                    "medianAge": s.median_age,
                    "medianIncome": s.median_income,
                    "medianNetWorth": s.median_net_worth,
                    "homeownershipRate": s.homeownership_rate,
                    "description": s.description,
                    "characteristics": s.characteristics,
                    "color": s.color,
                    "insight": s.insight
                }
                for s in segments
            ]
        }

        # Calculate extent for zoom based on buffer size
        # Approximate degrees per mile at mid-latitudes: ~0.0145 lat, ~0.018 lon
        radius_for_extent = buffer_miles if not drive_time_minutes else (drive_time_minutes * 0.5)  # Rough estimate: 30mph average
        lat_offset = radius_for_extent * 0.0145 * 1.2  # Add 20% padding
        lon_offset = radius_for_extent * 0.018 * 1.2

        # Build multiple operations: zoom, pin, and report
        operations = [
            # 1. Zoom to location with buffer extent
            {
                "type": "ZOOM_TO_LOCATION",
                "payload": {
                    "extent": {
                        "xmin": longitude - lon_offset,
                        "ymin": latitude - lat_offset,
                        "xmax": longitude + lon_offset,
                        "ymax": latitude + lat_offset
                    },
                    "spatialReference": {"wkid": 4326}
                }
            },
            # 2. Add pin at the location
            {
                "type": "SUGGEST_PIN",
                "payload": {
                    "pins": [{
                        "id": f"lifestyle_{latitude}_{longitude}",
                        "latitude": latitude,
                        "longitude": longitude,
                        "title": report.address,
                        "description": f"Lifestyle Analysis: {segments[0].name if segments else 'N/A'}"
                    }]
                }
            },
            # 3. Show buffer/radius on map
            {
                "type": "SHOW_ANALYSIS_AREA",
                "payload": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "radiusMiles": buffer_miles if not drive_time_minutes else None,
                    "driveTimeMinutes": drive_time_minutes,
                    "color": "#3b82f6",  # Blue
                    "opacity": 0.2
                }
            },
            # 4. Lifestyle report data
            {
                "type": "LIFESTYLE_REPORT_GENERATED",
                "payload": report_dict
            }
        ]

        operation = {"operations": operations}

        try:
            await ws_manager.send_operations_data(connection_id, operation)
            logger.info(f"Sent {len(operations)} operations including LIFESTYLE_REPORT_GENERATED with {len(segments)} segments")
        except Exception as e:
            logger.error(f"Failed to send lifestyle operation: {e}")

    # Step 7: Store in session state for marketing agent
    if tool_context and tool_context.session:
        tool_context.session.state["lifestyle_analysis"] = {
            "segments": [{"code": s.code, "name": s.name, "percentage": s.percentage} for s in segments],
            "top_segment": segments[0].name if segments else None,
            "top_segment_code": segments[0].code if segments else None,
            "address": report.address,
            "buffer_miles": buffer_miles if not drive_time_minutes else None,
            "drive_time_minutes": drive_time_minutes
        }

    # Step 8: Return formatted text response
    return _format_lifestyle_report(report)


def _format_lifestyle_report(report: LifestyleReport) -> str:
    """Format the lifestyle report as readable markdown text with friendly business tone."""
    lines = []

    # Friendly introduction
    area_desc = ""
    if report.drive_time_minutes:
        area_desc = f"{report.drive_time_minutes}-minute drive time"
    elif report.buffer_miles:
        area_desc = f"{report.buffer_miles:.0f}-mile radius"
    else:
        area_desc = "selected area"

    lines.append("# 📊 Lifestyle Segmentation Analysis\n")
    lines.append(f"Great news! I've analyzed the lifestyle segments for **{report.address}** within a **{area_desc}**.\n")
    lines.append(f"This trade area includes **{report.total_households:,} households** across **{len(report.segments)} distinct lifestyle segments**.\n")

    # Summary Table - All segments at a glance
    lines.append("## 📋 Segment Overview\n")
    lines.append("| Rank | Segment | Code | Share | Households |")
    lines.append("|:----:|---------|:----:|------:|----------:|")

    for i, segment in enumerate(report.segments, 1):
        lines.append(f"| {i} | {segment.name} | {segment.code} | {segment.percentage:.1f}% | {segment.household_count:,} |")

    lines.append("")

    # Top 5 Detailed Analysis
    lines.append("## 🎯 Top 5 Segments - Detailed Analysis\n")
    lines.append("Here's what you need to know about your dominant customer segments:\n")

    for i, segment in enumerate(report.segments[:5], 1):
        # Segment header with visual bar
        max_pct = max(s.percentage for s in report.segments) if report.segments else 1
        bar_width = int((segment.percentage / max_pct) * 15)
        bar = "▓" * bar_width + "░" * (15 - bar_width)

        lines.append(f"### {i}. {segment.name}")
        lines.append(f"**Code:** `{segment.code}` | **Share:** {segment.percentage:.1f}% | **Households:** {segment.household_count:,}")
        lines.append(f"`{bar}`\n")

        # Key demographics in a clean format
        demo_items = []
        if segment.median_age:
            demo_items.append(f"**Median Age:** {segment.median_age:.0f}")
        if segment.median_income:
            demo_items.append(f"**Household Income:** ${segment.median_income:,.0f}")
        if segment.median_net_worth:
            demo_items.append(f"**Net Worth:** ${segment.median_net_worth:,.0f}")
        if segment.homeownership_rate:
            demo_items.append(f"**Homeownership:** {segment.homeownership_rate * 100:.0f}%")

        if demo_items:
            lines.append("**Key Demographics:**")
            lines.append(" • " + " • ".join(demo_items))
            lines.append("")

        if segment.lifemode_group:
            lines.append(f"**LifeMode Group:** {segment.lifemode_group}\n")

        if segment.description:
            lines.append(f"*{segment.description[:250]}{'...' if len(segment.description) > 250 else ''}*\n")

        if segment.insight:
            lines.append(f"💡 **Targeting Strategy:** {segment.insight}\n")

        lines.append("---\n")

    # Business Strategy Section
    if report.business_insight:
        lines.append("## 📈 Strategic Recommendations\n")
        lines.append("Based on the segment composition of your trade area, here are my recommendations:\n")
        lines.append(report.business_insight)
        lines.append("")

    # Actionable Next Steps
    lines.append("## 🚀 What You Can Do With These Insights\n")
    lines.append("1. **Target Marketing:** Focus ad spend on channels preferred by your top segments")
    lines.append("2. **Product Mix:** Adjust inventory/services to match segment preferences")
    lines.append("3. **Store Experience:** Tailor in-store experience to dominant lifestyles")
    lines.append("4. **Competitive Analysis:** Compare your segments to competitor locations")
    lines.append("5. **Expansion Planning:** Find similar segment profiles in new markets\n")

    # CTA for marketing post
    lines.append("---\n")
    lines.append("💬 **Ready to take action?** Say *\"create a marketing post\"* and I'll generate targeted content for your top segments!")

    return "\n".join(lines)


# Create ADK FunctionTool
get_lifestyle_analysis_tool = FunctionTool(get_lifestyle_analysis)
