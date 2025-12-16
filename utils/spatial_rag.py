"""
Spatial-RAG Hybrid Search System

Combines spatial queries (location-based) with semantic RAG lookups for
intelligent location-aware responses. This enables queries like:
- "What's the demographic profile for downtown Dallas?"
- "Tell me about the consumer segments near 1600 Pennsylvania Ave"

The system:
1. Extracts location context from queries
2. Fetches spatial data (demographics, POI, tapestry)
3. Retrieves relevant knowledge from RAG
4. Combines into unified context for LLM responses
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.cache import (
    get_cached_demographics,
    cache_demographics,
    get_cached_geocode,
    cache_geocode,
)

logger = logging.getLogger(__name__)


class SpatialQueryType(Enum):
    """Types of spatial queries."""
    LOCATION_DEMOGRAPHICS = "demographics"
    LOCATION_POI = "poi"
    LOCATION_TAPESTRY = "tapestry"
    LOCATION_GENERAL = "general"
    ADDRESS_LOOKUP = "address"
    RADIUS_SEARCH = "radius"


@dataclass
class SpatialContext:
    """Container for spatial context data."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    radius_miles: float = 5.0
    query_type: SpatialQueryType = SpatialQueryType.LOCATION_GENERAL
    extracted_location: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SpatialRAGResult:
    """Result from Spatial-RAG hybrid search."""
    spatial_context: SpatialContext
    spatial_data: Dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    combined_context: str = ""
    success: bool = False
    error: Optional[str] = None


class SpatialRAGSearch:
    """
    Hybrid search combining spatial queries with RAG lookups.

    Usage:
        search = SpatialRAGSearch(gis_executor)
        result = await search.search(
            query="What's the demographic profile for downtown Dallas?",
            include_demographics=True,
            include_tapestry=True
        )
    """

    # Location extraction patterns
    LOCATION_PATTERNS = [
        # "in [location]", "at [location]", "near [location]"
        r'\b(?:in|at|near|around|for|of)\s+([A-Z][a-zA-Z\s,]+(?:,\s*[A-Z]{2})?)',
        # Addresses with numbers
        r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)[.,]?\s*[A-Za-z\s]*,?\s*[A-Z]{2}?\s*\d{0,5})',
        # ZIP codes
        r'\b(\d{5}(?:-\d{4})?)\b',
        # City, State format
        r'\b([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})\b',
    ]

    # Query type detection patterns
    QUERY_TYPE_PATTERNS = {
        SpatialQueryType.LOCATION_DEMOGRAPHICS: [
            r'\b(?:demographic|population|income|household|census|statistics)\b',
        ],
        SpatialQueryType.LOCATION_POI: [
            r'\b(?:restaurant|store|shop|business|poi|point\s+of\s+interest|nearby|places)\b',
        ],
        SpatialQueryType.LOCATION_TAPESTRY: [
            r'\b(?:tapestry|segment|consumer|lifestyle|profile|marketing)\b',
        ],
        SpatialQueryType.ADDRESS_LOOKUP: [
            r'\b(?:where\s+is|locate|find\s+address|geocode)\b',
        ],
        SpatialQueryType.RADIUS_SEARCH: [
            r'\b(?:within|radius|miles?|km|kilometers?|around)\b',
        ],
    }

    def __init__(self, gis_executor=None):
        """
        Initialize Spatial-RAG search.

        Args:
            gis_executor: GISToolExecutor instance for spatial queries
        """
        self.gis_executor = gis_executor

    def extract_location(self, query: str) -> SpatialContext:
        """
        Extract location information from a query.

        Args:
            query: User query text

        Returns:
            SpatialContext with extracted location info
        """
        context = SpatialContext()

        # Detect query type
        context.query_type = self._detect_query_type(query)

        # Extract radius if specified
        radius_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mile|mi|km|kilometer)', query, re.I)
        if radius_match:
            radius = float(radius_match.group(1))
            if 'km' in radius_match.group(0).lower():
                radius *= 0.621371  # Convert km to miles
            context.radius_miles = radius

        # Try each pattern to extract location
        for pattern in self.LOCATION_PATTERNS:
            match = re.search(pattern, query, re.I)
            if match:
                location = match.group(1).strip()
                # Clean up the location string
                location = re.sub(r'[.,]$', '', location)
                location = re.sub(r'\s+', ' ', location)

                if len(location) >= 3:  # Minimum location length
                    context.extracted_location = location
                    context.confidence = self._calculate_confidence(location)
                    break

        return context

    def _detect_query_type(self, query: str) -> SpatialQueryType:
        """Detect the type of spatial query."""
        query_lower = query.lower()

        for query_type, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type

        return SpatialQueryType.LOCATION_GENERAL

    def _calculate_confidence(self, location: str) -> float:
        """Calculate confidence score for extracted location."""
        confidence = 0.5

        # Higher confidence for addresses with numbers
        if re.search(r'\d+', location):
            confidence += 0.2

        # Higher confidence for city, state format
        if re.search(r',\s*[A-Z]{2}$', location):
            confidence += 0.2

        # Higher confidence for ZIP codes
        if re.search(r'\d{5}', location):
            confidence += 0.1

        return min(confidence, 1.0)

    async def geocode_location(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Geocode a location string to coordinates.

        Returns:
            Tuple of (latitude, longitude, formatted_address) or (None, None, None)
        """
        if not self.gis_executor:
            logger.warning("GIS executor not available for geocoding")
            return None, None, None

        try:
            # Check cache first
            cached = await get_cached_geocode(location)
            if cached and "data" in cached:
                candidates = cached["data"].get("candidates", [])
                if candidates:
                    best = candidates[0]
                    loc = best.get("location", {})
                    return loc.get("y"), loc.get("x"), best.get("address")

            # Geocode the location
            result = await self.gis_executor.get_coordinates_for_address(location, max_locations=1)

            if "data" in result:
                candidates = result["data"].get("candidates", [])
                if candidates:
                    best = candidates[0]
                    loc = best.get("location", {})
                    await cache_geocode(location, result)
                    return loc.get("y"), loc.get("x"), best.get("address")

            return None, None, None

        except Exception as e:
            logger.error(f"Error geocoding location: {e}")
            return None, None, None

    async def fetch_spatial_data(
        self,
        latitude: float,
        longitude: float,
        include_demographics: bool = True,
        include_poi: bool = False,
        include_tapestry: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch spatial data for a location.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            include_demographics: Include demographic data
            include_poi: Include POI data
            include_tapestry: Include tapestry segmentation

        Returns:
            Dictionary with spatial data
        """
        if not self.gis_executor:
            return {}

        spatial_data = {}
        tasks = []
        task_keys = []

        if include_demographics or include_tapestry:
            # Check cache
            cached = await get_cached_demographics(latitude, longitude)
            if cached:
                spatial_data["demographics"] = cached.get("general", {})
                spatial_data["tapestry"] = cached.get("tapestry", {})
            else:
                tasks.append(self.gis_executor.get_esri_geoenrich_data(latitude, longitude))
                task_keys.append("geoenrich")

        if include_poi:
            tasks.append(self.gis_executor.get_esri_poi_data(latitude, longitude))
            task_keys.append("poi")

        # Execute tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for key, result in zip(task_keys, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {key}: {result}")
                    continue

                if key == "geoenrich":
                    spatial_data["demographics"] = result.get("general", {})
                    spatial_data["tapestry"] = result.get("tapestry", {})
                elif key == "poi":
                    spatial_data["poi"] = result

        return spatial_data

    def format_spatial_context(self, spatial_data: Dict[str, Any], address: str = None) -> str:
        """
        Format spatial data into context string for LLM.

        Args:
            spatial_data: Dictionary with spatial data
            address: Formatted address string

        Returns:
            Formatted context string
        """
        parts = []

        if address:
            parts.append(f"Location: {address}")

        # Format demographics
        if "demographics" in spatial_data:
            demo = spatial_data["demographics"]
            features = demo.get("features", [])
            if features:
                attrs = features[0].get("attributes", {})
                demo_parts = []

                # Map common field names to readable labels
                field_labels = {
                    "TOTPOP_CY": "Total Population",
                    "TOTHH_CY": "Total Households",
                    "MEDHINC_CY": "Median Household Income",
                    "POPDENS_CY": "Population Density",
                    "MEDNW_CY": "Median Net Worth",
                }

                for field, label in field_labels.items():
                    if field in attrs and attrs[field]:
                        value = attrs[field]
                        if "Income" in label or "Worth" in label:
                            demo_parts.append(f"{label}: ${value:,.0f}")
                        elif "Density" in label:
                            demo_parts.append(f"{label}: {value:,.1f}/sq mi")
                        else:
                            demo_parts.append(f"{label}: {value:,.0f}")

                if demo_parts:
                    parts.append("Demographics: " + "; ".join(demo_parts))

        # Format tapestry
        if "tapestry" in spatial_data:
            tap = spatial_data["tapestry"]
            features = tap.get("features", [])
            if features:
                attrs = features[0].get("attributes", {})
                seg_name = attrs.get("TSEGNAME", "")
                life_name = attrs.get("TLIFENAME", "")

                if seg_name:
                    parts.append(f"Tapestry Segment: {seg_name}")
                if life_name:
                    parts.append(f"LifeMode Group: {life_name}")

        # Format POI summary
        if "poi" in spatial_data:
            poi = spatial_data["poi"]
            results = poi.get("results", [])
            if results:
                categories = {}
                for place in results[:20]:  # Limit to top 20
                    cat = place.get("categories", [{}])[0].get("label", "Other")
                    categories[cat] = categories.get(cat, 0) + 1

                if categories:
                    top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
                    poi_summary = ", ".join([f"{cat} ({count})" for cat, count in top_cats])
                    parts.append(f"Nearby POI: {poi_summary}")

        return "\n".join(parts)

    async def search(
        self,
        query: str,
        include_demographics: bool = True,
        include_poi: bool = False,
        include_tapestry: bool = True,
        rag_query_func=None
    ) -> SpatialRAGResult:
        """
        Perform hybrid spatial-RAG search.

        Args:
            query: User query text
            include_demographics: Include demographic data
            include_poi: Include POI data
            include_tapestry: Include tapestry segmentation
            rag_query_func: Optional async function to query RAG system

        Returns:
            SpatialRAGResult with combined context
        """
        result = SpatialRAGResult(spatial_context=SpatialContext())

        try:
            # Extract location from query
            context = self.extract_location(query)
            result.spatial_context = context

            if not context.extracted_location:
                result.error = "No location found in query"
                return result

            # Geocode the location
            lat, lon, address = await self.geocode_location(context.extracted_location)

            if lat is None or lon is None:
                result.error = f"Could not geocode location: {context.extracted_location}"
                return result

            context.latitude = lat
            context.longitude = lon
            context.address = address

            # Fetch spatial data
            spatial_data = await self.fetch_spatial_data(
                lat, lon,
                include_demographics=include_demographics,
                include_poi=include_poi,
                include_tapestry=include_tapestry
            )
            result.spatial_data = spatial_data

            # Format spatial context
            spatial_context_str = self.format_spatial_context(spatial_data, address)

            # Query RAG if function provided and tapestry data available
            rag_context = ""
            if rag_query_func and include_tapestry:
                tapestry = spatial_data.get("tapestry", {})
                features = tapestry.get("features", [])
                if features:
                    seg_name = features[0].get("attributes", {}).get("TSEGNAME", "")
                    if seg_name:
                        try:
                            rag_context = await rag_query_func(
                                f"Tell me about the {seg_name} tapestry segment"
                            )
                        except Exception as e:
                            logger.error(f"RAG query error: {e}")

            result.rag_context = rag_context

            # Combine contexts
            combined_parts = []
            if spatial_context_str:
                combined_parts.append(f"[Spatial Data]\n{spatial_context_str}")
            if rag_context:
                combined_parts.append(f"[Segment Insights]\n{rag_context}")

            result.combined_context = "\n\n".join(combined_parts)
            result.success = True

        except Exception as e:
            logger.error(f"Spatial-RAG search error: {e}")
            result.error = str(e)

        return result

    def should_use_spatial_search(self, query: str) -> bool:
        """
        Determine if a query should use spatial search.

        Args:
            query: User query text

        Returns:
            True if query appears to be location-related
        """
        # Check for location patterns
        for pattern in self.LOCATION_PATTERNS:
            if re.search(pattern, query, re.I):
                return True

        # Check for spatial keywords
        spatial_keywords = [
            'location', 'address', 'city', 'state', 'zip',
            'demographic', 'population', 'nearby', 'around',
            'tapestry', 'segment', 'area', 'region'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in spatial_keywords)


# Global instance
_spatial_rag_search = None


def get_spatial_rag_search(gis_executor=None) -> SpatialRAGSearch:
    """Get or create global SpatialRAGSearch instance."""
    global _spatial_rag_search
    if _spatial_rag_search is None or (gis_executor and _spatial_rag_search.gis_executor != gis_executor):
        _spatial_rag_search = SpatialRAGSearch(gis_executor)
    return _spatial_rag_search


async def spatial_rag_search(
    query: str,
    gis_executor=None,
    include_demographics: bool = True,
    include_poi: bool = False,
    include_tapestry: bool = True,
    rag_query_func=None
) -> SpatialRAGResult:
    """
    Convenience function for spatial-RAG search.

    Args:
        query: User query text
        gis_executor: GIS executor instance
        include_demographics: Include demographic data
        include_poi: Include POI data
        include_tapestry: Include tapestry data
        rag_query_func: Optional RAG query function

    Returns:
        SpatialRAGResult with search results
    """
    search = get_spatial_rag_search(gis_executor)
    return await search.search(
        query,
        include_demographics=include_demographics,
        include_poi=include_poi,
        include_tapestry=include_tapestry,
        rag_query_func=rag_query_func
    )
