"""
Spatial Intelligence Engine

Core module for understanding complex GIS queries, planning multi-step workflows,
executing spatial operations, and generating business insights.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY PARSER - Understands natural language spatial queries
# =============================================================================

class SpatialIntent(Enum):
    """Types of spatial queries the system can handle."""
    LOCATION_ANALYSIS = "location_analysis"           # "Analyze demographics for Dallas"
    TRADE_AREA_ANALYSIS = "trade_area_analysis"       # "15 min drive time analysis"
    SEGMENT_ANALYSIS = "segment_analysis"             # "What lifestyles are in this area"
    COMPARISON = "comparison"                         # "Compare Austin vs Houston"
    SITE_SELECTION = "site_selection"                 # "Best location for coffee shop"
    GAP_ANALYSIS = "gap_analysis"                     # "Market gaps in my trade area"
    LAYER_QUERY = "layer_query"                       # "What is I3 segment"
    SEGMENT_INFO = "segment_info"                     # "Tell me about Savvy Suburbanites"
    NAVIGATION = "navigation"                         # "Zoom to Dallas"
    GENERAL = "general"


@dataclass
class QueryEntities:
    """Extracted entities from a spatial query."""
    location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    distance_value: Optional[float] = None
    distance_unit: Optional[str] = None  # "minutes", "miles", "km"
    analysis_type: Optional[str] = None  # "drive_time", "radius", "polygon"
    layer_name: Optional[str] = None
    segment_code: Optional[str] = None
    segment_name: Optional[str] = None
    business_type: Optional[str] = None
    comparison_locations: Optional[List[str]] = None
    output_format: Optional[str] = None  # "top_5", "all", "summary"
    time_period: Optional[str] = None


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query."""
    original_query: str
    intent: SpatialIntent
    entities: QueryEntities
    confidence: float
    requires_location: bool = True
    requires_layer_data: bool = False
    suggested_tools: List[str] = field(default_factory=list)


class SpatialQueryParser:
    """
    Parse natural language queries into structured spatial queries.
    """

    # Intent detection patterns
    INTENT_PATTERNS = {
        SpatialIntent.TRADE_AREA_ANALYSIS: [
            r'\b(?:drive\s*time|driving\s*time|travel\s*time)\b',
            r'\b(?:trade\s*area|catchment|service\s*area)\b',
            r'\b\d+\s*(?:min|minute|minutes|hour|hours)\s*(?:drive|driving|travel)\b',
        ],
        SpatialIntent.SEGMENT_ANALYSIS: [
            r'\b(?:lifestyle|lifestyles|segment|segments|tapestry)\b.*\b(?:area|nearby|around|for)\b',
            r'\b(?:what|which)\s+(?:lifestyle|segment|tapestry)\b',
            r'\b(?:top|main|dominant)\s*(?:\d+)?\s*(?:lifestyle|segment)\b',
        ],
        SpatialIntent.SEGMENT_INFO: [
            r'\bwhat\s+is\s+(?:the\s+)?(?:segment\s+)?([A-Z]\d+|[A-Z]{2,})\b',
            r'\btell\s+me\s+about\s+(?:segment\s+)?([A-Z]\d+|[A-Z]{2,})\b',
            r'\bexplain\s+(?:the\s+)?([A-Z]\d+|[A-Z]{2,})\s*(?:segment)?\b',
            r'\b([A-Z]\d+)\s+segment\b',
        ],
        SpatialIntent.LOCATION_ANALYSIS: [
            r'\b(?:analyze|analysis|demographics|demographic|profile)\b.*\b(?:for|of|in|at)\b',
            r'\b(?:what|show)\b.*\b(?:demographics|population|income)\b',
        ],
        SpatialIntent.COMPARISON: [
            r'\bcompare\b',
            r'\bvs\.?\b',
            r'\bdifference\s+between\b',
            r'\b(?:versus|against)\b',
        ],
        SpatialIntent.GAP_ANALYSIS: [
            r'\bgap\s*(?:analysis)?\b',
            r'\bopportunit(?:y|ies)\b',
            r'\bunderserved\b',
            r'\bmarket\s+(?:gap|opportunity)\b',
        ],
        SpatialIntent.SITE_SELECTION: [
            r'\bbest\s+(?:location|site|place)\b',
            r'\boptimal\s+(?:location|site)\b',
            r'\bwhere\s+(?:should|to)\s+(?:open|locate|put)\b',
        ],
        SpatialIntent.LAYER_QUERY: [
            r'\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:data|features|values)\b',
            r'\bshow\s+me\s+(?:the\s+)?(?:data|features|layer)\b',
            r'\bquery\s+(?:the\s+)?layer\b',
        ],
        SpatialIntent.NAVIGATION: [
            r'\b(?:zoom|go|navigate|fly|take\s+me)\s+to\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:map\s+of|area)\b',
        ],
    }

    # Entity extraction patterns
    LOCATION_PATTERNS = [
        # Full address with street number
        r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)[.,]?\s*[A-Za-z\s]*,?\s*[A-Z]{2}?\s*\d{0,5})',
        # City, State format
        r'\b([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})\b',
        # ZIP code
        r'\b(\d{5}(?:-\d{4})?)\b',
        # "for/in/at [location]"
        r'\b(?:for|in|at|near|around)\s+([A-Z][a-zA-Z\s,]+?)(?:\s+for|\s+with|\s*$|\s*\.)',
    ]

    DISTANCE_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)\s*(?:drive|driving|travel)?',
        r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)\s*(?:drive|driving|travel)?',
        r'(\d+(?:\.\d+)?)\s*(?:mile|miles|mi)\s*(?:radius)?',
        r'(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometers)\s*(?:radius)?',
    ]

    SEGMENT_CODE_PATTERN = r'\b([A-Z]\d+|[A-Z]{2}\d*)\b'

    BUSINESS_TYPE_PATTERNS = [
        r'\b(?:for\s+(?:a\s+)?)?(?:my\s+)?(restaurant|cafe|coffee\s*shop|retail|store|gym|fitness|salon|spa|clinic|medical|dental|office|bank|hotel|gas\s*station|convenience)\b',
    ]

    def __init__(self, tapestry_data: Optional[Dict] = None):
        """Initialize parser with optional Tapestry segment data."""
        self.tapestry_data = tapestry_data or {}
        self.segment_names = self._build_segment_names()

    def _build_segment_names(self) -> Dict[str, str]:
        """Build a mapping of segment names to codes."""
        names = {}
        segments = self.tapestry_data.get("segments", {})
        for code, data in segments.items():
            name = data.get("name", "").lower()
            if name:
                names[name] = code
        return names

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured form.

        Args:
            query: Natural language query

        Returns:
            ParsedQuery with intent, entities, and metadata
        """
        entities = QueryEntities()

        # Detect intent
        intent, confidence = self._detect_intent(query)

        # Extract entities
        entities.location = self._extract_location(query)
        distance_val, distance_unit = self._extract_distance(query)
        entities.distance_value = distance_val
        entities.distance_unit = distance_unit
        entities.segment_code = self._extract_segment_code(query)
        entities.business_type = self._extract_business_type(query)
        entities.comparison_locations = self._extract_comparison_locations(query)

        # Determine analysis type
        if distance_unit in ["minutes", "hours"]:
            entities.analysis_type = "drive_time"
        elif distance_unit in ["miles", "km"]:
            entities.analysis_type = "radius"

        # Determine required tools
        suggested_tools = self._suggest_tools(intent, entities)

        # Determine if location is required
        requires_location = intent not in [
            SpatialIntent.SEGMENT_INFO,
            SpatialIntent.LAYER_QUERY,
        ]

        # Determine if layer data is needed
        requires_layer_data = intent in [
            SpatialIntent.SEGMENT_ANALYSIS,
            SpatialIntent.TRADE_AREA_ANALYSIS,
            SpatialIntent.GAP_ANALYSIS,
            SpatialIntent.LAYER_QUERY,
        ]

        return ParsedQuery(
            original_query=query,
            intent=intent,
            entities=entities,
            confidence=confidence,
            requires_location=requires_location,
            requires_layer_data=requires_layer_data,
            suggested_tools=suggested_tools,
        )

    def _detect_intent(self, query: str) -> Tuple[SpatialIntent, float]:
        """Detect the intent of the query."""
        query_lower = query.lower()

        best_intent = SpatialIntent.GENERAL
        best_score = 0.0

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score = 0.8 + (0.1 * len(patterns))  # Higher score for more pattern matches
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        return best_intent, min(best_score, 1.0)

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query."""
        for pattern in self.LOCATION_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up
                location = re.sub(r'[.,]$', '', location)
                location = re.sub(r'\s+', ' ', location)
                if len(location) >= 3:
                    return location
        return None

    def _extract_distance(self, query: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract distance value and unit from query."""
        query_lower = query.lower()

        for pattern in self.DISTANCE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))

                if 'min' in match.group(0):
                    return value, "minutes"
                elif 'hour' in match.group(0) or 'hr' in match.group(0):
                    return value * 60, "minutes"  # Convert to minutes
                elif 'mile' in match.group(0) or 'mi' in match.group(0):
                    return value, "miles"
                elif 'km' in match.group(0):
                    return value, "km"

        return None, None

    def _extract_segment_code(self, query: str) -> Optional[str]:
        """Extract Tapestry segment code from query."""
        match = re.search(self.SEGMENT_CODE_PATTERN, query)
        if match:
            return match.group(1)

        # Check for segment names
        query_lower = query.lower()
        for name, code in self.segment_names.items():
            if name in query_lower:
                return code

        return None

    def _extract_business_type(self, query: str) -> Optional[str]:
        """Extract business type from query."""
        for pattern in self.BUSINESS_TYPE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_comparison_locations(self, query: str) -> Optional[List[str]]:
        """Extract locations for comparison queries."""
        # Look for "compare X and Y" or "X vs Y" patterns
        patterns = [
            r'compare\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)(?:\s+for|\s*$|\s*\.)',
            r'(.+?)\s+vs\.?\s+(.+?)(?:\s+for|\s*$|\s*\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

        return None

    def _suggest_tools(self, intent: SpatialIntent, entities: QueryEntities) -> List[str]:
        """Suggest tools based on intent and entities."""
        tools = []

        if intent == SpatialIntent.NAVIGATION:
            tools = ["geocode_address", "zoom_to_location"]

        elif intent == SpatialIntent.SEGMENT_INFO:
            tools = ["get_segment_info"]

        elif intent == SpatialIntent.LOCATION_ANALYSIS:
            tools = ["geocode_address", "get_demographics", "get_tapestry"]

        elif intent == SpatialIntent.TRADE_AREA_ANALYSIS:
            tools = [
                "geocode_address",
                "create_trade_area",
                "ensure_layer_visible",
                "query_layer_features",
                "aggregate_segments",
                "generate_insights",
            ]

        elif intent == SpatialIntent.SEGMENT_ANALYSIS:
            tools = [
                "geocode_address",
                "ensure_layer_visible",
                "query_layer_features",
                "aggregate_segments",
                "generate_insights",
            ]

        elif intent == SpatialIntent.COMPARISON:
            tools = [
                "geocode_address",
                "get_demographics",
                "get_tapestry",
                "compare_locations",
            ]

        elif intent == SpatialIntent.GAP_ANALYSIS:
            tools = [
                "geocode_address",
                "create_trade_area",
                "query_layer_features",
                "aggregate_segments",
                "calculate_gap_analysis",
            ]

        return tools


# =============================================================================
# PLANNING ENGINE - Creates execution plans for multi-step workflows
# =============================================================================

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str
    action: str
    tool: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    output_key: str = ""
    description: str = ""


@dataclass
class ExecutionPlan:
    """Complete execution plan for a spatial query."""
    query: str
    intent: SpatialIntent
    steps: List[PlanStep]
    estimated_duration_seconds: float = 0.0
    required_layers: List[str] = field(default_factory=list)

    def get_execution_order(self) -> List[List[str]]:
        """Get steps grouped by execution order (parallel groups)."""
        # Build dependency graph
        remaining = {step.id for step in self.steps}
        completed = set()
        order = []

        while remaining:
            # Find steps with all dependencies satisfied
            ready = []
            for step in self.steps:
                if step.id in remaining:
                    if all(dep in completed for dep in step.depends_on):
                        ready.append(step.id)

            if not ready:
                # Circular dependency or error
                ready = list(remaining)[:1]

            order.append(ready)
            for step_id in ready:
                remaining.discard(step_id)
                completed.add(step_id)

        return order


class PlanningEngine:
    """
    Creates execution plans for spatial queries.
    """

    def __init__(self):
        self.plan_templates = self._build_plan_templates()

    def _build_plan_templates(self) -> Dict[SpatialIntent, Callable]:
        """Build plan templates for each intent type."""
        return {
            SpatialIntent.NAVIGATION: self._plan_navigation,
            SpatialIntent.SEGMENT_INFO: self._plan_segment_info,
            SpatialIntent.LOCATION_ANALYSIS: self._plan_location_analysis,
            SpatialIntent.TRADE_AREA_ANALYSIS: self._plan_trade_area_analysis,
            SpatialIntent.SEGMENT_ANALYSIS: self._plan_segment_analysis,
            SpatialIntent.COMPARISON: self._plan_comparison,
            SpatialIntent.GAP_ANALYSIS: self._plan_gap_analysis,
        }

    def create_plan(self, parsed_query: ParsedQuery) -> ExecutionPlan:
        """
        Create an execution plan for a parsed query.

        Args:
            parsed_query: ParsedQuery from the query parser

        Returns:
            ExecutionPlan with steps to execute
        """
        template_func = self.plan_templates.get(
            parsed_query.intent,
            self._plan_general
        )

        return template_func(parsed_query)

    def _plan_navigation(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for navigation queries."""
        steps = []

        if pq.entities.location:
            steps.append(PlanStep(
                id="geocode",
                action="geocode_address",
                tool="gis_tools.geocode",
                parameters={"address": pq.entities.location},
                output_key="location",
                description=f"Geocoding {pq.entities.location}"
            ))

            steps.append(PlanStep(
                id="zoom",
                action="zoom_to_location",
                tool="map_operations.zoom",
                parameters={"location": "${location}"},
                depends_on=["geocode"],
                description="Zooming to location"
            ))

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=2.0
        )

    def _plan_segment_info(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for segment information queries."""
        steps = [
            PlanStep(
                id="get_segment",
                action="get_segment_info",
                tool="knowledge.get_segment",
                parameters={"segment_code": pq.entities.segment_code},
                output_key="segment_info",
                description=f"Getting info for segment {pq.entities.segment_code}"
            ),
            PlanStep(
                id="format_response",
                action="format_segment_response",
                tool="response.format_segment",
                parameters={"segment_info": "${segment_info}"},
                depends_on=["get_segment"],
                description="Formatting response"
            )
        ]

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=1.0
        )

    def _plan_location_analysis(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for location analysis queries."""
        steps = [
            PlanStep(
                id="geocode",
                action="geocode_address",
                tool="gis_tools.geocode",
                parameters={"address": pq.entities.location},
                output_key="location",
                description=f"Geocoding {pq.entities.location}"
            ),
            PlanStep(
                id="zoom",
                action="zoom_to_location",
                tool="map_operations.zoom",
                parameters={"location": "${location}"},
                depends_on=["geocode"],
                description="Zooming to location"
            ),
            PlanStep(
                id="demographics",
                action="get_demographics",
                tool="gis_tools.geoenrich",
                parameters={"location": "${location}"},
                depends_on=["geocode"],
                output_key="demographics",
                description="Getting demographics"
            ),
            PlanStep(
                id="tapestry",
                action="get_tapestry",
                tool="gis_tools.geoenrich_tapestry",
                parameters={"location": "${location}"},
                depends_on=["geocode"],
                output_key="tapestry",
                description="Getting tapestry data"
            ),
            PlanStep(
                id="insights",
                action="generate_insights",
                tool="insights.location_analysis",
                parameters={
                    "demographics": "${demographics}",
                    "tapestry": "${tapestry}"
                },
                depends_on=["demographics", "tapestry"],
                output_key="insights",
                description="Generating insights"
            )
        ]

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=5.0
        )

    def _plan_trade_area_analysis(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for trade area analysis queries."""
        steps = [
            PlanStep(
                id="geocode",
                action="geocode_address",
                tool="gis_tools.geocode",
                parameters={"address": pq.entities.location},
                output_key="location",
                description=f"Geocoding {pq.entities.location}"
            ),
            PlanStep(
                id="zoom",
                action="zoom_to_location",
                tool="map_operations.zoom",
                parameters={"location": "${location}"},
                depends_on=["geocode"],
                description="Zooming to location"
            ),
            PlanStep(
                id="trade_area",
                action="create_trade_area",
                tool="gis_tools.service_area",
                parameters={
                    "location": "${location}",
                    "time_minutes": pq.entities.distance_value or 15,
                    "travel_mode": "driving"
                },
                depends_on=["geocode"],
                output_key="trade_area",
                description=f"Creating {pq.entities.distance_value or 15} min drive time"
            ),
            PlanStep(
                id="ensure_layer",
                action="ensure_layer_visible",
                tool="layer_tools.ensure_visible",
                parameters={
                    "layer_name": "Tapestry Segmentation 2025",
                    "visible": True
                },
                description="Ensuring Tapestry layer is visible"
            ),
            PlanStep(
                id="query_segments",
                action="query_layer_features",
                tool="layer_tools.spatial_query",
                parameters={
                    "layer_name": "Tapestry Segmentation 2025",
                    "geometry": "${trade_area}",
                    "out_fields": ["TSEGCODE", "TSEGNAME", "TLIFENAME", "THHBASE"]
                },
                depends_on=["trade_area", "ensure_layer"],
                output_key="segment_features",
                description="Querying Tapestry segments in trade area"
            ),
            PlanStep(
                id="aggregate",
                action="aggregate_segments",
                tool="analytics.aggregate",
                parameters={
                    "features": "${segment_features}",
                    "group_by": "TSEGCODE",
                    "value_field": "THHBASE"
                },
                depends_on=["query_segments"],
                output_key="segment_stats",
                description="Aggregating segment data"
            ),
            PlanStep(
                id="top_5",
                action="get_top_segments",
                tool="analytics.rank",
                parameters={
                    "data": "${segment_stats}",
                    "n": 5
                },
                depends_on=["aggregate"],
                output_key="top_segments",
                description="Getting top 5 segments"
            ),
            PlanStep(
                id="enrich",
                action="enrich_with_insights",
                tool="knowledge.enrich_segments",
                parameters={"segments": "${top_segments}"},
                depends_on=["top_5"],
                output_key="enriched_segments",
                description="Enriching with segment insights"
            ),
            PlanStep(
                id="response",
                action="format_response",
                tool="response.trade_area_analysis",
                parameters={
                    "location": "${location}",
                    "trade_area": "${trade_area}",
                    "segments": "${enriched_segments}"
                },
                depends_on=["enrich"],
                description="Generating response"
            )
        ]

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=15.0,
            required_layers=["Tapestry Segmentation 2025"]
        )

    def _plan_segment_analysis(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for segment analysis in current view."""
        steps = [
            PlanStep(
                id="ensure_layer",
                action="ensure_layer_visible",
                tool="layer_tools.ensure_visible",
                parameters={
                    "layer_name": "Tapestry Segmentation 2025",
                    "visible": True
                },
                description="Ensuring Tapestry layer is visible"
            ),
            PlanStep(
                id="query_extent",
                action="query_visible_features",
                tool="layer_tools.query_extent",
                parameters={
                    "layer_name": "Tapestry Segmentation 2025",
                    "out_fields": ["TSEGCODE", "TSEGNAME", "TLIFENAME", "THHBASE"]
                },
                depends_on=["ensure_layer"],
                output_key="segment_features",
                description="Querying segments in current view"
            ),
            PlanStep(
                id="aggregate",
                action="aggregate_segments",
                tool="analytics.aggregate",
                parameters={
                    "features": "${segment_features}",
                    "group_by": "TSEGCODE",
                    "value_field": "THHBASE"
                },
                depends_on=["query_extent"],
                output_key="segment_stats",
                description="Aggregating segment data"
            ),
            PlanStep(
                id="top_5",
                action="get_top_segments",
                tool="analytics.rank",
                parameters={
                    "data": "${segment_stats}",
                    "n": 5
                },
                depends_on=["aggregate"],
                output_key="top_segments",
                description="Getting top 5 segments"
            ),
            PlanStep(
                id="enrich",
                action="enrich_with_insights",
                tool="knowledge.enrich_segments",
                parameters={"segments": "${top_segments}"},
                depends_on=["top_5"],
                output_key="enriched_segments",
                description="Enriching with segment insights"
            ),
            PlanStep(
                id="response",
                action="format_response",
                tool="response.segment_analysis",
                parameters={"segments": "${enriched_segments}"},
                depends_on=["enrich"],
                description="Generating response"
            )
        ]

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=10.0,
            required_layers=["Tapestry Segmentation 2025"]
        )

    def _plan_comparison(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for location comparison queries."""
        locations = pq.entities.comparison_locations or []
        if len(locations) < 2:
            return self._plan_general(pq)

        steps = [
            # Geocode both locations
            PlanStep(
                id="geocode_1",
                action="geocode_address",
                tool="gis_tools.geocode",
                parameters={"address": locations[0]},
                output_key="location_1",
                description=f"Geocoding {locations[0]}"
            ),
            PlanStep(
                id="geocode_2",
                action="geocode_address",
                tool="gis_tools.geocode",
                parameters={"address": locations[1]},
                output_key="location_2",
                description=f"Geocoding {locations[1]}"
            ),
            # Get data for both
            PlanStep(
                id="data_1",
                action="get_location_data",
                tool="gis_tools.geoenrich",
                parameters={"location": "${location_1}"},
                depends_on=["geocode_1"],
                output_key="data_1",
                description="Getting data for location 1"
            ),
            PlanStep(
                id="data_2",
                action="get_location_data",
                tool="gis_tools.geoenrich",
                parameters={"location": "${location_2}"},
                depends_on=["geocode_2"],
                output_key="data_2",
                description="Getting data for location 2"
            ),
            # Compare
            PlanStep(
                id="compare",
                action="compare_locations",
                tool="analytics.compare",
                parameters={
                    "location_1": "${data_1}",
                    "location_2": "${data_2}"
                },
                depends_on=["data_1", "data_2"],
                output_key="comparison",
                description="Comparing locations"
            ),
            PlanStep(
                id="response",
                action="format_response",
                tool="response.comparison",
                parameters={"comparison": "${comparison}"},
                depends_on=["compare"],
                description="Generating comparison response"
            )
        ]

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=8.0
        )

    def _plan_gap_analysis(self, pq: ParsedQuery) -> ExecutionPlan:
        """Plan for gap analysis queries."""
        # Similar to trade area but with customer data overlay
        steps = self._plan_trade_area_analysis(pq).steps

        # Add customer data steps
        steps.insert(-1, PlanStep(
            id="customer_data",
            action="get_customer_segments",
            tool="data.customer_segments",
            parameters={"trade_area": "${trade_area}"},
            depends_on=["trade_area"],
            output_key="customer_segments",
            description="Getting customer segment data"
        ))

        steps.insert(-1, PlanStep(
            id="gap_calc",
            action="calculate_gap",
            tool="analytics.gap_analysis",
            parameters={
                "market_segments": "${segment_stats}",
                "customer_segments": "${customer_segments}"
            },
            depends_on=["aggregate", "customer_data"],
            output_key="gap_analysis",
            description="Calculating market gaps"
        ))

        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=steps,
            estimated_duration_seconds=20.0,
            required_layers=["Tapestry Segmentation 2025"]
        )

    def _plan_general(self, pq: ParsedQuery) -> ExecutionPlan:
        """Fallback plan for general queries."""
        return ExecutionPlan(
            query=pq.original_query,
            intent=pq.intent,
            steps=[],
            estimated_duration_seconds=1.0
        )


# =============================================================================
# TAPESTRY KNOWLEDGE BASE - Segment information and insights
# =============================================================================

class TapestryKnowledgeBase:
    """
    Knowledge base for Tapestry segment information.
    """

    def __init__(self, data_path: Optional[str] = None):
        """Initialize with data file path."""
        self.data_path = data_path or str(
            Path(__file__).parent.parent / "data" / "tapestry_segments_2025.json"
        )
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load Tapestry data from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading Tapestry data: {e}")
            return {"segments": {}, "lifemode_groups_reference": {}}

    def get_segment(self, segment_code: str) -> Optional[Dict]:
        """Get full segment information by code."""
        return self.data.get("segments", {}).get(segment_code)

    def get_segment_by_name(self, name: str) -> Optional[Dict]:
        """Get segment by name (case-insensitive)."""
        name_lower = name.lower()
        for code, segment in self.data.get("segments", {}).items():
            if segment.get("name", "").lower() == name_lower:
                return segment
        return None

    def get_lifemode_group(self, group_code: str) -> Optional[Dict]:
        """Get LifeMode group information."""
        return self.data.get("lifemode_groups_reference", {}).get(group_code)

    def search_segments(self, query: str) -> List[Dict]:
        """Search segments by keyword."""
        query_lower = query.lower()
        results = []

        for code, segment in self.data.get("segments", {}).items():
            score = 0

            # Check name
            if query_lower in segment.get("name", "").lower():
                score += 10

            # Check overview
            if query_lower in segment.get("overview", "").lower():
                score += 5

            # Check characteristics
            for char in segment.get("characteristics", []):
                if query_lower in char.lower():
                    score += 2

            if score > 0:
                results.append({"segment": segment, "score": score})

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def format_segment_response(self, segment: Dict) -> str:
        """Format segment information as a readable response."""
        if not segment:
            return "Segment not found."

        lines = [
            f"## {segment.get('name')} ({segment.get('code')})",
            f"**LifeMode Group:** {segment.get('lifemode_group')}",
            f"**Urbanization:** {segment.get('urbanization', 'N/A')}",
            "",
            f"### Overview",
            segment.get('overview', 'No overview available.'),
            "",
            "### Demographics",
        ]

        demo = segment.get("demographics", {})
        if demo:
            lines.append(f"- Median Age: {demo.get('median_age', 'N/A')}")
            lines.append(f"- Median Household Income: ${demo.get('median_household_income', 0):,}")
            lines.append(f"- Median Net Worth: ${demo.get('median_net_worth', 0):,}")
            lines.append(f"- Homeownership Rate: {demo.get('homeownership_rate', 0):.0%}")

        lines.append("")
        lines.append("### Key Characteristics")
        for char in segment.get("characteristics", []):
            lines.append(f"- {char}")

        lines.append("")
        lines.append("### Marketing Recommendations")
        for rec in segment.get("marketing_recommendations", []):
            lines.append(f"- {rec}")

        bi = segment.get("business_implications", {})
        if bi:
            lines.append("")
            lines.append("### Business Implications")
            for sector, implication in bi.items():
                lines.append(f"- **{sector.title()}:** {implication}")

        return "\n".join(lines)


# =============================================================================
# INSIGHT GENERATOR - Business insights and recommendations
# =============================================================================

class InsightGenerator:
    """
    Generate business insights from spatial analysis results.

    Transforms raw segment data into actionable business recommendations.
    """

    def __init__(self, knowledge_base: TapestryKnowledgeBase):
        """Initialize with knowledge base reference."""
        self.kb = knowledge_base

    def generate_trade_area_insights(
        self,
        top_segments: List[Dict],
        total_households: int,
        trade_area_type: str = "15-minute drive time",
        location_name: str = "selected location"
    ) -> str:
        """
        Generate comprehensive insights for a trade area analysis.

        Args:
            top_segments: List of top segments with code, name, value, percentage
            total_households: Total households in trade area
            trade_area_type: Description of the trade area
            location_name: Name/address of the center location

        Returns:
            Formatted markdown string with insights
        """
        lines = []

        # Header
        lines.append(f"## Trade Area Analysis: {location_name}")
        lines.append(f"**Coverage:** {trade_area_type}")
        lines.append(f"**Total Households:** {total_households:,}")
        lines.append("")

        # Top Segments Summary
        lines.append("### Top Consumer Segments")
        lines.append("")
        lines.append("| Rank | Segment | Households | Share |")
        lines.append("|------|---------|------------|-------|")

        for i, seg in enumerate(top_segments[:5], 1):
            code = seg.get("code", "")
            name = seg.get("name", self.kb.get_segment(code).get("name", "Unknown"))
            value = seg.get("value", 0)
            pct = seg.get("percentage", 0)
            lines.append(f"| {i} | {name} ({code}) | {value:,} | {pct:.1f}% |")

        lines.append("")

        # Dominant Lifestyle Theme
        if top_segments:
            primary_seg = top_segments[0]
            primary_code = primary_seg.get("code", "")
            primary_info = self.kb.get_segment(primary_code)

            lines.append("### Dominant Lifestyle")
            lines.append(f"**{primary_info.get('name', 'Unknown')}** ({primary_code})")
            lines.append("")
            lines.append(f"> {primary_info.get('description', 'No description available')}")
            lines.append("")

            # Key characteristics
            chars = primary_info.get("characteristics", [])[:3]
            if chars:
                lines.append("**Key Traits:**")
                for char in chars:
                    lines.append(f"- {char}")
                lines.append("")

        # Marketing Recommendations
        lines.append("### Marketing Recommendations")
        lines.append("")

        recommendations = self._generate_marketing_recommendations(top_segments)
        for rec in recommendations:
            lines.append(f"- {rec}")

        lines.append("")

        # Business Opportunities
        lines.append("### Business Opportunities")
        lines.append("")

        opportunities = self._identify_business_opportunities(top_segments)
        for opp in opportunities:
            lines.append(f"- {opp}")

        return "\n".join(lines)

    def _generate_marketing_recommendations(self, segments: List[Dict]) -> List[str]:
        """Generate marketing recommendations based on top segments."""
        recommendations = []

        if not segments:
            return ["Insufficient data for recommendations"]

        # Collect all marketing recommendations from top segments
        all_recs = []
        for seg in segments[:3]:
            code = seg.get("code", "")
            segment_info = self.kb.get_segment(code)
            recs = segment_info.get("marketing_recommendations", [])
            all_recs.extend(recs[:2])  # Take top 2 from each segment

        # Deduplicate and take top recommendations
        seen = set()
        for rec in all_recs:
            rec_lower = rec.lower()
            if rec_lower not in seen:
                seen.add(rec_lower)
                recommendations.append(rec)

        # Add default if empty
        if not recommendations:
            recommendations = [
                "Focus on digital marketing channels",
                "Emphasize value and quality in messaging",
                "Consider local community engagement"
            ]

        return recommendations[:5]

    def _identify_business_opportunities(self, segments: List[Dict]) -> List[str]:
        """Identify business opportunities based on segment mix."""
        opportunities = []

        if not segments:
            return ["Further analysis needed"]

        # Analyze segment characteristics
        urbanicity_scores = {"urban": 0, "suburban": 0, "rural": 0}
        income_levels = {"high": 0, "medium": 0, "low": 0}
        family_scores = {"families": 0, "singles": 0, "seniors": 0}

        for seg in segments[:5]:
            code = seg.get("code", "")
            weight = seg.get("percentage", 0)
            segment_info = self.kb.get_segment(code)

            # Analyze demographics
            demo = segment_info.get("demographics", {})

            # Income analysis
            income = demo.get("median_income", "")
            if income:
                if "150" in income or "200" in income or "high" in income.lower():
                    income_levels["high"] += weight
                elif "50" in income or "60" in income or "lower" in income.lower():
                    income_levels["low"] += weight
                else:
                    income_levels["medium"] += weight

            # Age/family analysis
            age = demo.get("median_age", "")
            if "senior" in age.lower() or "65" in age or "70" in age:
                family_scores["seniors"] += weight
            elif "family" in str(segment_info).lower() or "children" in str(segment_info).lower():
                family_scores["families"] += weight
            else:
                family_scores["singles"] += weight

        # Generate opportunities based on analysis
        dominant_income = max(income_levels, key=income_levels.get)
        if dominant_income == "high":
            opportunities.append("Premium product and service offerings")
            opportunities.append("Luxury brand partnerships")
        elif dominant_income == "low":
            opportunities.append("Value-focused promotions")
            opportunities.append("Budget-friendly product lines")
        else:
            opportunities.append("Mid-range product positioning")
            opportunities.append("Quality-value balance messaging")

        dominant_family = max(family_scores, key=family_scores.get)
        if dominant_family == "families":
            opportunities.append("Family-oriented programs and events")
        elif dominant_family == "seniors":
            opportunities.append("Senior-focused services and accessibility")
        else:
            opportunities.append("Convenience and time-saving solutions")

        return opportunities[:5]

    def generate_comparison_insights(
        self,
        location_a: Dict,
        location_b: Dict
    ) -> str:
        """Generate insights comparing two trade areas."""
        lines = []

        lines.append("## Trade Area Comparison")
        lines.append("")
        lines.append("| Metric | Location A | Location B |")
        lines.append("|--------|------------|------------|")

        # Compare top segments
        top_a = location_a.get("top_segments", [])[:3]
        top_b = location_b.get("top_segments", [])[:3]

        lines.append(f"| Total HH | {location_a.get('total_households', 0):,} | {location_b.get('total_households', 0):,} |")

        if top_a:
            lines.append(f"| #1 Segment | {top_a[0].get('name', 'N/A')} | {top_b[0].get('name', 'N/A') if top_b else 'N/A'} |")

        lines.append("")
        lines.append("### Key Differences")

        # Identify unique segments
        codes_a = {s.get("code") for s in top_a}
        codes_b = {s.get("code") for s in top_b}

        unique_a = codes_a - codes_b
        unique_b = codes_b - codes_a

        if unique_a:
            lines.append(f"- Location A has unique presence of: {', '.join(unique_a)}")
        if unique_b:
            lines.append(f"- Location B has unique presence of: {', '.join(unique_b)}")

        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_query_parser: Optional[SpatialQueryParser] = None
_planning_engine: Optional[PlanningEngine] = None
_knowledge_base: Optional[TapestryKnowledgeBase] = None
_insight_generator: Optional[InsightGenerator] = None


def get_query_parser() -> SpatialQueryParser:
    """Get or create the query parser."""
    global _query_parser
    if _query_parser is None:
        kb = get_knowledge_base()
        _query_parser = SpatialQueryParser(kb.data)
    return _query_parser


def get_planning_engine() -> PlanningEngine:
    """Get or create the planning engine."""
    global _planning_engine
    if _planning_engine is None:
        _planning_engine = PlanningEngine()
    return _planning_engine


def get_knowledge_base() -> TapestryKnowledgeBase:
    """Get or create the knowledge base."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = TapestryKnowledgeBase()
    return _knowledge_base


def get_insight_generator() -> InsightGenerator:
    """Get or create the insight generator."""
    global _insight_generator
    if _insight_generator is None:
        kb = get_knowledge_base()
        _insight_generator = InsightGenerator(kb)
    return _insight_generator


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_spatial_query(query: str) -> ParsedQuery:
    """Parse a spatial query."""
    return get_query_parser().parse(query)


def create_execution_plan(query: str) -> ExecutionPlan:
    """Create an execution plan for a query."""
    parsed = parse_spatial_query(query)
    return get_planning_engine().create_plan(parsed)


def get_segment_info(segment_code: str) -> str:
    """Get formatted segment information."""
    kb = get_knowledge_base()
    segment = kb.get_segment(segment_code)
    return kb.format_segment_response(segment)


def generate_trade_area_insights(
    top_segments: List[Dict],
    total_households: int,
    trade_area_type: str = "15-minute drive time",
    location_name: str = "selected location"
) -> str:
    """
    Generate business insights for a trade area analysis.

    Args:
        top_segments: List of segment dictionaries with code, name, value, percentage
        total_households: Total households in the trade area
        trade_area_type: Description of trade area coverage
        location_name: Name/address of the center location

    Returns:
        Formatted markdown string with insights and recommendations
    """
    return get_insight_generator().generate_trade_area_insights(
        top_segments=top_segments,
        total_households=total_households,
        trade_area_type=trade_area_type,
        location_name=location_name
    )


def generate_comparison_insights(location_a: Dict, location_b: Dict) -> str:
    """
    Generate insights comparing two trade areas.

    Args:
        location_a: First location data with top_segments and total_households
        location_b: Second location data with top_segments and total_households

    Returns:
        Formatted markdown string with comparison insights
    """
    return get_insight_generator().generate_comparison_insights(location_a, location_b)
