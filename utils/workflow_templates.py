"""
Static Workflow Templates

Pre-defined execution paths for common operations that bypass dynamic planning.
This significantly reduces latency for frequent operations like:
- Zoom to location
- Show/hide layer
- Filter layer
- Get location info

Instead of going through full AI planning, these templates execute
directly with minimal overhead.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


def _normalize_layer_name(name: str) -> str:
    """Normalize layer name for fuzzy matching.

    Removes special characters and extra whitespace to allow matching like:
    - "age median" matches "Age: Median"
    - "population density" matches "Population Density"
    - "housing affordability" matches "Housing Affordability Index"
    """
    # Remove common special characters
    normalized = re.sub(r'[:\-_/\\|]', ' ', name.lower())
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def _match_layer(search_name: str, layers: list) -> Optional[dict]:
    """Find best matching layer using fuzzy matching.

    Args:
        search_name: User's search term (e.g., "age median")
        layers: List of layer dicts with "title" and "id" keys

    Returns:
        Best matching layer dict or None
    """
    search_normalized = _normalize_layer_name(search_name)
    search_words = set(search_normalized.split())

    best_match = None
    best_score = 0

    for layer in layers:
        title = layer.get("title", "")
        title_normalized = _normalize_layer_name(title)
        title_words = set(title_normalized.split())

        # Check if all search words appear in title
        if search_words.issubset(title_words):
            # Prefer exact/shorter matches
            score = len(search_words) / len(title_words) if title_words else 0
            if score > best_score:
                best_score = score
                best_match = layer
        # Also try substring match as fallback
        elif search_normalized in title_normalized:
            score = 0.5
            if score > best_score:
                best_score = score
                best_match = layer

    return best_match


class WorkflowType(Enum):
    """Types of workflow templates."""
    ZOOM_TO_LOCATION = "zoom_to_location"
    SHOW_LAYER = "show_layer"
    HIDE_LAYER = "hide_layer"
    FILTER_LAYER = "filter_layer"
    REMOVE_FILTER = "remove_filter"
    ADD_PIN = "add_pin"
    REMOVE_PINS = "remove_pins"
    GET_LOCATION_INFO = "get_location_info"
    PAN_MAP = "pan_map"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


@dataclass
class WorkflowMatch:
    """Result of matching a query to a workflow template."""
    matched: bool
    workflow_type: Optional[WorkflowType] = None
    confidence: float = 0.0
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    original_query: str = ""


@dataclass
class WorkflowResult:
    """Result of executing a workflow template."""
    success: bool
    data: Any = None
    message: str = ""
    operations: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowTemplate:
    """Base class for workflow templates."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        patterns: List[str],
        param_extractors: Dict[str, str] = None,
    ):
        """
        Initialize workflow template.

        Args:
            workflow_type: Type of workflow
            patterns: Regex patterns to match queries
            param_extractors: Regex patterns to extract parameters
        """
        self.workflow_type = workflow_type
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.param_extractors = param_extractors or {}

    def match(self, query: str) -> WorkflowMatch:
        """
        Check if query matches this workflow template.

        Args:
            query: User query text

        Returns:
            WorkflowMatch with match results
        """
        for pattern in self.patterns:
            match = pattern.search(query)
            if match:
                # Extract parameters
                params = {}
                for param_name, extractor in self.param_extractors.items():
                    param_match = re.search(extractor, query, re.IGNORECASE)
                    if param_match:
                        params[param_name] = param_match.group(1)

                return WorkflowMatch(
                    matched=True,
                    workflow_type=self.workflow_type,
                    confidence=0.9,
                    extracted_params=params,
                    original_query=query,
                )

        return WorkflowMatch(matched=False, original_query=query)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """
        Execute the workflow template.

        Args:
            params: Extracted parameters
            context: Additional context (map_context, gis_executor, etc.)

        Returns:
            WorkflowResult with execution results
        """
        raise NotImplementedError("Subclasses must implement execute()")


class ZoomToLocationWorkflow(WorkflowTemplate):
    """Workflow for zooming to a location."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.ZOOM_TO_LOCATION,
            patterns=[
                r'\b(?:zoom|go|navigate|fly)\s+(?:to|into)\s+(.+)',
                r'\bshow\s+(?:me\s+)?(.+)\s+on\s+(?:the\s+)?map\b',
                r'\btake\s+me\s+to\s+(.+)',
            ],
            param_extractors={
                "location": r'(?:zoom|go|navigate|fly|show|take)\s+(?:to|into|me)?\s*(?:me\s+)?(.+?)(?:\s+on\s+(?:the\s+)?map)?$',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute zoom to location."""
        try:
            location = params.get("location", "").strip()
            if not location:
                return WorkflowResult(
                    success=False,
                    message="No location specified"
                )

            # Get GIS executor from context
            gis_executor = context.get("gis_executor") if context else None
            if not gis_executor:
                return WorkflowResult(
                    success=False,
                    message="GIS executor not available"
                )

            # Geocode the location
            suggestions = await gis_executor.get_coordinates_suggestions(location, max_candidates=1)

            if not suggestions:
                return WorkflowResult(
                    success=False,
                    message=f"Could not find location: {location}"
                )

            best = suggestions[0]
            extent = best.get("extent", {})

            if extent:
                # Zoom to extent
                result = await gis_executor.zoom_to_location(
                    extent,
                    {"wkid": 4326}
                )
                return WorkflowResult(
                    success=True,
                    data=best,
                    message=f"Zoomed to {best.get('address', location)}",
                    operations=result.get("operations", [])
                )

            return WorkflowResult(
                success=False,
                message=f"Could not determine extent for: {location}"
            )

        except Exception as e:
            logger.error(f"Error in ZoomToLocationWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class ShowLayerWorkflow(WorkflowTemplate):
    """Workflow for showing a layer."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.SHOW_LAYER,
            patterns=[
                r'\bshow\s+(?:the\s+)?(.+?)\s+layer\b',
                r'\benable\s+(?:the\s+)?(.+?)\s+layer\b',
                r'\bturn\s+on\s+(?:the\s+)?(.+?)\s+layer\b',
                r'\bdisplay\s+(?:the\s+)?(.+?)\s+layer\b',
            ],
            param_extractors={
                "layer_name": r'(?:show|enable|turn\s+on|display)\s+(?:the\s+)?(.+?)\s+layer',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute show layer."""
        try:
            layer_name = params.get("layer_name", "").strip()
            if not layer_name:
                return WorkflowResult(success=False, message="No layer name specified")

            gis_executor = context.get("gis_executor") if context else None
            map_context = context.get("map_context", {}) if context else {}

            if not gis_executor:
                return WorkflowResult(success=False, message="GIS executor not available")

            # Find layer by name using fuzzy matching
            layers = map_context.get("layers", [])
            logger.info(f"[ShowLayerWorkflow] Searching for '{layer_name}' in {len(layers)} layers")

            if not layers:
                logger.warning(f"[ShowLayerWorkflow] No layers in map_context! Cannot toggle layer.")
                # Return failure so AI can try to help
                return WorkflowResult(
                    success=False,
                    message=f"NO_LAYERS_CONTEXT"  # Special marker for root_agent to handle
                )

            logger.debug(f"[ShowLayerWorkflow] Available layers: {[(l.get('id'), l.get('title')) for l in layers]}")

            matched_layer = _match_layer(layer_name, layers)

            if not matched_layer:
                logger.warning(f"[ShowLayerWorkflow] Layer not found: {layer_name}")
                available_titles = [l.get('title', 'Unknown') for l in layers[:5]]
                return WorkflowResult(
                    success=False,
                    message=f"Layer '{layer_name}' not found. Available layers: {', '.join(available_titles)}"
                )

            logger.info(f"[ShowLayerWorkflow] Matched layer: id={matched_layer.get('id')}, title={matched_layer.get('title')}")

            # Toggle visibility
            result = await gis_executor.toggle_layer_visibility(
                matched_layer.get("id"),
                matched_layer.get("title"),
                True
            )

            logger.info(f"[ShowLayerWorkflow] Sending toggle operation: {result}")

            return WorkflowResult(
                success=True,
                data=matched_layer,
                message=f"Showing layer: {matched_layer.get('title')}",
                operations=result.get("operations", [])
            )

        except Exception as e:
            logger.error(f"Error in ShowLayerWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class HideLayerWorkflow(WorkflowTemplate):
    """Workflow for hiding a layer."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.HIDE_LAYER,
            patterns=[
                r'\bhide\s+(?:the\s+)?(.+?)\s+layer\b',
                r'\bdisable\s+(?:the\s+)?(.+?)\s+layer\b',
                r'\bturn\s+off\s+(?:the\s+)?(.+?)\s+layer\b',
            ],
            param_extractors={
                "layer_name": r'(?:hide|disable|turn\s+off)\s+(?:the\s+)?(.+?)\s+layer',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute hide layer."""
        try:
            layer_name = params.get("layer_name", "").strip()
            if not layer_name:
                return WorkflowResult(success=False, message="No layer name specified")

            gis_executor = context.get("gis_executor") if context else None
            map_context = context.get("map_context", {}) if context else {}

            if not gis_executor:
                return WorkflowResult(success=False, message="GIS executor not available")

            # Find layer by name using fuzzy matching
            layers = map_context.get("layers", [])
            logger.info(f"[HideLayerWorkflow] Searching for '{layer_name}' in {len(layers)} layers")

            if not layers:
                logger.warning(f"[HideLayerWorkflow] No layers in map_context! Cannot toggle layer.")
                return WorkflowResult(
                    success=False,
                    message=f"NO_LAYERS_CONTEXT"
                )

            matched_layer = _match_layer(layer_name, layers)

            if not matched_layer:
                logger.warning(f"[HideLayerWorkflow] Layer not found: {layer_name}")
                available_titles = [l.get('title', 'Unknown') for l in layers[:5]]
                return WorkflowResult(
                    success=False,
                    message=f"Layer '{layer_name}' not found. Available layers: {', '.join(available_titles)}"
                )

            logger.info(f"[HideLayerWorkflow] Matched layer: id={matched_layer.get('id')}, title={matched_layer.get('title')}")

            # Toggle visibility
            result = await gis_executor.toggle_layer_visibility(
                matched_layer.get("id"),
                matched_layer.get("title"),
                False
            )

            logger.info(f"[HideLayerWorkflow] Sending toggle operation: {result}")

            return WorkflowResult(
                success=True,
                data=matched_layer,
                message=f"Hidden layer: {matched_layer.get('title')}",
                operations=result.get("operations", [])
            )

        except Exception as e:
            logger.error(f"Error in HideLayerWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class ZoomInWorkflow(WorkflowTemplate):
    """Workflow for zooming in."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.ZOOM_IN,
            patterns=[
                r'\bzoom\s+in\b',
                r'\bget\s+closer\b',
                r'\bmagnify\b',
            ],
            param_extractors={
                "amount": r'(?:zoom\s+in|magnify)\s+(\d+)%?',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute zoom in."""
        try:
            gis_executor = context.get("gis_executor") if context else None
            if not gis_executor:
                return WorkflowResult(success=False, message="GIS executor not available")

            amount = int(params.get("amount", 50))
            result = await gis_executor.zoom_map("zoom_in", amount)

            return WorkflowResult(
                success=True,
                message="Zoomed in",
                operations=result.get("operations", [])
            )

        except Exception as e:
            logger.error(f"Error in ZoomInWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class ZoomOutWorkflow(WorkflowTemplate):
    """Workflow for zooming out."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.ZOOM_OUT,
            patterns=[
                r'\bzoom\s+out\b',
                r'\bget\s+further\b',
                r'\bsee\s+more\b',
            ],
            param_extractors={
                "amount": r'(?:zoom\s+out)\s+(\d+)%?',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute zoom out."""
        try:
            gis_executor = context.get("gis_executor") if context else None
            if not gis_executor:
                return WorkflowResult(success=False, message="GIS executor not available")

            amount = int(params.get("amount", 50))
            result = await gis_executor.zoom_map("zoom_out", amount)

            return WorkflowResult(
                success=True,
                message="Zoomed out",
                operations=result.get("operations", [])
            )

        except Exception as e:
            logger.error(f"Error in ZoomOutWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class PanMapWorkflow(WorkflowTemplate):
    """Workflow for panning the map."""

    def __init__(self):
        super().__init__(
            workflow_type=WorkflowType.PAN_MAP,
            patterns=[
                r'\bpan\s+(north|south|east|west|up|down|left|right)\b',
                r'\bmove\s+(?:the\s+)?map\s+(north|south|east|west|up|down|left|right)\b',
            ],
            param_extractors={
                "direction": r'\b(north|south|east|west|up|down|left|right)\b',
            }
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """Execute pan map."""
        try:
            gis_executor = context.get("gis_executor") if context else None
            if not gis_executor:
                return WorkflowResult(success=False, message="GIS executor not available")

            direction = params.get("direction", "").lower()
            if not direction:
                return WorkflowResult(success=False, message="No direction specified")

            result = await gis_executor.pan_map(direction, 20)

            return WorkflowResult(
                success=True,
                message=f"Panned map {direction}",
                operations=result.get("operations", [])
            )

        except Exception as e:
            logger.error(f"Error in PanMapWorkflow: {e}")
            return WorkflowResult(success=False, message=str(e))


class WorkflowMatcher:
    """
    Matches queries to workflow templates and executes them.

    Usage:
        matcher = WorkflowMatcher()

        # Check if query matches a template
        match = matcher.match_query("zoom to Dallas, TX")
        if match.matched:
            result = await matcher.execute(match, context)
    """

    def __init__(self):
        """Initialize with default workflow templates."""
        self.templates: List[WorkflowTemplate] = [
            ZoomToLocationWorkflow(),
            ShowLayerWorkflow(),
            HideLayerWorkflow(),
            ZoomInWorkflow(),
            ZoomOutWorkflow(),
            PanMapWorkflow(),
        ]

    def match_query(self, query: str) -> WorkflowMatch:
        """
        Find matching workflow template for query.

        Args:
            query: User query text

        Returns:
            WorkflowMatch with best match (or no match)
        """
        best_match = WorkflowMatch(matched=False, original_query=query)

        for template in self.templates:
            match = template.match(query)
            if match.matched and match.confidence > best_match.confidence:
                best_match = match

        if best_match.matched:
            logger.info(
                f"Query matched workflow template: {best_match.workflow_type.value} "
                f"(confidence: {best_match.confidence:.2f})"
            )

        return best_match

    async def execute(
        self,
        match: WorkflowMatch,
        context: Dict[str, Any] = None
    ) -> WorkflowResult:
        """
        Execute matched workflow template.

        Args:
            match: WorkflowMatch from match_query()
            context: Execution context (gis_executor, map_context, etc.)

        Returns:
            WorkflowResult with execution results
        """
        if not match.matched:
            return WorkflowResult(success=False, message="No workflow matched")

        # Find template
        template = None
        for t in self.templates:
            if t.workflow_type == match.workflow_type:
                template = t
                break

        if not template:
            return WorkflowResult(
                success=False,
                message=f"Template not found: {match.workflow_type}"
            )

        return await template.execute(match.extracted_params, context)

    def add_template(self, template: WorkflowTemplate) -> None:
        """Add a custom workflow template."""
        self.templates.append(template)

    def should_use_template(self, query: str) -> bool:
        """
        Quick check if query should use a workflow template.

        Args:
            query: User query text

        Returns:
            True if query matches any template
        """
        match = self.match_query(query)
        return match.matched and match.confidence >= 0.8


# Global matcher instance
_workflow_matcher: Optional[WorkflowMatcher] = None


def get_workflow_matcher() -> WorkflowMatcher:
    """Get or create global WorkflowMatcher instance."""
    global _workflow_matcher
    if _workflow_matcher is None:
        _workflow_matcher = WorkflowMatcher()
    return _workflow_matcher


def should_use_workflow(query: str) -> bool:
    """
    Check if query should use a workflow template.

    Args:
        query: User query text

    Returns:
        True if query matches a template with high confidence
    """
    matcher = get_workflow_matcher()
    return matcher.should_use_template(query)


async def execute_workflow(
    query: str,
    context: Dict[str, Any] = None
) -> Optional[WorkflowResult]:
    """
    Execute workflow if query matches a template.

    Args:
        query: User query text
        context: Execution context

    Returns:
        WorkflowResult if matched and executed, None otherwise
    """
    matcher = get_workflow_matcher()
    match = matcher.match_query(query)

    if not match.matched:
        return None

    return await matcher.execute(match, context)
