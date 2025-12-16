"""
Unit tests for Workflow Templates module.
"""

import pytest
from utils.workflow_templates import (
    WorkflowType,
    WorkflowMatch,
    WorkflowResult,
    WorkflowMatcher,
    ZoomToLocationWorkflow,
    ShowLayerWorkflow,
    HideLayerWorkflow,
    ZoomInWorkflow,
    ZoomOutWorkflow,
    PanMapWorkflow,
    get_workflow_matcher,
    should_use_workflow,
)


class TestWorkflowMatching:
    """Tests for workflow pattern matching."""

    @pytest.fixture
    def matcher(self):
        """Create a fresh matcher."""
        return WorkflowMatcher()

    def test_zoom_to_location_patterns(self, matcher):
        """Test zoom to location pattern matching."""
        queries = [
            "zoom to Dallas, TX",
            "go to New York",
            "navigate to San Francisco",
            "fly to Chicago",
            "take me to Miami",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.ZOOM_TO_LOCATION

    def test_show_layer_patterns(self, matcher):
        """Test show layer pattern matching."""
        queries = [
            "show the demographics layer",
            "enable the traffic layer",
            "turn on the satellite layer",
            "display the boundaries layer",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.SHOW_LAYER

    def test_hide_layer_patterns(self, matcher):
        """Test hide layer pattern matching."""
        queries = [
            "hide the demographics layer",
            "disable the traffic layer",
            "turn off the satellite layer",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.HIDE_LAYER

    def test_zoom_in_patterns(self, matcher):
        """Test zoom in pattern matching."""
        queries = [
            "zoom in",
            "get closer",
            "magnify",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.ZOOM_IN

    def test_zoom_out_patterns(self, matcher):
        """Test zoom out pattern matching."""
        queries = [
            "zoom out",
            "see more",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.ZOOM_OUT

    def test_pan_map_patterns(self, matcher):
        """Test pan map pattern matching."""
        queries = [
            "pan north",
            "pan south",
            "move the map east",
            "move the map west",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert match.matched, f"Should match: {query}"
            assert match.workflow_type == WorkflowType.PAN_MAP

    def test_no_match_complex_queries(self, matcher):
        """Test that complex queries don't match templates."""
        queries = [
            "What is the population of Dallas?",
            "Analyze the demographics for downtown Austin",
            "Create a marketing post for Beverly Hills",
            "Show me the Salesforce accounts",
        ]

        for query in queries:
            match = matcher.match_query(query)
            assert not match.matched, f"Should NOT match: {query}"


class TestParameterExtraction:
    """Tests for parameter extraction from queries."""

    def test_extract_location(self):
        """Test location extraction from zoom queries."""
        workflow = ZoomToLocationWorkflow()
        match = workflow.match("zoom to Dallas, TX")

        assert match.matched
        # Location should be extracted
        assert "location" in match.extracted_params or len(match.extracted_params) == 0

    def test_extract_layer_name(self):
        """Test layer name extraction."""
        workflow = ShowLayerWorkflow()
        match = workflow.match("show the demographics layer")

        assert match.matched
        assert "layer_name" in match.extracted_params
        assert "demographics" in match.extracted_params["layer_name"].lower()

    def test_extract_direction(self):
        """Test direction extraction from pan queries."""
        workflow = PanMapWorkflow()
        match = workflow.match("pan north")

        assert match.matched
        assert "direction" in match.extracted_params
        assert match.extracted_params["direction"].lower() == "north"


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_success_result(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            success=True,
            message="Zoomed to Dallas",
            data={"lat": 32.78, "lng": -96.80},
            operations=[{"type": "zoom", "extent": {}}]
        )

        assert result.success
        assert "Dallas" in result.message
        assert result.data["lat"] == 32.78
        assert len(result.operations) == 1

    def test_failure_result(self):
        """Test failed workflow result."""
        result = WorkflowResult(
            success=False,
            message="Location not found"
        )

        assert not result.success
        assert "not found" in result.message


class TestShouldUseWorkflow:
    """Tests for should_use_workflow function."""

    def test_simple_commands_should_use_workflow(self):
        """Simple commands should use workflow templates."""
        assert should_use_workflow("zoom to Dallas")
        assert should_use_workflow("zoom in")
        assert should_use_workflow("show the traffic layer")

    def test_complex_queries_should_not_use_workflow(self):
        """Complex queries should not use workflow templates."""
        assert not should_use_workflow("What demographics does Dallas have?")
        assert not should_use_workflow("Analyze the best location for a coffee shop")


class TestWorkflowMatchConfidence:
    """Tests for match confidence scoring."""

    @pytest.fixture
    def matcher(self):
        return WorkflowMatcher()

    def test_high_confidence_matches(self, matcher):
        """Clear patterns should have high confidence."""
        match = matcher.match_query("zoom to Dallas")
        if match.matched:
            assert match.confidence >= 0.8

    def test_should_use_template_threshold(self, matcher):
        """should_use_template should respect confidence threshold."""
        # Clear match
        assert matcher.should_use_template("zoom to Dallas")

        # No match
        assert not matcher.should_use_template("analyze demographics")
