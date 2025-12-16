"""
Pure Google ADK Salesforce Agent

Specialized agent for Salesforce operations with real-time streaming capabilities.
Inherits directly from LlmAgent and uses streaming callbacks for tool execution visibility.
"""

import logging
from typing import List

from decouple import config
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import AgentTool

from agents.address_pattern_agent import AddressPatternAgent
from agents.streaming_callbacks import (
    after_tool_modifier,
    before_tool_modifier,
    after_model_modifier,
    before_model_modifier,
)
from schemas.address_patterns import (
    AddressPatternAnalysis,
    validate_address_pattern_analysis,
)
from tools.salesforce_adk_tools import SALESFORCE_ADK_TOOLS

logger = logging.getLogger(__name__)


class SalesforceAgent(LlmAgent):
    """
    Pure ADK Salesforce Agent with streaming capabilities.

    Specialized for Salesforce operations including:
    - SOQL query construction and execution
    - Data export and analysis workflows
    - Object and field schema exploration
    - Real-time tool execution streaming
    """

    def __init__(
        self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True
    ):
        """
        Initialize Salesforce Agent with ADK architecture and streaming.

        Args:
            model: LLM model identifier (default: "gemini-2.5-flash-lite")
            allow_override: If True, SUB_AGENT_MODEL env var can override.
                          If False, uses provided model (e.g., for audio streaming).
        """
        # Determine final model based on override setting
        final_model = (
            config("SUB_AGENT_MODEL", default=None) or model
            if allow_override
            else model
        )

        # Create specialized address pattern analysis agent
        pattern_agent = AddressPatternAgent(final_model)

        super().__init__(
            model=final_model,
            name="salesforce_agent",
            description="Specialized agent for Salesforce SOQL queries, CRM operations, and intelligent address pattern analysis",
            instruction=self._get_system_instruction(),
            tools=[*SALESFORCE_ADK_TOOLS, AgentTool(agent=pattern_agent)],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
            # Prevent sub-agent from seeing prior conversation history
            # This avoids verbose re-introductions during agent transfers
            include_contents="default",
        )

        logger.info(
            f"SalesforceAgent initialized with model: {final_model} "
            f"(override {'allowed' if allow_override else 'disabled'})"
        )

    def _get_system_instruction(self) -> str:
        """
        Get comprehensive system instruction for Salesforce operations.
        """

        return """Salesforce Agent for CRM operations. Execute tasks immediately without introduction.

## Rules
- Be concise, show results only
- All SOQL must have LIMIT (enforced)
- Never show Salesforce IDs to users

## Tools
- get_all_objects, get_all_fields_for_object - Schema exploration
- query_soql, validate_soql_query, query_sosl - Query execution
- get_record_count, query_with_pagination - Large datasets (up to 50k)
- aggregate_query - COUNT, SUM, AVG, etc.
- export_query_to_csv, export_query_to_geojson - Data export
- geocode_existing_records, analyze_address_patterns - Address/mapping

## Query Patterns
- Default: SELECT FIELDS(ALL) FROM Object LIMIT 200
- Performance: Add WHERE filters (e.g., BillingCity != null)
- Large data: Use pagination for >200 records
- Mapping: Include Id, Name + address fields

## Response Format
- Max 5 records in Markdown table (Name + 2-3 fields)
- Add "[X more records...]" for additional
- End with: "Export to CSV?" or "Plot on map?"
- Mask URLs: [Download Here](url)

## Mapping Workflow
1. analyze_address_patterns (detect address fields)
2. query_soql/pagination (fetch data)
3. export_query_to_geojson or geocode_existing_records

## Errors
Validate queries, suggest fixes, recommend alternatives

## Voice Mode
STT only → Root Agent for TTS. ~20 words max.
"""

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""

        return [
            "SOQL query construction and execution",
            "SOSL cross-object searches",
            "Object and field schema analysis",
            "Data relationship mapping",
            "Large dataset pagination",
            "Aggregate query operations",
            "CSV data export",
            "Query performance optimization",
            "Address data extraction for mapping",
            "Intelligent address pattern analysis for geocoding",
            "Pydantic-validated address pattern responses",
            "Agent-as-Tool pattern analysis delegation",
            "Real-time tool execution streaming",
            "GeoJSON export for map visualization",
            "Automatic address field detection (billing/shipping/mailing)",
            "Batch geocoding with ArcGIS integration",
            "Multi-address type support in GeoJSON output",
        ]

    def parse_address_pattern_analysis(
        self, analysis_response: str
    ) -> AddressPatternAnalysis:
        """
        Parse and validate address pattern analysis response from agent tool.

        Args:
            analysis_response: JSON string response from analyze_address_patterns agent tool

        Returns:
            Validated AddressPatternAnalysis object

        Raises:
            ValueError: If response doesn't match expected schema
        """
        try:
            import json

            analysis_dict = json.loads(analysis_response)
            return validate_address_pattern_analysis(analysis_dict)
        except Exception as e:
            logger.error(f"Failed to parse address pattern analysis: {e}")
            raise ValueError(f"Invalid address pattern analysis format: {e}")

    def get_address_pattern_schema(self) -> dict:
        """
        Get the JSON schema for address pattern analysis responses.

        Returns:
            JSON schema dictionary for validation and documentation
        """
        from schemas.address_patterns import get_address_pattern_schema

        return get_address_pattern_schema()
