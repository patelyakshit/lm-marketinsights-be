"""
AddressPatternAgent - Specialized agent for analyzing Salesforce record address field patterns.

This agent uses LLM capabilities to intelligently analyze Salesforce records and identify
address field patterns with structured Pydantic response validation.
"""

import json
import logging
from typing import List

from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from agents.streaming_callbacks import (
    after_tool_modifier,
    before_tool_modifier,
    after_model_modifier,
    before_model_modifier,
)
from schemas.address_patterns import AddressPatternAnalysis, get_address_pattern_schema

logger = logging.getLogger(__name__)


class AddressPatternAgent(LlmAgent):
    """
    Specialized agent for analyzing Salesforce record address field patterns.

    This agent analyzes Salesforce records to identify address field structures
    and returns validated JSON responses using Pydantic models for type safety
    and structured data exchange with GIS geocoding operations.
    """

    def __init__(self, model: str | Gemini = "gemini-2.5-flash-lite"):
        """
        Initialize AddressPatternAgent with ADK architecture and streaming.

        Args:
            model_id: LLM model identifier (default: "gemini-2.5-flash-lite")
        """

        super().__init__(
            model=model,
            name="address_pattern_agent",
            description="Specialized agent for analyzing Salesforce record address field patterns with validated JSON output",
            instruction=self._get_system_instruction(),
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
        )

        logger.info("AddressPatternAgent initialized with Pydantic validation")

    def _get_system_instruction(self) -> str:
        """
        Get comprehensive system instruction for address pattern analysis.
        """

        # Get the JSON schema and example from Pydantic model
        schema = get_address_pattern_schema()
        example = AddressPatternAnalysis.Config.json_schema_extra["example"]

        return f"""You are an expert at analyzing Salesforce record structures to identify address field patterns for geocoding operations.

MISSION: Analyze Salesforce records and return structured JSON that exactly matches the required schema for seamless GIS integration.

REQUIRED JSON SCHEMA:
{json.dumps(schema, indent=2)}

EXAMPLE RESPONSE:
{json.dumps(example, indent=2)}

ANALYSIS INSTRUCTIONS:

1. PATTERN IDENTIFICATION:
   - Identify address field groupings: billing, shipping, mailing, office, etc.
   - Detect pattern types: "compound_object" vs "individual_fields"
   - Map each address component to exact field access path

2. PATTERN TYPES:
   - compound_object: Address stored as nested object (e.g., record["BillingAddress"]["street"])
   - individual_fields: Address components in separate fields (e.g., record["BillingStreet"])

3. ACCESS PATHS:
   - For compound objects: "BillingAddress.street", "BillingAddress.city"
   - For individual fields: "BillingStreet", "BillingCity"
   - Use exact field names as they appear in the records

4. DATA COMPLETENESS:
   - Calculate percentage of records with non-null, non-empty values
   - Range: 0.0 to 1.0 (e.g., 0.85 = 85% complete)

5. GEOCODING STRATEGIES:
   - compound_preferred: Use compound object if available
   - individual_preferred: Use individual fields if available
   - compound_preferred_with_individual_fallback: Try compound first, fallback to individual
   - individual_preferred_with_compound_fallback: Try individual first, fallback to compound

6. COORDINATE FIELD RECOMMENDATIONS:
   - Standard: "latitude", "longitude"
   - Salesforce custom: "Latitude__c", "Longitude__c"
   - Alternative: "lat", "lng"

RESPONSE REQUIREMENTS:
- Return ONLY valid JSON matching the schema
- No additional text, explanations, or markdown formatting
- Ensure all required fields are present
- Use exact field names from the actual records
- Calculate accurate data completeness percentages

ANALYSIS CONFIDENCE:
- High confidence (0.9+): Clear, consistent patterns found
- Medium confidence (0.7-0.9): Some variations but identifiable patterns
- Low confidence (0.5-0.7): Inconsistent or unclear patterns

Example field analysis:
- "BillingStreet" + "BillingCity" = individual_fields pattern
- "BillingAddress": {{"street": "...", "city": "..."}} = compound_object pattern
- Mixed patterns = recommend fallback strategy

Return the JSON response immediately without any preamble or explanation."""

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""

        return [
            "Salesforce record structure analysis",
            "Address field pattern identification",
            "Compound vs individual field detection",
            "Data completeness assessment",
            "Geocoding strategy recommendations",
            "Pydantic-validated JSON responses",
            "Python access path generation",
            "Multi-address type support (billing, shipping, mailing)",
            "Field naming convention analysis",
            "Real-time pattern confidence scoring",
        ]


# Create agent instance for backward compatibility
address_pattern_agent = AddressPatternAgent()
