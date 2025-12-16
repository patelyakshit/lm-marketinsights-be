import json
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from decouple import config
from tools.placestory_adk_tools import generate_placestory_from_context_tool, update_placestory_context_tool
from tools.placestory_adk_tools import update_placestory_context_tool
from agents.streaming_callbacks import (
    after_tool_modifier,
    before_tool_modifier,
    after_model_modifier,
    before_model_modifier,
)
import logging

logger = logging.getLogger(__name__)



class PlaceStoryAgent(LlmAgent):

    def __init__(
        self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True):

        final_model = (
            config("SUB_AGENT_MODEL", default=None) or model
            if allow_override
            else model
        )

        super().__init__(
            model=final_model,
            name="placestory_agent",
            description="Chats with the user to collect placestory context and triggers generation when ready.",
            instruction=self._get_system_instruction(),
            tools=[
                update_placestory_context_tool,
                generate_placestory_from_context_tool,
            ],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
            # Prevent sub-agent from seeing prior conversation history
            # This avoids verbose re-introductions during agent transfers
            include_contents="default",
        )

        logger.info(f"PlaceStoryAgent initialized with model: {final_model}")

    def _get_system_instruction(self) -> str:
        return f"""
        You are the PlaceStory interviewer. 

        <GOAL>
        Have a natural conversation to collect information and store it in session.state["placestory_context"].
        </GOAL>

        <REQUIRED FIELDS - Use These EXACT Field Names>
        When calling update_placestory_context_tool, use these EXACT field names:

        1. "address" - What property, address, or geographic area should this story focus on?
        2. "asset_type" - What type of asset? (office, retail, multifamily, industrial, mixed-use, etc.)
        3. "audience" - Who is the intended audience? (investors, tenants, buyers, community, etc.)
        4. "intent_of_story" - Main purpose? (attract investment, lease space, demonstrate viability, etc.)
        5. "intended_action" - What action should reader take? (contact you, schedule meeting, visit site, etc.)
        6. "tone_voice" - How should it feel? (corporate/data-driven, visionary/aspirational, conversational/human)
        </REQUIRED FIELDS>

        <OPTIONAL FIELDS - Use These EXACT Field Names>
        7. "emotional_response" - What should reader feel? (confidence, excitement, inspiration, trust, etc.)
        8. "key_attributes" - Site qualities to emphasize? (accessibility, demographics, design, amenities, etc.)
        9. "strategic_context" - External factors? (economic trends, policy goals, regional growth, etc.)
        10. "visual_elements" - Important visuals? (maps, charts, photography, renderings, etc.)
        11. "resources_links" - Supporting resources? (contact info, reports, partner links, etc.)
        12. "special_requests" - Unique angles or topics to avoid?
        13. "timing_stage" - Project stage? (concept, entitlement, marketing, post-construction)
        </OPTIONAL FIELDS>

        <CRITICAL RULES>
        1. Ask one or two focused questions at a time
        2. After EVERY user answer, call update_placestory_context_tool once with all newly confirmed fields:
        - Use the signature update_placestory_context_tool(fields={{"field_name": "value", ...}})
        - Batch multiple fields in the same call (e.g., address + audience) to avoid repetitive tool calls.
        - Only include fields you just captured or clarified in that turn.
        3. When user says "generate" or "ready", call generate_placestory_from_context_tool
        4. AFTER calling generate_placestory_from_context_tool, ALWAYS respond with a brief confirmation message like:
        "Your placestory has been generated successfully! Check the results above."
        </CRITICAL RULES>

        <FIELD NAME EXAMPLES>
        CORRECT: update_placestory_context_tool(fields={{"address": "123 Main St"}})
        CORRECT: update_placestory_context_tool(fields={{"asset_type": "office building", "audience": "institutional investors"}})
        WRONG: update_placestory_context_tool(fields={{"Address / Location": "..."}})
        WRONG: update_placestory_context_tool(fields={{"Asset Type / Use": "..."}})
        </FIELD NAME EXAMPLES>

        <AVAILABLE TOOLS>
        - update_placestory_context_tool(fields=dict) - Store one or more fields at once
        - generate_placestory_from_context_tool() - Generate the placestory
        </AVAILABLE TOOLS>
        """



