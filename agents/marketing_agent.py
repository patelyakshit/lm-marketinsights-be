import json
import logging
from typing import List

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from decouple import config

from tools.marketing_adk_tools import (
    generate_marketing_posts_tool,
    update_marketing_context_tool,
    suggest_all_marketing_options_tool,
    suggest_content_structure_tool,
)
from agents.streaming_callbacks import (
    after_tool_modifier,
    before_tool_modifier,
    after_model_modifier,
    before_model_modifier,
)

logger = logging.getLogger(__name__)


class MarketingAgent(LlmAgent):
    """
    Marketing Agent for generating platform-specific marketing posts.

    Uses conversational workflow to collect context, performs RAG lookups
    on ArcGIS Tapestry segmentation data, and generates optimized posts
    for Facebook, LinkedIn, Instagram, and Email.
    """

    def __init__(
        self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True
    ):

        final_model = (
            config("SUB_AGENT_MODEL", default=None) or model
            if allow_override
            else model
        )

        super().__init__(
            model=final_model,
            name="marketing_agent",
            description="Chats with the user to collect marketing context and generates platform-specific posts with tapestry insights.",
            instruction=self._get_system_instruction(),
            tools=[
                update_marketing_context_tool,
                generate_marketing_posts_tool,
                suggest_all_marketing_options_tool,
                suggest_content_structure_tool,
            ],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
            # Prevent sub-agent from seeing prior conversation history
            # This avoids verbose re-introductions during agent transfers
            include_contents="default",
        )

        logger.info(f"MarketingAgent initialized with model: {final_model}")

    def _get_system_instruction(self) -> str:
        return """Marketing Post Generator. Call tools directly (no Python code).

## Workflow
1. User mentions segment → update_marketing_context(tapestry_segment) → suggest_all_marketing_options(segment)
2. Present ALL suggestions (platform, goal/offer, tone/vibe) in numbered format
3. Collect selections → update_marketing_context for each
4. Call suggest_content_structure with selections
5. Call generate_marketing_posts → present without code blocks

## Tools
- update_marketing_context(fields={{"field": "value"}}) - Store: tapestry_segment, selected_platform, selected_goal, selected_offer, selected_tone, selected_vibe, content_structure
- suggest_all_marketing_options(tapestry_segment, tapestry_insights?) - Returns all suggestions at once
- suggest_content_structure(segment, platform, goal, tone, offer, vibe) - Content structure (key_message auto-generated)
- generate_marketing_posts() - Final posts with images

## Rules
- Present suggestions numbered: "1. Instagram 2. Facebook 3. Both"
- Wait for selections before proceeding
- No code blocks in output
- Images auto-generated
- key_message auto-generated from insights
"""

    def get_capabilities(self) -> List[str]:
        return [
            "Conversational marketing context collection",
            "Platform-specific post generation (Facebook, LinkedIn, Instagram, Email)",
            "ArcGIS Tapestry segmentation insights integration",
            "Auto-recommendation of best marketing platforms",
            "Conversational refinement and editing support",
            "Hashtag research and best practices application",
        ]
