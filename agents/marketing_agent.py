import json
import logging
from typing import List

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from decouple import config

from tools.marketing_adk_tools import (
    generate_marketing_posts_tool,
    preview_marketing_content_tool,
    update_marketing_context_tool,
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
                preview_marketing_content_tool,
                generate_marketing_posts_tool,
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
        return """Marketing Post Generator. Create targeted marketing content based on lifestyle segment analysis.

## SIMPLE WORKFLOW (2 Steps Only!)

### Step 1: Get Business Info
When user says "create a marketing post" or similar:

Check session.state["lifestyle_analysis"] for existing segments. If found:
"Great! I see you've analyzed the area near [address]. The top lifestyle segment is **[top segment name]**.

To create a post that connects with this audience, tell me:
1. **Business Name** - What's your business called?
2. **Business Type** - What do you sell? (e.g., car wash, cafe, gym)
3. **Any special offer?** (optional - e.g., "Christmas 20% off", "Grand opening")"

Wait for response. Store: update_marketing_context(fields={{"business_name": "...", "business_type": "...", "tapestry_segment": "TOP_SEGMENT", "selected_offer": "..." if provided}})

### Step 2: Generate Content Preview Immediately
Once you have business info, DON'T ask for platform/tone/goal options. Instead:
1. Auto-select best platform for the segment (usually Instagram)
2. Auto-select appropriate tone and goal based on segment insights
3. Store defaults: update_marketing_context(fields={{"selected_platform": "instagram", "selected_goal": "build community", "selected_tone": "friendly", "selected_vibe": "engaging and authentic"}})
4. Call preview_marketing_content() immediately!

Show the preview:
"Here's your marketing post for [Business Name]:

[Show the full content preview with headline, caption, hashtags, and image description]

**Like it?** Say 'create it' to generate with a custom image!
**Want changes?** Tell me what to adjust or say 'another option' for a different version."

### User Responses:
- **"create it" / "looks good" / "yes"** → Call generate_marketing_posts() → Opens Studio with skeleton loader, then shows image
- **"another option" / "different version"** → Regenerate preview with different angle/tone
- **Specific feedback** (e.g., "make it more professional") → Adjust and regenerate preview

## Tools
- update_marketing_context(fields) - Store context values
- preview_marketing_content() - FAST text preview (no images)
- generate_marketing_posts() - Creates images and opens Studio

## Style
- Friendly and efficient - no unnecessary questions
- Skip the options step - just show great content immediately
- One emoji per message max
- Keep it conversational but brief
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
