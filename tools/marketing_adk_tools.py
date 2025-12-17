from typing import Any, Dict, List, Optional
import json
import logging
import uuid
from datetime import datetime

from google.adk.tools import FunctionTool, ToolContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from decouple import config

from config.config import in_memory_session_service
from utils.genai_client_manager import get_genai_client
from tools.rag_tools import retrieve_tapestry_insights_vertex_rag
from utils.image_generator import generate_and_save_image_async
from utils.Tommy_Prompt import get_tommy_context, get_tommy_logo_image
from managers.websocket_manager import manager as ws_manager
from utils.context_reducer import summarize_tapestry_insights, reduce_context_size
from utils.rate_limiter import get_rate_limiter, estimate_tokens
from utils.error_handlers import ErrorRecovery

logger = logging.getLogger(__name__)


async def _get_tapestry_insights(
    tapestry_segment: str,
    tool_context: ToolContext = None,
) -> str:
    """
    Get tapestry insights from cache or fetch via RAG if not available.

    This function checks if insights are already cached in the session state for the
    same segment. If cached insights exist and the segment matches, it returns them.
    Otherwise, it fetches insights via RAG and stores them in the context.

    Args:
        tapestry_segment: The target tapestry segment name
        tool_context: ADK tool context with session information

    Returns:
        Tapestry insights string
    """
    # If no tool_context, fetch directly (fallback for backward compatibility)
    if not tool_context or not tool_context.session:
        logger.warning("No tool_context provided, fetching insights directly")
        rag_query = f"Tell me about {tapestry_segment} tapestry segment: demographics, lifestyle, preferences, income level, and digital behavior"
        return await retrieve_tapestry_insights_vertex_rag(rag_query)

    session = tool_context.session
    state = session.state
    context: Dict[str, Any] = state.get("marketing_context", {}) or {}

    # Check if we have cached insights for this segment
    cached_segment = context.get("tapestry_segment_for_insights")
    cached_insights = context.get("tapestry_insights")

    # If we have cached insights and the segment matches, return cached version
    if cached_insights and cached_segment == tapestry_segment:
        logger.info(f"Using cached tapestry insights for segment: {tapestry_segment}")
        return cached_insights

    # Fetch insights via RAG
    logger.info(f"Fetching tapestry insights via RAG for segment: {tapestry_segment}")
    rag_query = f"Tell me about {tapestry_segment} tapestry segment: demographics, lifestyle, preferences, income level, and digital behavior"
    tapestry_insights = await retrieve_tapestry_insights_vertex_rag(rag_query)

    # Store insights in context for future use
    context["tapestry_insights"] = tapestry_insights
    context["tapestry_segment_for_insights"] = tapestry_segment
    session.state["marketing_context"] = context

    # Create event to track state change
    event_id = str(uuid.uuid4())
    event = Event(
        id=event_id,
        invocation_id=event_id,
        author="system",
        actions=EventActions(state_delta={"marketing_context": context}),
        timestamp=datetime.now().timestamp(),
    )

    # Append event to in-memory session
    await in_memory_session_service.append_event(session, event)

    logger.info(f"Cached tapestry insights for segment: {tapestry_segment}")
    return tapestry_insights


async def suggest_all_marketing_options(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate all marketing suggestions (platforms, goals_and_offers, tones_and_vibes) in a single call.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with all suggestions (platforms, goals_and_offers, tones_and_vibes)
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )

        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, generate comprehensive marketing suggestions for this audience.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Generate ALL of the following suggestions in a single response:

1. PLATFORMS (2-3 options): Suggest best marketing platforms (LinkedIn, Facebook, Instagram, Email, Twitter/X, TikTok, YouTube)
2. GOALS_AND_OFFERS (2-3 options): Suggest combined campaign goals and offer types. Each suggestion should include both a goal (e.g., increase awareness, drive foot traffic, promote event, generate leads, boost sales, build community, launch product, seasonal promotion) AND a matching offer type (e.g., first time discount, limited time offer, buy one get one, free consultation, early bird special, loyalty reward, referral bonus, seasonal promotion)
3. TONES_AND_VIBES (2-3 options): Suggest combined tone and vibe. Each suggestion should include both a tone (e.g., professional, casual, friendly, inspirational, urgent, playful, sophisticated, authentic, energetic) AND a matching vibe/mood description (e.g., "fun, fast, funny, and focused on getting clean cars quickly" or "professional, trustworthy, and detail-oriented")

Return ONLY a JSON object with this exact structure:
{{
  "platforms": [
    {{"platform": "linkedin", "reason": "brief reason why this platform fits"}},
    {{"platform": "instagram", "reason": "brief reason why this platform fits"}}
  ],
  "goals_and_offers": [
    {{"goal": "increase awareness", "offer": "first time discount offer", "reason": "brief reason why this combination fits"}},
    {{"goal": "generate leads", "offer": "limited time offer", "reason": "brief reason why this combination fits"}}
  ],
  "tones_and_vibes": [
    {{"tone": "professional", "vibe": "trustworthy, detail-oriented, and focused on quality", "reason": "brief reason why this combination fits"}},
    {{"tone": "friendly", "vibe": "fun, fast, and engaging", "reason": "brief reason why this combination fits"}}
  ]
}}

Return ONLY the JSON object, no markdown, no code blocks, no additional text."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        # Clean up code fence markers
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        suggestions_json = json.loads(text)
        return json.dumps(suggestions_json, indent=2)

    except Exception as e:
        logger.error(f"Error generating all marketing suggestions: {e}")
        # Return default suggestions as fallback
        return json.dumps(
            {
                "platforms": [
                    {
                        "platform": "facebook",
                        "reason": "Broad reach platform suitable for most audiences.",
                    },
                    {
                        "platform": "instagram",
                        "reason": "Visual platform effective for engaging content.",
                    },
                ],
                "goals_and_offers": [
                    {
                        "goal": "increase awareness",
                        "offer": "first time discount offer",
                        "reason": "Effective combination for reaching new audiences and attracting first-time customers.",
                    },
                    {
                        "goal": "generate leads",
                        "offer": "limited time offer",
                        "reason": "Creates urgency and drives immediate engagement and conversions.",
                    },
                ],
                "tones_and_vibes": [
                    {
                        "tone": "professional",
                        "vibe": "trustworthy, detail-oriented, and focused on quality",
                        "reason": "Appropriate for most business contexts, builds credibility and confidence.",
                    },
                    {
                        "tone": "friendly",
                        "vibe": "fun, fast, and engaging",
                        "reason": "Engaging and approachable for broad audiences, appeals to modern, active users.",
                    },
                ],
            },
            indent=2,
        )


async def suggest_platforms(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate platform suggestions based on tapestry segment insights.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with platform suggestions
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, suggest the best marketing platforms for this audience.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Consider platforms like: LinkedIn, Facebook, Instagram, Email, Twitter/X, TikTok, YouTube

Return ONLY a JSON array with 2-3 platform suggestions, each with:
- "platform": platform name (lowercase, e.g., "linkedin", "facebook", "instagram", "email")
- "reason": brief reason why this platform fits (1-2 sentences)

Example format:
[
  {{"platform": "linkedin", "reason": "This segment consists of professionals who value thought leadership content."}},
  {{"platform": "instagram", "reason": "Young, visual-oriented audience that engages with lifestyle content."}}
]

Return ONLY the JSON array, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        # Clean up code fence markers
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        platforms_json = json.loads(text)
        return json.dumps({"suggestions": platforms_json}, indent=2)

    except Exception as e:
        logger.error(f"Error generating platform suggestions: {e}")
        return json.dumps(
            {
                "suggestions": [
                    {
                        "platform": "facebook",
                        "reason": "Broad reach platform suitable for most audiences.",
                    },
                    {
                        "platform": "instagram",
                        "reason": "Visual platform effective for engaging content.",
                    },
                ]
            }
        )


async def suggest_goals(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate campaign goal suggestions based on tapestry segment insights.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with goal suggestions
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, suggest 2-3 appropriate campaign goals for this audience.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Consider goals like: increase awareness, drive foot traffic, promote event, generate leads, boost sales, build community, launch product, seasonal promotion

Return ONLY a JSON array with 2-3 goal suggestions, each with:
- "goal": goal name (e.g., "increase awareness", "drive foot traffic")
- "reason": brief reason why this goal fits (1-2 sentences)

Example format:
[
  {{"goal": "increase awareness", "reason": "This segment is new to the market and needs brand introduction."}},
  {{"goal": "generate leads", "reason": "High-intent audience ready to engage with offers."}}
]

Return ONLY the JSON array, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        goals_json = json.loads(text)
        return json.dumps({"suggestions": goals_json}, indent=2)

    except Exception as e:
        logger.error(f"Error generating goal suggestions: {e}")
        return json.dumps(
            {
                "suggestions": [
                    {
                        "goal": "increase awareness",
                        "reason": "Effective for reaching new audiences.",
                    },
                    {
                        "goal": "generate leads",
                        "reason": "Drives engagement and conversions.",
                    },
                ]
            }
        )


async def suggest_tones(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate tone suggestions based on tapestry segment insights.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with tone suggestions
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, suggest 2-3 appropriate tones for marketing content.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Consider tones like: professional, casual, friendly, inspirational, urgent, playful, sophisticated, authentic, energetic

Return ONLY a JSON array with 2-3 tone suggestions, each with:
- "tone": tone name (e.g., "professional", "casual", "friendly")
- "reason": brief reason why this tone fits (1-2 sentences)

Example format:
[
  {{"tone": "professional", "reason": "This segment values credibility and expertise."}},
  {{"tone": "friendly", "reason": "Approachable tone resonates with community-focused audience."}}
]

Return ONLY the JSON array, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        tones_json = json.loads(text)
        return json.dumps({"suggestions": tones_json}, indent=2)

    except Exception as e:
        logger.error(f"Error generating tone suggestions: {e}")
        return json.dumps(
            {
                "suggestions": [
                    {
                        "tone": "professional",
                        "reason": "Appropriate for most business contexts.",
                    },
                    {
                        "tone": "friendly",
                        "reason": "Engaging and approachable for broad audiences.",
                    },
                ]
            }
        )


async def suggest_offers(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate offer suggestions based on tapestry segment insights.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with offer suggestions
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, suggest 2-3 appropriate offer types for marketing campaigns.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Consider offers like: first time discount, limited time offer, buy one get one, free consultation, early bird special, loyalty reward, referral bonus, seasonal promotion

Return ONLY a JSON array with 2-3 offer suggestions, each with:
- "offer": offer type description (e.g., "first time discount offer", "limited time 20% off")
- "reason": brief reason why this offer fits (1-2 sentences)

Example format:
[
  {{"offer": "first time discount offer", "reason": "Attracts new customers who are price-sensitive."}},
  {{"offer": "limited time offer", "reason": "Creates urgency for segments that value exclusivity."}}
]

Return ONLY the JSON array, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        offers_json = json.loads(text)
        return json.dumps({"suggestions": offers_json}, indent=2)

    except Exception as e:
        logger.error(f"Error generating offer suggestions: {e}")
        return json.dumps(
            {
                "suggestions": [
                    {
                        "offer": "first time discount offer",
                        "reason": "Effective for attracting new customers.",
                    },
                    {
                        "offer": "limited time offer",
                        "reason": "Creates urgency and drives action.",
                    },
                ]
            }
        )


async def suggest_vibes(
    tapestry_segment: str,
    tapestry_insights: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate vibe suggestions based on tapestry segment insights.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment (optional, will fetch if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with vibe suggestions
    """
    try:
        # Fetch insights if not provided (use cached version if available)
        if not tapestry_insights:
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following tapestry segment information, suggest 2-3 appropriate vibes/moods for marketing content.

Tapestry Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Consider vibes like: fun and fast, professional and trustworthy, energetic and exciting, calm and reassuring, trendy and modern, authentic and relatable

Return ONLY a JSON array with 2-3 vibe suggestions, each with:
- "vibe": vibe description (e.g., "fun, fast, funny, and focused on getting clean cars quickly")
- "reason": brief reason why this vibe fits (1-2 sentences)

Example format:
[
  {{"vibe": "fun, fast, funny, and focused on getting clean cars quickly", "reason": "This segment values efficiency and appreciates humor."}},
  {{"vibe": "professional, trustworthy, and detail-oriented", "reason": "This segment values quality and reliability."}}
]

Return ONLY the JSON array, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        vibes_json = json.loads(text)
        return json.dumps({"suggestions": vibes_json}, indent=2)

    except Exception as e:
        logger.error(f"Error generating vibe suggestions: {e}")
        return json.dumps(
            {
                "suggestions": [
                    {
                        "vibe": "fun, fast, and engaging",
                        "reason": "Appeals to modern, active audiences.",
                    },
                    {
                        "vibe": "professional and trustworthy",
                        "reason": "Builds credibility and confidence.",
                    },
                ]
            }
        )


async def _generate_key_message(
    tapestry_segment: str,
    tapestry_insights: str,
    selected_goal: str,
    selected_offer: str,
    selected_tone: str = "",
    selected_vibe: str = "",
) -> str:
    """
    Generate a key message automatically based on tapestry insights, goal, and offer.

    Args:
        tapestry_segment: The target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment
        selected_goal: User-selected campaign goal
        selected_offer: User-selected offer
        selected_tone: User-selected tone (optional)
        selected_vibe: User-selected vibe (optional)

    Returns:
        Generated key message string
    """
    try:
        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following marketing campaign details, generate a compelling key message for the marketing campaign.

Target Segment: {tapestry_segment}

Segment Insights:
{tapestry_insights}

Campaign Goal: {selected_goal}
Offer: {selected_offer}
{f"Tone: {selected_tone}" if selected_tone else ""}
{f"Vibe: {selected_vibe}" if selected_vibe else ""}

Generate a concise, compelling key message (1-2 sentences) that:
1. Resonates with the target segment based on the insights
2. Aligns with the campaign goal
3. Highlights the offer if provided
4. Captures the essence of what we want to communicate

Return ONLY the key message text, no markdown, no code blocks, no additional explanation."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        # Clean up code fence markers
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        return (
            text
            if text
            else f"Discover what {tapestry_segment} needs with our {selected_offer if selected_offer else 'exclusive offer'}"
        )

    except Exception as e:
        logger.error(f"Error generating key message: {e}")
        # Fallback key message
        return f"Discover what {tapestry_segment} needs with our {selected_offer if selected_offer else 'exclusive offer'}"


async def suggest_content_structure(
    tapestry_segment: str,
    selected_platform: str,
    selected_goal: str,
    selected_tone: str,
    selected_offer: str,
    selected_vibe: str,
    key_message: str = "",
    tool_context: ToolContext = None,
) -> str:
    """
    Generate content structure suggestions based on user selections.

    Args:
        tapestry_segment: The target tapestry segment name
        selected_platform: User-selected platform
        selected_goal: User-selected campaign goal
        selected_tone: User-selected tone
        selected_offer: User-selected offer
        selected_vibe: User-selected vibe
        key_message: The main message to communicate (optional, will be auto-generated if not provided)
        tool_context: ADK tool context

    Returns:
        JSON string with content structure suggestions
    """
    try:
        # Generate key_message if not provided
        if not key_message:
            # Fetch insights to generate key message (use cached version if available)
            tapestry_insights = await _get_tapestry_insights(
                tapestry_segment, tool_context
            )
            key_message = await _generate_key_message(
                tapestry_segment=tapestry_segment,
                tapestry_insights=tapestry_insights,
                selected_goal=selected_goal,
                selected_offer=selected_offer,
                selected_tone=selected_tone,
                selected_vibe=selected_vibe,
            )
            logger.info(f"Auto-generated key message: {key_message[:100]}...")

        client = get_genai_client()
        model_name = "gemini-2.5-flash"

        prompt = f"""Based on the following marketing campaign details, suggest a content structure for the marketing post.

Campaign Details:
- Target Segment: {tapestry_segment}
- Platform: {selected_platform}
- Goal: {selected_goal}
- Tone: {selected_tone}
- Offer: {selected_offer}
- Vibe: {selected_vibe}
- Key Message: {key_message}

Suggest a content structure that includes:
1. Visual elements (e.g., hero image, layout suggestions)
2. Text content structure with:
   - Headline
   - Offer presentation
   - Key benefits
   - Call-to-action (CTA)

Return ONLY a JSON object with this structure:
{{
  "visual_elements": {{
    "hero_image": "description of suggested hero image",
    "layout": "layout suggestion (e.g., 'single column with image at top')"
  }},
  "text_content": {{
    "headline": "suggested headline approach",
    "offer": "how to present the offer",
    "key_benefits": ["benefit 1", "benefit 2", "benefit 3"],
    "cta": "suggested call-to-action"
  }}
}}

Return ONLY the JSON object, no markdown, no code blocks."""

        # Rate limiting to prevent 429 errors
        estimated_tokens = estimate_tokens(prompt)
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire(estimated_tokens)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ],
        )

        text = ""
        for c in response.candidates or []:
            if not c.content:
                continue
            for p in c.content.parts or []:
                if getattr(p, "text", None):
                    text += p.text

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

        structure_json = json.loads(text)
        return json.dumps(structure_json, indent=2)

    except Exception as e:
        logger.error(f"Error generating content structure suggestions: {e}")
        return json.dumps(
            {
                "visual_elements": {
                    "hero_image": "Engaging image related to the key message",
                    "layout": "Standard social media layout",
                },
                "text_content": {
                    "headline": "Compelling headline that captures attention",
                    "offer": "Clear presentation of the offer",
                    "key_benefits": ["Benefit 1", "Benefit 2", "Benefit 3"],
                    "cta": "Clear call-to-action",
                },
            }
        )


async def update_marketing_context(
    fields: Dict[str, Any],
    tool_context: ToolContext,
) -> str:
    """
    Store marketing context fields in session state.

    Note: This updates the in-memory session only. Persistence to database
    happens automatically after agent processing completes by root_agent.

    Args:
        fields: Dictionary of marketing context fields to update
        tool_context: ADK tool context with session information

    Returns:
        JSON string with updated context
    """
    session = tool_context.session

    context = session.state.get("marketing_context", {}) or {}

    # Handle case where fields might be a JSON string or already a dict
    if isinstance(fields, dict):
        # Already a dictionary, use as-is
        pass
    elif isinstance(fields, str):
        # Try to parse as JSON string
        try:
            fields = json.loads(fields)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse fields as JSON: {fields[:100] if len(str(fields)) > 100 else fields}"
            )
            return json.dumps(
                {
                    "error": "Invalid fields format - must be a dictionary or valid JSON string"
                },
                indent=0,
            )
    else:
        # Not a dict or string - invalid type
        logger.error(
            f"Fields must be a dictionary or JSON string, got: {type(fields)} - {fields}"
        )
        return json.dumps(
            {
                "error": f"Fields must be a dictionary or JSON string, got {type(fields).__name__}"
            },
            indent=0,
        )

    context.update(fields)
    session.state["marketing_context"] = context

    # Create event with state_delta to properly track state changes (ADK pattern)
    # This ensures the state change is properly tracked by ADK's session management
    event_id = str(uuid.uuid4())
    event = Event(
        id=event_id,
        invocation_id=event_id,
        author="system",
        actions=EventActions(state_delta={"marketing_context": context}),
        timestamp=datetime.now().timestamp(),
    )

    # Append event to in-memory session to properly track state change
    await in_memory_session_service.append_event(session, event)

    logger.info(
        f"Updated marketing context with fields: {list(fields.keys())}. "
        f"Full context keys: {list(context.keys())}. "
        f"Session state keys: {list(session.state.keys())}"
    )
    return json.dumps(
        {
            "status": "updated",
            "marketing_context": context,
        },
        indent=2,
    )


async def preview_marketing_content(tool_context: ToolContext) -> str:
    """
    Generate a PREVIEW of marketing content (text only, no images).

    This is the fast step - generates content ideas for user approval.
    After user confirms, call generate_marketing_posts() to generate images.

    Args:
        tool_context: ADK tool context with session information

    Returns:
        JSON string with content preview (no images yet)
    """
    session = tool_context.session
    state = session.state
    context: Dict[str, Any] = state.get("marketing_context", {}) or {}

    # Validate required fields
    tapestry_segment = context.get("tapestry_segment")
    if not tapestry_segment:
        return json.dumps({
            "status": "error",
            "message": "Please select a tapestry segment first."
        })

    selected_platform = context.get("selected_platform", "instagram")
    selected_goal = context.get("selected_goal", "increase awareness")
    selected_tone = context.get("selected_tone", "professional")
    selected_offer = context.get("selected_offer", "")
    selected_vibe = context.get("selected_vibe", "")
    business_name = context.get("business_name", "")
    business_type = context.get("business_type", "")

    logger.info(f"Generating content preview for segment: {tapestry_segment}")

    # Get tapestry insights
    tapestry_insights = await _get_tapestry_insights(tapestry_segment, tool_context)
    # Reduce tapestry insights size to prevent token limit issues
    tapestry_insights = summarize_tapestry_insights(tapestry_insights, max_length=1000)

    # Generate key message
    key_message = await _generate_key_message(
        tapestry_segment=tapestry_segment,
        tapestry_insights=tapestry_insights,
        selected_goal=selected_goal,
        selected_offer=selected_offer,
        selected_tone=selected_tone,
        selected_vibe=selected_vibe,
    )
    context["key_message"] = key_message
    session.state["marketing_context"] = context

    # Generate content preview (text only)
    client = get_genai_client()

    prompt = f"""Generate marketing content for a specific business.

CRITICAL: The content MUST be specifically about this business:
- Business Name: **{business_name}**
- Business Type: **{business_type}**
{f"- Special Offer: **{selected_offer}**" if selected_offer else ""}

Target Audience (Tapestry Segment): {tapestry_segment}
Platform: {selected_platform}
Campaign Goal: {selected_goal}
Tone: {selected_tone}
Vibe: {selected_vibe if selected_vibe else selected_tone}

Segment Insights (use to understand audience):
{tapestry_insights}

REQUIREMENTS:
1. The headline MUST include "{business_name}" or reference it directly
2. The body copy MUST describe what "{business_name}" offers as a {business_type}
3. All content must be relevant to a {business_type} business
{f"4. Highlight the special offer: {selected_offer}" if selected_offer else ""}

Generate ONLY ONE post with:
1. **Headline** (MUST mention {business_name}, 5-10 words)
2. **Body Copy** (2-3 sentences about {business_name}'s {business_type} services)
3. **Call to Action** (visit {business_name}, try their services, etc.)
4. **Hashtags** (3-5 relevant hashtags for {selected_platform} including #{business_name.replace(' ', '').replace("'", '')} if possible)
5. **Image Suggestion** (describe ideal image showing {business_type} services)

Format as clean markdown, no JSON."""

    # Estimate tokens and acquire rate limiter permission
    estimated_tokens = estimate_tokens(prompt)
    rate_limiter = get_rate_limiter()
    await rate_limiter.acquire(estimated_tokens)
    
    # Make API call with 429 error handling
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
            )
            break  # Success, exit retry loop
        except Exception as e:
            # Check if it's a 429 error
            if ErrorRecovery.is_429_error(e):
                if attempt < max_retries - 1:
                    logger.warning(f"429 error on attempt {attempt + 1}, handling backoff...")
                    await ErrorRecovery.handle_429_error(e)
                    continue
                else:
                    logger.error(f"429 error after {max_retries} attempts, giving up")
                    raise
            else:
                # Not a 429 error, re-raise immediately
                raise
    
    if response is None:
        raise Exception("Failed to generate content after retries")

    text = ""
    for c in response.candidates or []:
        if c.content:
            for p in c.content.parts or []:
                if hasattr(p, "text"):
                    text += p.text

    # Store preview in context
    context["content_preview"] = text.strip()
    context["preview_approved"] = False
    session.state["marketing_context"] = context

    logger.info("Content preview generated successfully")

    return json.dumps({
        "status": "preview",
        "platform": selected_platform,
        "segment": tapestry_segment,
        "content": text.strip(),
        "message": "Here's your content preview! Say 'looks good' or 'generate it' to create the final post with images, or tell me what to change."
    }, indent=2)


async def generate_marketing_posts(tool_context: ToolContext) -> str:
    """
    Generate platform-specific marketing posts WITH IMAGES.

    Call this AFTER user approves the content preview from preview_marketing_content().
    This step generates the actual images and sends to frontend.

    Workflow:
    1. Validate required fields (tapestry_segment)
    2. Retrieve tapestry segment insights via RAG
    3. Auto-generate key_message from insights if not provided
    4. Determine best platforms based on demographics
    5. Generate optimized posts for each platform
    6. Generate images for each post
    7. Return in operations format for frontend display

    Args:
        tool_context: ADK tool context with session information

    Returns:
        JSON string with operations containing generated posts
    """
    session = tool_context.session
    state = session.state
    context: Dict[str, Any] = state.get("marketing_context", {}) or {}

    # Step 1: Validate required fields
    required_fields = ["tapestry_segment"]
    missing = [f for f in required_fields if not context.get(f)]

    if missing:
        logger.warning(
            f"Missing required fields for marketing post generation: {missing}"
        )
        missing_md_list = "\n".join(
            f"- **{field.replace('_', ' ').title()}**" for field in missing
        )
        error_response = {
            "status": "error",
            "reason": "missing_fields",
            "missing_fields": missing,
            "message": (
                "I still need the following information before I can generate marketing posts:\n"
                f"{missing_md_list}\n\n"
                "Please provide each one and I'll store it automatically."
            ),
        }
        return json.dumps(error_response, indent=0)

    tapestry_segment = context.get("tapestry_segment")
    # Use selected values if available, otherwise fall back to defaults
    selected_platform = context.get("selected_platform")
    selected_goal = context.get("selected_goal") or context.get(
        "campaign_goal", "increase awareness"
    )
    selected_tone = context.get("selected_tone") or context.get(
        "tone", "professional and engaging"
    )
    selected_offer = context.get("selected_offer", "")
    selected_vibe = context.get("selected_vibe", "")
    key_message = context.get("key_message", "")
    target_action = context.get("target_action", "learn more")
    additional_context = context.get("additional_context", "")
    content_structure = context.get("content_structure", {})

    logger.info(f"Generating marketing posts for segment: {tapestry_segment}")

    # Step 2: Retrieve tapestry segment insights (use cached version if available)
    tapestry_insights = ""
    try:
        tapestry_insights = await _get_tapestry_insights(tapestry_segment, tool_context)
        # Reduce tapestry insights size to prevent token limit issues
        tapestry_insights = summarize_tapestry_insights(tapestry_insights, max_length=1000)
        logger.info(f"Retrieved and reduced tapestry insights: {len(tapestry_insights)} chars")
    except Exception as e:
        logger.error(f"Error retrieving tapestry insights: {e}")
        tapestry_insights = f"General insights for {tapestry_segment} segment"

    # Step 2.5: Generate key_message automatically if not provided
    if not key_message:
        key_message = await _generate_key_message(
            tapestry_segment=tapestry_segment,
            tapestry_insights=tapestry_insights,
            selected_goal=selected_goal,
            selected_offer=selected_offer,
            selected_tone=selected_tone,
            selected_vibe=selected_vibe,
        )
        logger.info(f"Auto-generated key message: {key_message[:100]}...")
        # Store the generated key_message in context for future reference
        context["key_message"] = key_message
        session.state["marketing_context"] = context

    # Step 3: Determine platforms - use selected platform or determine from insights
    if selected_platform:
        # Parse selected platform if it's a JSON string
        try:
            platform_data = (
                json.loads(selected_platform)
                if isinstance(selected_platform, str)
                and selected_platform.startswith("{")
                else selected_platform
            )
            if isinstance(platform_data, dict) and "platform" in platform_data:
                recommended_platforms = [platform_data["platform"]]
            elif isinstance(platform_data, list) and len(platform_data) > 0:
                recommended_platforms = [
                    p["platform"] if isinstance(p, dict) else p
                    for p in platform_data[:3]
                ]
            elif isinstance(platform_data, str):
                recommended_platforms = [platform_data]
            else:
                recommended_platforms = _determine_platforms(
                    tapestry_segment, tapestry_insights
                )
        except:
            recommended_platforms = (
                [selected_platform]
                if isinstance(selected_platform, str)
                else _determine_platforms(tapestry_segment, tapestry_insights)
            )
    else:
        recommended_platforms = _determine_platforms(
            tapestry_segment, tapestry_insights
        )
    logger.info(f"Using platforms: {recommended_platforms}")

    # Step 4: Generate platform-specific posts
    model_name = "gemini-2.5-pro"
    client = get_genai_client()

    system_prompt = _build_marketing_system_prompt()
    user_prompt = _build_marketing_user_prompt(
        tapestry_segment=tapestry_segment,
        campaign_goal=selected_goal,
        key_message=key_message,
        tone=selected_tone,
        target_action=target_action,
        additional_context=additional_context,
        tapestry_insights=tapestry_insights,
        recommended_platforms=recommended_platforms,
        selected_offer=selected_offer,
        selected_vibe=selected_vibe,
        content_structure=content_structure,
        tool_context=tool_context,
    )

    full_prompt = system_prompt + "\n\n" + user_prompt

    logger.info("Calling Gemini to generate marketing posts...")
    logger.info(f"Full Prompt length: {len(full_prompt)} chars")
    
    # Estimate tokens and acquire rate limiter permission
    estimated_tokens = estimate_tokens(full_prompt)
    rate_limiter = get_rate_limiter()
    await rate_limiter.acquire(estimated_tokens)
    
    # Make API call with 429 error handling
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=full_prompt)],
                    )
                ],
            )
            break  # Success, exit retry loop
        except Exception as e:
            # Check if it's a 429 error
            if ErrorRecovery.is_429_error(e):
                if attempt < max_retries - 1:
                    logger.warning(f"429 error on attempt {attempt + 1}, handling backoff...")
                    await ErrorRecovery.handle_429_error(e)
                    continue
                else:
                    logger.error(f"429 error after {max_retries} attempts, giving up")
                    raise
            else:
                # Not a 429 error, re-raise immediately
                raise
    
    if response is None:
        raise Exception("Failed to generate content after retries")

    # Extract response text
    text = ""
    for c in response.candidates or []:
        if not c.content:
            continue
        for p in c.content.parts or []:
            if getattr(p, "text", None):
                text += p.text

    text = text.strip()

    # Clean up code fence markers if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    if text.lower().startswith("json"):
        text = text[4:].lstrip()

    try:
        posts_json = json.loads(text)
        logger.info("Successfully generated marketing posts")

        # Step 4.5: Send MARKETING_GENERATION_STARTED immediately to open Studio with skeleton loaders
        posts = posts_json.get("posts", {})
        connection_id = tool_context.state.get("connection_id")
        if connection_id:
            # Send initial posts without images - frontend will show skeleton loaders
            initial_posts = []
            for platform, post_data in posts.items():
                initial_posts.append({
                    "id": f"post-{uuid.uuid4().hex[:8]}",
                    "platform": platform,
                    "headline": post_data.get("subject_line", "") or post_data.get("content", "")[:50] + "...",
                    "caption": post_data.get("content", ""),
                    "hashtags": post_data.get("hashtags", []),
                    "imageUrl": None,  # No image yet - will trigger skeleton loader
                    "businessName": context.get("business_name"),
                    "segmentName": tapestry_segment,
                })

            start_operation = {
                "operations": [{
                    "type": "MARKETING_GENERATION_STARTED",
                    "payload": {
                        "posts": initial_posts,
                        "businessName": context.get("business_name"),
                        "businessType": context.get("business_type"),
                        "message": "Generating images for your marketing posts..."
                    }
                }]
            }
            try:
                await ws_manager.send_operations_data(connection_id, start_operation)
                logger.info(f"Sent MARKETING_GENERATION_STARTED operation - Studio will open with skeleton loaders")
            except Exception as e:
                logger.error(f"Failed to send start operation: {e}")

        # Step 5: Generate images for each platform post
        image_index = 0
        for platform, post_data in posts.items():
            try:
                # Generate image prompt based on segment, message, and platform
                visual_desc = ""
                if isinstance(content_structure, dict):
                    visual_elements = content_structure.get("visual_elements", {})
                    visual_desc = visual_elements.get(
                        "hero_image", "engaging marketing image"
                    )
                elif isinstance(content_structure, str):
                    try:
                        cs = json.loads(content_structure)
                        visual_desc = cs.get("visual_elements", {}).get(
                            "hero_image", "engaging marketing image"
                        )
                    except:
                        visual_desc = "engaging marketing image"
                else:
                    visual_desc = "engaging marketing image"

                # Build professional marketing image prompt
                business_name = context.get("business_name", "")
                business_type = context.get("business_type", "")
                selected_offer = context.get("selected_offer", "")

                # Professional marketing advertisement prompt structure
                image_prompt = f"""Create a high-quality, professional social media marketing advertisement image for {platform}.

BUSINESS: {business_name} - {business_type}
HEADLINE: {key_message}
{f"SPECIAL OFFER: {selected_offer}" if selected_offer else ""}
TARGET AUDIENCE: {tapestry_segment}

STYLE REQUIREMENTS:
- Modern, clean, eye-catching design suitable for {platform} advertising
- Professional photography or sleek graphic design aesthetic
- Vibrant, attention-grabbing colors that pop on social media feeds
- Clear visual hierarchy with space for text overlay
- {selected_vibe if selected_vibe else selected_tone} mood and tone
- Include visual elements: {visual_desc}

IMAGE SPECIFICATIONS:
- Square format (1:1 aspect ratio) optimized for social media
- High contrast and saturation for mobile viewing
- Professional lighting and composition
- Should look like a paid advertisement, not a stock photo
- Include subtle branding elements for {business_name}

DO NOT include any text, logos, or watermarks in the image - just the visual content."""

                # Add Tommy context and get logo image if enabled
                context_image = None
                if _should_include_tommy_prompt(tool_context):
                    tommy_context = get_tommy_context()
                    # Extract key visual/style elements from Tommy context for image generation
                    image_prompt += f" Brand context: {tommy_context}"

                    # Get logo image for multimodal input
                    logo_image = get_tommy_logo_image()
                    if logo_image:
                        context_image = logo_image
                        # Add instruction to incorporate logo
                        image_prompt += " Incorporate the Tommy Terific's Car Wash logo and branding elements naturally into the image design, ensuring brand consistency and recognition. Use the provided logo as a reference for style, colors, and branding elements."

                logger.info(f"Generating image for {platform}...")
                logger.info(f"Image Prompt: {image_prompt}")
                image_url = await generate_and_save_image_async(
                    image_prompt, image_index, context_image=context_image
                )
                image_index += 1

                # Add image URL to post data
                if "image_url" not in post_data:
                    post_data["image_url"] = image_url
                    logger.info(
                        f"Added image URL for {platform}: {image_url[:50] if image_url else 'None'}..."
                    )
            except Exception as e:
                logger.error(f"Error generating image for {platform}: {e}")
                # Continue without image if generation fails
                post_data["image_url"] = None

        # Step 6: Emit MARKETING_POSTS_GENERATED operation to frontend
        connection_id = tool_context.state.get("connection_id")
        if connection_id:
            # Format posts for frontend operation
            marketing_posts = []
            for platform, post_data in posts.items():
                marketing_posts.append({
                    "id": f"post-{uuid.uuid4().hex[:8]}",
                    "platform": platform,
                    "headline": post_data.get("subject_line", "") or post_data.get("content", "")[:50] + "...",
                    "caption": post_data.get("content", ""),
                    "hashtags": post_data.get("hashtags", []),
                    "imageUrl": post_data.get("image_url"),
                    "businessName": context.get("business_name"),
                    "segmentName": tapestry_segment,
                })

            operation = {
                "operations": [{
                    "type": "MARKETING_POSTS_GENERATED",
                    "payload": {
                        "posts": marketing_posts,
                        "businessName": context.get("business_name"),
                        "businessType": context.get("business_type"),
                    }
                }]
            }
            try:
                await ws_manager.send_operations_data(connection_id, operation)
                logger.info(f"Sent MARKETING_POSTS_GENERATED operation with {len(marketing_posts)} posts")
            except Exception as e:
                logger.error(f"Failed to send marketing operation: {e}")
        else:
            logger.warning("No connection_id available, skipping MARKETING_POSTS_GENERATED operation")

        # Format posts as readable text response
        formatted_response = _format_posts_as_text(posts_json)
        return formatted_response
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse marketing posts JSON: {e}")
        return f"I encountered an error generating the marketing posts. Please try again with more specific details. Error: {str(e)}"


def _determine_platforms(tapestry_segment: str, tapestry_insights: str) -> list:
    """
    Determine best marketing platforms based on tapestry segment characteristics.

    Platform mapping logic:
    - LinkedIn: Professional, high-income, B2B segments
    - Facebook: Broad reach, community-focused, family-oriented
    - Instagram: Visual, younger demographics, lifestyle-focused
    - Email: Personalized, detail-oriented, high-intent

    Args:
        tapestry_segment: Target tapestry segment name
        tapestry_insights: RAG-retrieved insights about the segment

    Returns:
        List of recommended platform names
    """
    segment_lower = tapestry_segment.lower()
    insights_lower = tapestry_insights.lower()

    platforms = []

    # LinkedIn indicators: professional, high-income, educated, business
    linkedin_keywords = [
        "professional",
        "high-income",
        "educated",
        "executive",
        "tech",
        "business",
        "career",
    ]
    if any(
        keyword in segment_lower or keyword in insights_lower
        for keyword in linkedin_keywords
    ):
        platforms.append("linkedin")

    # Instagram indicators: young, visual, lifestyle, urban, trendy
    instagram_keywords = [
        "young",
        "millennial",
        "gen z",
        "urban",
        "trendy",
        "lifestyle",
        "fashion",
        "visual",
    ]
    if any(
        keyword in segment_lower or keyword in insights_lower
        for keyword in instagram_keywords
    ):
        platforms.append("instagram")

    # Facebook indicators: community, family, suburban, broad demographics
    facebook_keywords = [
        "family",
        "community",
        "suburban",
        "neighborhood",
        "diverse",
        "local",
    ]
    if any(
        keyword in segment_lower or keyword in insights_lower
        for keyword in facebook_keywords
    ):
        platforms.append("facebook")

    # Email is generally effective for most segments, especially high-intent
    if (
        "premier" in segment_lower
        or "established" in insights_lower
        or "professional" in insights_lower
    ):
        platforms.append("email")

    # Default to Facebook and Instagram if no specific matches
    if not platforms:
        platforms = ["facebook", "instagram"]

    # Return unique platforms, max 3 for focus
    return list(dict.fromkeys(platforms))[:3]


def _should_include_tommy_prompt(tool_context: Optional[ToolContext] = None) -> bool:
    """
    Determine whether to include Tommy's Car Wash context in prompts.

    Checks organization_id from authenticated user_info in session state.
    Returns True if organization_id equals 3 (Tommy's Car Wash).
    Falls back to ENABLE_TOMMY_PROMPT environment variable if user_info is not available.

    Args:
        tool_context: ADK tool context with session information (optional)

    Returns:
        Boolean indicating whether to include Tommy context
    """
    # First, check organization_id from session state if available
    if tool_context and tool_context.session:
        user_info = tool_context.session.state.get("user_info", {})

        if user_info:
            # Try multiple possible field names for organization_id
            organization_id = (
                user_info.get("organization_id")
                or user_info.get("organizationId")
                or user_info.get("org_id")
            )

            # Check if organization_id equals 3 (handle both int and string)
            if organization_id is not None:
                # Convert to int for comparison (handles both "3" and 3)
                try:
                    org_id_int = int(organization_id)
                    if org_id_int == 3:
                        logger.info("Tommy prompt enabled: organization_id == 3")
                        return True
                except (ValueError, TypeError):
                    # If conversion fails, try string comparison
                    if str(organization_id) == "3":
                        logger.info("Tommy prompt enabled: organization_id == '3'")
                        return True

    # Fallback to environment variable for backward compatibility
    enable_tommy_prompt = config("ENABLE_TOMMY_PROMPT", default=False, cast=bool)
    if enable_tommy_prompt:
        logger.info("Tommy prompt enabled via ENABLE_TOMMY_PROMPT environment variable")

    return enable_tommy_prompt


def _build_marketing_system_prompt() -> str:
    """Build comprehensive system prompt with platform-specific guidelines."""
    return """
You are an expert marketing copywriter specializing in platform-specific content optimization.

Your task is to generate engaging, high-converting marketing posts tailored to specific social media platforms and email.

# PLATFORM SPECIFICATIONS

## LinkedIn
- Character Limit: 3,000 (aim for 150-300 for optimal engagement)
- Tone: Professional, insightful, thought-leadership
- Hashtags: 3-5 relevant industry hashtags
- Format: Hook question or stat → Context → Value proposition → CTA
- Best Practices:
  * Start with a compelling question or surprising statistic
  * Use line breaks for readability (double space between paragraphs)
  * Include professional credibility indicators
  * CTA should be business-focused (e.g., "Learn more", "Contact us", "Download guide")

## Facebook
- Character Limit: 63,206 (aim for 40-80 words for optimal engagement)
- Tone: Conversational, community-focused, friendly
- Hashtags: 1-3 relevant hashtags
- Format: Attention-grabbing opener → Story/benefit → CTA
- Best Practices:
  * Use emojis strategically (1-3 per post)
  * Ask questions to encourage comments
  * Create sense of community
  * CTA should be accessible (e.g., "Learn more", "Visit us", "Join us")

## Instagram
- Character Limit: 2,200 (aim for 138-150 characters for optimal engagement)
- Tone: Visual, aspirational, authentic
- Hashtags: 10-30 relevant hashtags (mix of popular and niche)
- Format: Hook → Value/emotion → CTA → Hashtag block
- Best Practices:
  * Lead with strong visual description or emotion
  * Use line breaks for easy mobile reading
  * Emojis are essential (3-5 per post)
  * Mix trending and niche hashtags
  * CTA in first 2 lines (above "more" fold)

## Email
- Subject Line: 40-50 characters optimal
- Preview Text: 35-55 characters
- Body: 50-125 words for promotional emails
- Tone: Personalized, direct, value-focused
- Format: Compelling subject → Preview hook → Value proposition → Clear CTA
- Best Practices:
  * Subject line must create curiosity or urgency
  * Preview text should complement, not repeat subject
  * Use "you" language (personalization)
  * Single clear CTA button/link
  * Mobile-first formatting (short paragraphs)

# OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no extra text) with this exact structure:

{
  "recommended_platforms": ["platform1", "platform2"],
  "posts": {
    "platform_name": {
      "content": "Full post text with proper formatting",
      "hashtags": ["#hashtag1", "#hashtag2"],
      "character_count": 245,
      "best_practices_applied": ["Practice 1", "Practice 2"],
      "subject_line": "Only for email - compelling subject",
      "preview_text": "Only for email - preview text"
    }
  },
  "tapestry_insights": "Brief summary of how tapestry insights influenced content",
  "generation_reasoning": "Why these specific platforms were recommended"
}

# CRITICAL REQUIREMENTS

1. **Platform Optimization**: Each post MUST be uniquely crafted for its platform (don't just repeat the same content)
2. **Hashtag Research**: Use real, relevant hashtags (not generic ones)
3. **Character Counts**: Stay within limits, aim for optimal engagement lengths
4. **Authentic Voice**: Match the tone to both platform and audience segment
5. **Clear CTA**: Every post needs a specific, actionable call-to-action
6. **Mobile-First**: Format for mobile readability (short lines, emojis, breaks)
""".strip()


def _build_marketing_user_prompt(
    tapestry_segment: str,
    campaign_goal: str,
    key_message: str,
    tone: str,
    target_action: str,
    additional_context: str,
    tapestry_insights: str,
    recommended_platforms: list,
    selected_offer: str = "",
    selected_vibe: str = "",
    content_structure: dict = None,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """Build user prompt with collected marketing context."""
    content_structure = content_structure or {}

    # Parse content structure if it's a JSON string
    if isinstance(content_structure, str):
        try:
            content_structure = json.loads(content_structure)
        except:
            content_structure = {}

    structure_text = ""
    if content_structure:
        visual_elements = content_structure.get("visual_elements", {})
        text_content = content_structure.get("text_content", {})

        if visual_elements or text_content:
            structure_text = "\n# CONTENT STRUCTURE\n\n"
            if visual_elements:
                structure_text += f"**Visual Elements:**\n"
                if visual_elements.get("hero_image"):
                    structure_text += f"- Hero Image: {visual_elements['hero_image']}\n"
                if visual_elements.get("layout"):
                    structure_text += f"- Layout: {visual_elements['layout']}\n"

            if text_content:
                structure_text += f"\n**Text Content Structure:**\n"
                if text_content.get("headline"):
                    structure_text += f"- Headline: {text_content['headline']}\n"
                if text_content.get("offer"):
                    structure_text += f"- Offer: {text_content['offer']}\n"
                if text_content.get("key_benefits"):
                    benefits = text_content["key_benefits"]
                    if isinstance(benefits, list):
                        structure_text += f"- Key Benefits: {', '.join(benefits)}\n"
                    else:
                        structure_text += f"- Key Benefits: {benefits}\n"
                if text_content.get("cta"):
                    structure_text += f"- CTA: {text_content['cta']}\n"

    # Check if Tommy context should be included
    tommy_context_text = ""
    if _should_include_tommy_prompt(tool_context):
        tommy_context_text = "\n\n" + get_tommy_context()

    return f"""
Generate marketing posts for the following campaign:

# CAMPAIGN DETAILS

**Target Audience**: {tapestry_segment}
**Campaign Goal**: {campaign_goal}
**Key Message**: {key_message}
**Desired Tone**: {tone}
**Target Action**: {target_action}
{f"**Offer**: {selected_offer}" if selected_offer else ""}
{f"**Vibe**: {selected_vibe}" if selected_vibe else ""}
{f"**Additional Context**: {additional_context}" if additional_context else ""}

# TAPESTRY SEGMENT INSIGHTS

{tapestry_insights}

# PLATFORMS TO GENERATE

Create optimized posts for these platforms: {", ".join(recommended_platforms)}
{structure_text}{tommy_context_text}
# REQUIREMENTS

1. Use the tapestry insights to inform language, references, and appeals
2. Each platform post should feel native to that platform
3. Include relevant, researched hashtags (not generic ones)
4. Apply all platform-specific best practices
5. Ensure CTAs align with the target action: {target_action}
6. Match the overall tone: {tone}
{f"7. Incorporate the vibe: {selected_vibe}" if selected_vibe else ""}
{f"8. Highlight the offer: {selected_offer}" if selected_offer else ""}
{f"9. Follow the content structure guidelines provided above" if structure_text else ""}

Generate the posts now in valid JSON format.
""".strip()


def _format_posts_as_text(posts_json: Dict[str, Any]) -> str:
    """
    Format generated posts JSON as readable markdown text.

    Args:
        posts_json: Generated posts in JSON format

    Returns:
        Formatted markdown string with all posts
    """
    output = []

    # Add header
    output.append("# 🎯 Marketing Posts Generated\n")

    # Add tapestry insights if available
    if "tapestry_insights" in posts_json:
        output.append(f"**Audience Insights:** {posts_json['tapestry_insights']}\n")

    # Add platform recommendations reasoning
    if "generation_reasoning" in posts_json:
        output.append(
            f"**Why these platforms:** {posts_json['generation_reasoning']}\n"
        )

    output.append("---\n")

    # Format each platform's post
    posts = posts_json.get("posts", {})
    platform_emojis = {
        "linkedin": "💼",
        "facebook": "👥",
        "instagram": "📸",
        "email": "📧",
    }

    for platform, post_data in posts.items():
        emoji = platform_emojis.get(platform, "📱")
        output.append(f"\n## {emoji} {platform.upper()}\n")

        # Email has special fields
        if platform == "email":
            if "subject_line" in post_data:
                output.append(f"**Subject:** {post_data['subject_line']}\n")
            if "preview_text" in post_data:
                output.append(f"**Preview:** {post_data['preview_text']}\n")
            output.append("")

        # Image URL if available
        if "image_url" in post_data and post_data["image_url"]:
            output.append(f"**Image:** {post_data['image_url']}\n")

        # Post content
        output.append("**Post:**")
        output.append("```")
        output.append(post_data.get("content", ""))
        output.append("```\n")

        # Hashtags
        if "hashtags" in post_data and post_data["hashtags"]:
            hashtags_str = " ".join(post_data["hashtags"])
            output.append(f"**Hashtags:** {hashtags_str}\n")

        # Metadata
        if "character_count" in post_data:
            output.append(f"**Character Count:** {post_data['character_count']}")

        if "best_practices_applied" in post_data:
            practices = ", ".join(post_data["best_practices_applied"])
            output.append(f"**Best Practices Applied:** {practices}")

        output.append("\n---")

    output.append(
        "\n💡 **Tip:** You can ask me to refine these posts by saying things like:"
    )
    output.append("- 'Make it more casual'")
    output.append("- 'Add more urgency'")
    output.append("- 'Focus only on Instagram'")
    output.append("- 'Change the tone to professional'")

    return "\n".join(output)


# Create ADK FunctionTool instances
update_marketing_context_tool = FunctionTool(update_marketing_context)
preview_marketing_content_tool = FunctionTool(preview_marketing_content)
generate_marketing_posts_tool = FunctionTool(generate_marketing_posts)
suggest_all_marketing_options_tool = FunctionTool(suggest_all_marketing_options)
# Keep individual tools for backward compatibility, but prefer suggest_all_marketing_options
suggest_platforms_tool = FunctionTool(suggest_platforms)
suggest_goals_tool = FunctionTool(suggest_goals)
suggest_tones_tool = FunctionTool(suggest_tones)
suggest_offers_tool = FunctionTool(suggest_offers)
suggest_vibes_tool = FunctionTool(suggest_vibes)
suggest_content_structure_tool = FunctionTool(suggest_content_structure)
