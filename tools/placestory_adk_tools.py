from typing import Any, Dict
import json
import logging
from decouple import config

from google.adk.tools import FunctionTool, ToolContext

from managers.websocket_manager import manager
from tools.placestory_tools import create_statistical_profile
from tools.gis_tools import gis_executor
from tools.placestory_tools import PLACE_STORY_SYSTEM_PROMPT, generate_placestory_user_prompt
from config.config import session_service
from tools.placestory_tools import _send_placestory_status, format_execution_time
from google.genai import types as genai_types
from utils.genai_client_manager import get_genai_client
from utils.image_generator import process_placestory_images
import time

logger = logging.getLogger(__name__)


async def generate_placestory_from_context(tool_context: ToolContext) -> Dict[str, Any]:
    """
    1. Read placestory_context from session.state
    2. Geocode address if lat/lng not provided
    3. Generate statistical profile from ArcGIS
    4. Call Gemini via google-genai client with PLACE_STORY_SYSTEM_PROMPT + user prompt
    5. Return parsed JSON placestory (wrapped in operations for frontend)
    """

    start_time = time.time() # for calculating approx time taken to generate the placestory

    await _send_placestory_status(
        tool_context,
        step_id="placestory_generation_start",
        label="PlaceStory generation started",
        status="in_progress",
        details="Starting PlaceStory generation from the captured context.",
    )

    session = tool_context.session
    state = session.state
    context: Dict[str, Any] = state.get("placestory_context", {}) or {}


    await _send_placestory_status(
        tool_context,
        step_id="validate_fields",
        label="Validating fields",
        status="in_progress",
        details="Checking if all required fields are present.",
    )

    required_fields = [
        "address",
        "asset_type",
        "audience",
        "intent_of_story",
        "intended_action",
        "tone_voice",
    ]

    missing = [f for f in required_fields if not context.get(f)]
    if missing:
        print(f"❌ Missing required fields: {missing}")

        missing_md_list = "\n".join(
            f"- **{field.replace('_', ' ').title()}**" for field in missing
        )
        markdown_details = (
            "I still need the following info before I can generate your Placestory:\n"
            f"{missing_md_list}\n\n"
            "Please provide each one and I'll store it automatically."
        )

        await _send_placestory_status(
            tool_context,
            step_id="validate_fields",
            status="error",
            details=markdown_details,
        )

        return {
            "status": "error",
            "reason": "missing_fields",
            "missing_fields": missing,
        }

    await _send_placestory_status(
        tool_context,
        step_id="validate_fields",
        status="done",
    )

    address = context.get("address")
    latitude = context.get("latitude")
    longitude = context.get("longitude")
    use_case = context.get("intent_of_story")

    # 2. Geocode if needed
    await _send_placestory_status(
        tool_context,
        step_id="geocode",
        label="Geocoding address",
        status="in_progress",
        details="Determining map coordinates from the provided address."
    )

    if (latitude is None or longitude is None) and address:

        geocode_result = await gis_executor.get_coordinates_for_address(
            address, max_locations=1
        )

        if "data" in geocode_result and geocode_result["data"].get("candidates"):
            candidate = geocode_result["data"]["candidates"][0]
            latitude = candidate["location"]["y"]
            longitude = candidate["location"]["x"]

            context["latitude"] = latitude
            context["longitude"] = longitude
            context["geocoded_address"] = candidate.get("address", address)

            state["placestory_context"] = context
            await session_service.update_session(session)

            await _send_placestory_status(
                tool_context,
                step_id="geocode",
                status="done",
            )
        else:
            latitude = None
            longitude = None

            markdown_details = (
                f"I couldn't find a valid address for '{address}'. Could you please try again with a more complete address?"
            )

            await _send_placestory_status(
                tool_context,
                step_id="geocode",
                status="error",
                details=markdown_details,
            )
    else:
        print(f"📍 Using existing coordinates: ({latitude}, {longitude})")


    await _send_placestory_status(
        tool_context,
        step_id="stat_profile",
        label="Statistical profile generation",
        status="in_progress",
        details="Generating statistical profile around the address.",
    )

    statistical_profile = None
    if latitude is not None and longitude is not None and address and use_case:
        statistical_profile = await create_statistical_profile(
            latitude=latitude,
            longitude=longitude,
            address=address,
            use_case=use_case,
        )
        if statistical_profile:
            await _send_placestory_status(
                tool_context,
                step_id="stat_profile",
                status="done",
            )
        else:
            print("⚠️ Statistical profile is empty")
            markdown_details = (
                "I couldn't generate a statistical profile for the address. Please try again with a different use case."
            )
            await _send_placestory_status(
                tool_context,
                step_id="stat_profile",
                status="error",
                details=markdown_details,
            )
    else:
        await _send_placestory_status(
            tool_context,
            step_id="stat_profile",
            details="Missing latitude, longitude, or use case.",
            status="done",
        )

    await _send_placestory_status(
        tool_context,
        step_id="build_prompt",
        label="Prompt building",
        status="in_progress",
        details="Preparing LLM context for PlaceStory generation.",
    )

    llm_context: Dict[str, Any] = {
        **context,
        "statistical_profile": statistical_profile,
    }

    model_name = "gemini-2.5-pro"

    client = get_genai_client()

    system_prompt = PLACE_STORY_SYSTEM_PROMPT
    user_prompt = generate_placestory_user_prompt(llm_context)

    full_prompt = system_prompt + "\n\n" + user_prompt

    await _send_placestory_status(
        tool_context,
        step_id="build_prompt",
        status="done",
    )


    await _send_placestory_status(
        tool_context,
        step_id="generate_placestory",
        label="Generating PlaceStory",
        status="in_progress",
        details="Using Gemini to generate PlaceStory content fromt the captured context.",
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=full_prompt)],
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


    try:
        placestory_json = json.loads(text)

        await _send_placestory_status(
            tool_context,
            step_id="generate_placestory",
            status="done",
        )


        await _send_placestory_status(
            tool_context,
            step_id="process_images",
            label="Generating PlaceStory images",
            status="in_progress",
            details="Creating HD & highly relevant AI-generated images for the PlaceStory.",
        )

        placestory_json = await process_placestory_images(placestory_json, tool_context)

        await _send_placestory_status(
            tool_context,
            step_id="process_images",
            status="done",
        )

        # Wrap in operations so UnifiedEventHandler can direct-pass to frontend
        result = {
            "operations": [
                {
                    "type": "PLACESTORY_GENERATED",
                    "payload": placestory_json,
                }
            ]
        }
        await _send_placestory_status(
        tool_context,
        step_id="placestory_generation_start",
        status="done",
        )

        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_formatted = format_execution_time(execution_time)    

        await _send_placestory_status(
            tool_context,
            step_id="placestory_generation_completed",
            label="PlaceStory generation completed",
            details=f"### ⚡ PlaceStory generated successfully in **{execution_time_formatted}**.",
            status="done",
        )


        return json.dumps(result, indent=0)
    except json.JSONDecodeError as e:
        print(f"Failed to parse placestory JSON: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})



async def update_placestory_context(
    fields: Dict[str, Any],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    session = tool_context.session

    context = session.state.get("placestory_context", {}) or {}

    context.update(fields)

    session.state["placestory_context"] = context
    await session_service.update_session(session)

    return json.dumps(context, indent=0)


update_placestory_context_tool = FunctionTool(update_placestory_context)
generate_placestory_from_context_tool = FunctionTool(generate_placestory_from_context)
