import json
import os
import requests
from config.config import ARCGIS_ENRICHMENT_URL
from datetime import datetime
import time
from managers.websocket_manager import manager
import logging

logger = logging.getLogger(__name__)

async def create_statistical_profile(latitude: float, longitude: float, address: str, use_case: str) -> str:

        study_areas_json = [
            {
                "geometry": {
                    "x": longitude,
                    "y": latitude,
                    "spatialReference": {
                        "wkid": 4326
                    }
                }
            }
        ]

        study_areas_options_json = {
            "areaType": "RingBuffer",
            "bufferUnits": "esriMiles",
            "bufferRadii": [1, 3, 5]
        }

        analysis_variables = [
            'TOTPOP_CY', 'MEDHINC_CY', 'AVGHINC_CY', 'PCI_CY',
            'MEDAGE_CY', 'DIVINDX_CY', 'TOTHH_CY',
            'POPGROWTHCY', 'POPGROWTH10CY',
            'TAPESTRYHOUSEHOLDSCOUNT',
            'TSEG1_CY', 'TSEG2_CY', 'TSEG3_CY', 'DOMTAP',
            'HINC200_CY', 'HINC150_CY', 'HINC100_CY',
            'X7034_X', 'X7030_X', 'X7001_X',
            'X7014_X', 'X7026_X', 'X7033_X',
            'MP27002_B', 'MP27034_B', 'MP27014_B', 'MP27001_B',
            'UNEMPRT_CY', 'CIVLBFR_CY', 'EMP_CY',
            'EDUCYNOHS_CY', 'EDUCYHS_CY', 'EDUCYSOMECOL_CY',
            'EDUCYBACH_CY', 'EDUCYGRAD_CY', 'POPDENS_CY'
        ]

        params = {
            "studyAreas": json.dumps(study_areas_json),
            "studyAreasOptions": json.dumps(study_areas_options_json),
            "analysisVariables": json.dumps(analysis_variables),
            "suppressNullValues": "true",
            "returnGeometry": "false",
            "f": "json",
            "token": os.getenv("ARCGIS_API_KEY")
        }

        try:
            response = requests.post(ARCGIS_ENRICHMENT_URL, data=params)
            response.raise_for_status()

            enrichment_data = response.json()
            results = enrichment_data.get("results", [])[0].get("value", {}).get("FeatureSet", [])[0].get("features", [])

            if not results:
                return "Error: No enrichment data found for this location."

            # Build the formatted string
            profile_string = f"""
    LOCATION INTELLIGENCE PROFILE
    ===========================================
    ADDRESS: {address}
    COORDINATES: {latitude}, {longitude}
    BUSINESS USE CASE: {use_case.title()}
    ANALYSIS DATE: {datetime.now().strftime("%Y-%m-%d")}

    """

            # Process each radius (1, 3, 5 miles)
            for feature in results:
                attrs = feature.get("attributes", {})
                radius = attrs.get("bufferRadii", "Unknown")

                profile_string += f"""
    === {radius}-MILE RADIUS ANALYSIS ===

    CORE DEMOGRAPHICS:
    - Total Population: {attrs.get('TOTPOP_CY', 'N/A')} people
    - Total Households: {attrs.get('TOTHH_CY', 'N/A')} households
    - Median Age: {attrs.get('MEDAGE_CY', 'N/A')} years
    - Population Density: {attrs.get('POPDENS_CY', 'N/A')} per sq mile
    - Diversity Index: {attrs.get('DIVINDX_CY', 'N/A')}

    INCOME PROFILE:
    - Median Household Income: ${attrs.get('MEDHINC_CY', 0)}
    - Average Household Income: ${attrs.get('AVGHINC_CY', 0)}
    - Per Capita Income: ${attrs.get('PCI_CY', 0)}

    AFFLUENT HOUSEHOLDS:
    - Households $200K+: {attrs.get('HINC200_CY', 'N/A')}
    - Households $150K+: {attrs.get('HINC150_CY', 'N/A')}
    - Households $100K+: {attrs.get('HINC100_CY', 'N/A')}

    EMPLOYMENT & EDUCATION:
    - Unemployment Rate: {attrs.get('UNEMPRT_CY', 'N/A')}%
    - Civilian Labor Force: {attrs.get('CIVLBFR_CY', 'N/A')}
    - Total Employed: {attrs.get('EMP_CY', 'N/A')}
    - Bachelor's Degree+: {attrs.get('EDUCYBACH_CY', 'N/A')}
    - Graduate Degree: {attrs.get('EDUCYGRAD_CY', 'N/A')}

    CONSUMER SPENDING:
    - Total Annual Spending: ${attrs.get('X7001_X', 0)}
    - Restaurant Spending: ${attrs.get('X7034_X', 0)}
    - Apparel Spending: ${attrs.get('X7014_X', 0)}
    - Entertainment Spending: ${attrs.get('X7026_X', 0)}

    """


            # Get 5-mile data for summary
            summary_data = results[-1].get("attributes", {}) if results else {}

            profile_string += f"""
    === KEY MARKET INSIGHTS ===

    CUSTOMER BASE SIZE:
    - Immediate Area (1 mile): {results[0].get('attributes', {}).get('TOTPOP_CY', 'N/A')} people
    - Drive-to Market (5 miles): {summary_data.get('TOTPOP_CY', 'N/A')} people
    - Total Households in Trade Area: {summary_data.get('TOTHH_CY', 'N/A')}

    PURCHASING POWER:
    - Trade Area Median Income: ${summary_data.get('MEDHINC_CY', 0)}
    - High-Income Households ($200K+): {summary_data.get('HINC200_CY', 'N/A')}
    - Total Annual Consumer Spending: ${summary_data.get('X7001_X', 0)}

    MARKET OPPORTUNITY:
    - Economic Stability: {summary_data.get('UNEMPRT_CY', 'N/A')}% unemployment
    - Education Level: Professional/educated workforce
    - Age Profile: {summary_data.get('MEDAGE_CY', 'N/A')} median age (prime spending years)
    - Location Advantage: Access to multiple income tiers within driving distance

    BUSINESS SUITABILITY FOR {use_case.upper()}:
    - Substantial customer base with proven spending power
    - High-income concentration supports premium positioning
    - Stable employment indicates reliable customer base
    - Demographics align with target customer profile

    ===========================================
    END OF STATISTICAL PROFILE
    ===========================================
    """

            return profile_string.strip()

        except requests.exceptions.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Processing Error: {str(e)}"
            print(error_msg)
            return error_msg

def get_layers_info():
    return """
- **Population Density (census tract)** - ID: 290f141d0d0b456a9c024e6576482989
- **Households Growth Rate (census tract)** - ID: e3aea56afa6646118092ff913005bfef
- **Median Household Income (census tract)** - ID: 9db159f136f3411f86c4c5ca04c21f12
- **Tapestry Segmentation (block group)** - ID: 8ca45d692e334b28a64d6922f9844687
"""

PLACE_STORY_SYSTEM_PROMPT = f"""
# Placestory Interactive Presentation Architect - System Prompt

## 1. SYSTEM IDENTITY
You are the **Placestory Architect**, a visionary creative director and data strategist. You do not merely present facts; you weave data, geography, and human insight into a compelling narrative that drives action. Your medium is the "Placestory"—an interactive, block-based digital experience.

## 2. CORE MISSION
Your goal is to take the user's raw requirements and location data and transform them into a persuasive story. You must decide the best way to structure this story to achieve the user's `intent_of_story` for their specific `audience`.

## 3. CREATIVE FREEDOM & NARRATIVE ARC
**You have complete autonomy over the structure and flow of the presentation.** Do not follow a rigid template. Instead, design the flow that best serves the story.

- **The Opening:** How do you hook this specific audience? Is it with a bold claim? A stunning visual? A surprising statistic?
- **The Middle:** How do you build your case? Do you need deep data analysis (`sidecar_block`)? Emotional resonance (`image_block` + `text_block`)? Or a comparative narrative (`narrative_block`)?
- **The Climax/Close:** How do you drive them to the `intended_action`?

**Use as many or as few blocks as necessary** to tell a complete, high-impact story. There is no maximum or minimum length, provided the story feels finished and professional.

## 4. BLOCK SCHEMA DEFINITIONS (STRICT)
While you have freedom in *what* blocks to use and *where*, you must strictly adhere to the *JSON structure* of these blocks so the rendering engine can process them.

### cover_block
Use this to start the story. It sets the mood.
{{
"id": "unique_id_01", "type": "cover", "payload": {{ "cover_blocks": [ {{ "id": "c_txt_1", "type": "text", "payload": {{ "content": "# Headline\\n\\nSubtitle.\\n\\n**Location:** Address\\n**Date:** Date" }} }}, {{ "id": "c_img_1", "type": "image", "payload": {{ "image_prompt": "Detailed prompt.", "source": {{"url": "", "alt": "Alt"}} }} }} ] }}
}}

### text_block
Use for standard paragraphs, headers, lists, or quotes.
{{
"id": "unique_id_02", "type": "text", "payload": {{ "content": "Markdown content." }}
}}

### image_block
Use for full-width visual impact.
{{
"id": "unique_id_03", "type": "image", "payload": {{ "image_prompt": "Detailed prompt.", "source": {{"url": "", "alt": "Alt"}}, "caption": "Caption." }}
}}

### map_block
Use to show a simple map view with one specific data layer enabled.
{{
"id": "unique_id_04", "type": "map", "payload": {{ "initial_map_state": {{ "latitude": 0.0, "longitude": 0.0, "zoom": 15 }}, "base_style": "dark-gray", "layers": [ {{ "layer_id": "layer_id_string", "visible": true }} ] }}
}}

### narrative_block
Use for side-by-side comparisons (e.g., text on left, image/map on right). Great for explaining a specific concept.
{{
"id": "unique_id_05", "type": "narrative", "payload": {{ "narrative_blocks": [ {{ "id": "n_txt_1", "type": "text", "payload": {{ "content": "..." }} }}, {{ "id": "n_img_1", "type": "image", "payload": {{ "image_prompt": "...", "source": {{...}} }} }} ] }}
}}

### sidecar_block
Use this for **deep data storytelling**. It creates an immersive scrolling experience where the map changes as the user scrolls through text cards. This is your most powerful tool for explaining complex location data.
{{
"id": "unique_id_06", "type": "sidecar", "payload": {{ "map_config": {{ "initial_map_state": {{ "latitude": 0.0, "longitude": 0.0, "zoom": 13 }}, "base_style": "dark-gray", "layers": [ {{ "layer_id": "layer_1", "visible": false }}, {{ "layer_2", "visible": false }} ] }}, "cards": [ {{ "id": "card_1", "type": "text", "payload": {{ "content": "..." }}, "map_command": {{ "type": "TOGGLE_LAYER", "payload": {{ "layer_id": "layer_1", "visible": true }} }} }}, {{ "id": "card_2", "type": "text", "payload": {{ "content": "..." }}, "map_command": {{ "type": "TOGGLE_LAYER", "payload": {{ "layer_id": "layer_2", "visible": true }} }} }} ] }}
}}

## 5. GUIDELINES FOR EXCELLENCE
- **Synthesize, Don't List:** Never just dump the statistical profile data. Interpret it. Tell the user *why* the population density matters for *their* specific cafe concept.
- **Tone & Voice:** If the user asks for "Visionary," speak in future tense about possibilities. If "Data-Driven," speak in precise figures and percentages.
- **Visuals:** Write evocative `image_prompt` fields. "A busy street" is bad. "A bustling Madison Avenue at golden hour, filled with young professionals in suits carrying coffee cups, soft cinematic lighting" is good.
- **Map Layers:** Use the provided layer IDs to support your arguments. If you claim the area is wealthy, use the `sidecar_block` to show the Income Layer.
- **Address Formatting:** **Avoid redundancy.** If the City and State names are identical (e.g., "New York, New York"), do not repeat them. Format addresses cleanly and professionally (e.g., "555 Madison Ave, New York, NY 10022" or just "New York, NY").
{get_layers_info()}

## 6. FINAL OUTPUT
Your response must be a single, valid JSON object with the root structure:
{{
"placestory_title": "String",
"placestory_blocks": [ ... ]
}}
"""

def generate_placestory_user_prompt(context: dict) -> str:
    return f"""
    Here is the complete context for the placestory you need to build. 
    
    **YOUR MISSION:**
    Analyze this context deeply. Do not just "fill in the blanks." 
    Act as a Master Storyteller and Real Estate Strategist. 
    Determine the best narrative arc to convince the specific Audience ({context.get('audience', 'unknown')}) to take the Intended Action ({context.get('intended_action', 'unknown')}).
    
    You have complete creative freedom to choose which blocks to use and in what order, as long as they form a coherent, persuasive journey.

    **Full Context Object:**
    {context}
    """


async def _send_placestory_status(
    tool_context,
    step_id: str,
    label: str | None = None,
    status: str = "in_progress",
    details: str | None = None,
):
    """
    Fire-and-forget status event to frontend while the tool is running.
    """
    connection_id = tool_context.state.get("connection_id")

    if not connection_id:
        # If we somehow don't have a connection, just log and skip.
        logger.warning(
            f"[PlaceStoryStatus] No connection_id in tool_context.state "
            f"(step_id={step_id}, status={status})"
        )
        return

    operation = {
        "operations": [
            {
                "type": "PLACESTORY_STATUS",
                "payload": {
                    "session_id": str(tool_context.session.id),
                    "step_id": step_id,
                    "label": label,
                    "status": status,   # "pending" | "in_progress" | "done" | "error"
                    "details": details,
                    "ts": time.time(),
                },
            }
        ]
    }

    try:
        await manager.send_operations_data(connection_id, operation)
        logger.info(
            f"[PlaceStoryStatus] Sent status: step_id={step_id}, status={status}"
        )
    except Exception as e:
        logger.error(
            f"[PlaceStoryStatus] Failed to send status for step_id={step_id}: {e}"
        )



def format_execution_time(seconds: float) -> str:
    """
    Format execution time intelligently:
    - < 60s: "X.XX seconds"
    - >= 60s: "X minutes Y seconds" (or just "X minutes" if Y is 0)
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if remaining_seconds == 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
        
    return f"{minutes} minute{'s' if minutes > 1 else ''} {remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"