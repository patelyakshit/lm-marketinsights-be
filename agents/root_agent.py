"""
Pure Google ADK Root Agent with Sub-Agent Coordination

Orchestrates specialized agents (Salesforce, GIS) through Google ADK's
sub_agents parameter while providing real-time streaming capabilities.
"""

import asyncio
import copy
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import ClassVar, Dict, List, Optional

from decouple import config
from google.adk.agents import LlmAgent, RunConfig, LiveRequestQueue
from google.adk.agents.run_config import StreamingMode
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.genai import types

from agents.gis_agent import GISAgent
from agents.placestory_agent import PlaceStoryAgent
from agents.salesforce_agent import SalesforceAgent
from agents.rag_agent import RAGAgent
from agents.marketing_agent import MarketingAgent
from knowledge import get_tapestry_service
from agents.streaming_callbacks import (
    after_tool_modifier,
    before_tool_modifier,
    after_model_modifier,
    before_model_modifier,
)
from constants.chat_constants import ProgressPhase
from config.config import (
    session_service,
    database_session_service,
    in_memory_session_service,
)
from utils.query_classifier import classify_query, QueryType
from utils.workflow_templates import should_use_workflow, execute_workflow
from dao import ChatInputOutputDAO
from handlers import UnifiedEventHandler
from managers.websocket_manager import manager
from utils.enums import UserTypeConstants

logger = logging.getLogger(__name__)


class RootAgent(LlmAgent):
    """
    Pure ADK Root Agent with sub-agent coordination and streaming capabilities.

    Uses Google ADK's sub_agents parameter to orchestrate specialized agents:
    - Salesforce Agent for CRM operations
    - GIS Agent for mapping and geocoding
    - Automatic multi-agent workflow coordination
    """

    def __init__(
        self,
        model: str | Gemini = config("ROOT_MODEL", "gemini-2.5-flash-lite"),
        allow_sub_agent_override: bool = True,
    ):
        """
        Initialize Root Agent with ADK architecture and sub-agent coordination.

        Args:
            model: LLM model identifier (default: "gemini-2.5-flash-lite")
            allow_sub_agent_override: If True, sub-agents can use SUB_AGENT_MODEL override.
                                     If False, sub-agents must use this model (e.g., for audio).
        """

        super().__init__(
            model=model,
            name="root_agent",
            description="Root orchestrator agent that coordinates specialized sub-agents for multi-agent workflows",
            instruction=self._get_system_instruction(),
            sub_agents=[
                SalesforceAgent(model, allow_override=allow_sub_agent_override),
                GISAgent(model, allow_override=allow_sub_agent_override),
                PlaceStoryAgent(model, allow_override=allow_sub_agent_override),
                RAGAgent(model, allow_override=allow_sub_agent_override),
                MarketingAgent(model, allow_override=allow_sub_agent_override),
            ],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
        )

        logger.info(
            f"RootAgent initialized with model: {model}, "
            f"sub-agent override: {'enabled' if allow_sub_agent_override else 'disabled'}"
        )

    def _get_system_instruction(self) -> str:
        """
        Get optimized system instruction for multi-agent coordination.
        Compressed for faster inference (~40% token reduction).
        """

        return """You orchestrate specialized agents for Salesforce CRM, GIS, RAG, PlaceStory, and Marketing tasks.

## Agents & Routing

**GIS**: Map ops (zoom/pan/layers/pins), geocoding, spatial queries, TRADE AREA analysis (drive time, radius), lifestyle/segment analysis for locations, DATA QUERIES about any geographic/market data (rent, demographics, population, housing, etc).
**Salesforce**: SOQL/SOSL, Salesforce objects (Account/Contact/Lead/Opportunity/__c), data exports, SF mapping.
**RAG**: ArcGIS Tapestry segmentation knowledge (segment codes like 1A, 2B, I3), LifeMode groups (A-L), segment characteristics.
**PlaceStory**: Location narratives, place reports, market analysis stories.
**Marketing**: Social media posts (FB/LinkedIn/IG), email campaigns, tapestry-targeted content.

## Quick Route
- Salesforce keywords (SOQL/Account/Contact/__c) → Salesforce
- Map navigation/layers/pins → GIS
- "drive time"/"drive-time"/"radius"/"nearby lifestyles"/"trade area"/"within X minutes" + address → GIS (use geocode then create_drive_time_polygon)
- Tapestry segment codes (1A, 2B, I3), LifeMode groups, "what is segment X" → RAG
- "placestory"/"create story"/"market analysis" → PlaceStory
- "marketing post"/"social media"/"create post" → Marketing
- **DATA QUESTIONS** (rent, demographics, housing, market data, population, income, etc.) → GIS (uses dynamic layer discovery)
- **STORE/LOCATION QUERIES** ("zoom to store X", "find store X", "go to store X") → GIS (queries Stores layer directly from map)
- General questions → Answer directly

## IMPORTANT: Data Questions
When users ask about ANY geographic or market data (rent prices, demographics, population, housing, commercial data, etc.):
1. Route to GIS immediately - it has DYNAMIC LAYER DISCOVERY
2. GIS will automatically find relevant data layers and query them
3. Do NOT say "I don't have that data" - GIS can discover it dynamically
Examples: "What's the median rent in Dallas?", "Show me population data for Texas", "Demographics for this area"

## Critical: Address + Drive Time Queries
When user asks about lifestyles/segments for an address with drive time:
1. Route to GIS immediately
2. GIS will: geocode address → create drive-time polygon → get tapestry intelligence
3. Do NOT ask for clarification if address and time are provided
Example: "nearby lifestyles for 1101 Coit Rd, Plano for 15 min" → GIS handles fully

## Critical: Store/Location Queries from Map Layers
When user asks about stores, locations, or POIs that are on the map (e.g., "zoom to store 18", "find store 5"):
1. Route to GIS immediately - do NOT use external knowledge bases
2. GIS will query the Stores/Locations layer DIRECTLY from map_context using execute_structured_query
3. GIS extracts coordinates from the query result geometry and zooms to location
4. Then GIS can query other layers (rent, demographics) at that location
Example: "zoom to store 18, tell me about rent" → GIS queries Stores layer → gets coordinates → zooms → queries rent data

## Rules
- Each agent handles its domain end-to-end (no splitting)
- ONLY ask clarification if truly missing info (address unclear, no time specified)
- Be concise

## Voice Mode
- Root Agent: sole TTS authority, ~20 words max
- Sub-agents: return data only, no TTS
"""

    def get_capabilities(self) -> List[str]:
        """Get list of root agent capabilities."""

        return [
            "Multi-agent workflow orchestration",
            "Salesforce CRM and GIS agent coordination",
            "Intelligent query analysis and routing",
            "Real-time streaming of agent execution",
            "Complex workflow pattern detection",
            "Cross-agent data transformation",
            "Session context management",
            "Fallback and error handling",
            "Thinking token transparency",
            "RAG agent for ArcGIS tapestry knowledge base 2025",
            "ADK sub-agent integration",
            "Placestory generation from context",
            "Conversation with the user to collect placestory context",
            "Marketing post generation with tapestry insights",
            "Platform-specific social media content creation",
        ]

    # Performance optimization: Speculative responses for common greetings
    QUICK_RESPONSES: ClassVar[Dict[str, str]] = {
        "hi": "Hello! I'm your GIS assistant. How can I help you explore the map today?",
        "hello": "Hi there! I'm ready to help you with geographic insights, data analysis, or map navigation. What would you like to know?",
        "hey": "Hey! How can I assist you with the map today?",
        "hi there": "Hello! What geographic data or location would you like to explore?",
        "hello there": "Hi! I'm here to help with maps, location data, and geographic analysis. What can I do for you?",
        "good morning": "Good morning! Ready to help you explore and analyze geographic data. What would you like to know?",
        "good afternoon": "Good afternoon! How can I assist you with the map today?",
        "good evening": "Good evening! What geographic insights can I help you with?",
        # Capability questions
        "what can you do": """I can help you with:
• **Map Navigation** - Zoom, pan, and navigate to locations
• **Add Pins** - Mark locations with notes
• **Layer Management** - Show/hide data layers (e.g., "turn on age median")
• **Lifestyle Analysis** - Get Tapestry segment insights for any address
• **Trade Area Analysis** - Create drive-time polygons and radius buffers
• **Demographics** - Population, income, and household data
• **Site Selection** - Find optimal locations for businesses
• **Data Discovery** - Ask about any market data, I'll find the right layers!

Try: "What's the population in Texas?" or "Show me housing data"

Just ask me about any location or analysis you need!""",
        "help": """I'm your GIS assistant! Here's what I can do:

**Quick Commands:**
• "Zoom to [city]" - Navigate to a location
• "Add pin on [address]" - Mark a location
• "Show lifestyle layer" - Toggle data layers

**Analysis:**
• "What are nearby lifestyles for [address]?" - Tapestry segment analysis
• "Create 15-min drive time for [address]" - Trade area polygon
• "Get demographics for [location]" - Population & income data

**Knowledge:**
• "What is D2 segment?" - Tapestry segment info
• "What is LifeMode A?" - LifeMode group details

What would you like to explore?""",
        "what are your capabilities": """I specialize in GIS and location intelligence:

**1. Map Operations**
- Navigate to any location
- Add/remove pins with notes
- Control layer visibility (just say "turn on population density")

**2. Location Analysis**
- Tapestry lifestyle segmentation
- Demographics (population, income, age)
- Points of interest nearby

**3. Trade Area Analysis**
- Drive-time polygons (5, 10, 15, 30 min)
- Radius buffers (1, 3, 5, 10 miles)
- Competitor analysis

**4. Business Intelligence**
- Site selection recommendations
- Market analysis
- Customer profiling

**5. Dynamic Data Discovery** (NEW!)
- Ask about ANY data: "What's the median rent in Dallas?"
- I'll automatically find relevant data layers from 97+ sources
- Population, income, housing, demographics, and more

Ask me anything about a location or market data!""",
    }

    # Note: Tapestry segment data is now provided by TapestryKnowledgeService
    # See knowledge/tapestry_service.py for the comprehensive 2025 segment database

    def _get_segment_quick_response(self, query: str) -> Optional[str]:
        """
        Check if query is asking about a Tapestry segment or LifeMode group.
        Returns instant response without LLM call if matched.

        Uses TapestryKnowledgeService for comprehensive 2025 segment data.

        Patterns matched:
        - "what is A1 segment" / "what is segment A1"
        - "tell me about D2" / "D2 segment"
        - "what is lifemode A" / "what is LifeMode B"
        - "what is tapestry"
        """
        import re

        query = query.lower().strip()

        # Get the Tapestry knowledge service (singleton, instant)
        tapestry = get_tapestry_service()

        # Pattern 1: Segment questions - "what is [code] segment" etc.
        segment_patterns = [
            r"what is (\w+) segment",
            r"what is segment (\w+)",
            r"tell me about (\w+) segment",
            r"tell me about segment (\w+)",
            r"describe (\w+) segment",
            r"explain (\w+) segment",
            r"^(\w{1,2}\d?) segment$",  # Just "A1 segment" or "D2 segment"
            r"what's (\w+) segment",
            r"whats (\w+) segment",
            r"^(\w\d) segment",  # "A1 segment details"
            r"segment (\w\d)$",  # "segment A1"
        ]

        for pattern in segment_patterns:
            match = re.search(pattern, query)
            if match:
                segment_code = match.group(1).upper()
                # Use the knowledge service for response
                return tapestry.get_segment_response(segment_code)

        # Pattern 2: LifeMode questions - "what is lifemode A"
        lifemode_patterns = [
            r"what is lifemode (\w)",
            r"what is life mode (\w)",
            r"tell me about lifemode (\w)",
            r"what's lifemode (\w)",
            r"lifemode (\w)$",
            r"^lifemode (\w)$",
        ]

        for pattern in lifemode_patterns:
            match = re.search(pattern, query)
            if match:
                lifemode_letter = match.group(1).upper()
                # Use the knowledge service for response
                return tapestry.get_lifemode_response(lifemode_letter)

        # Pattern 3: General tapestry overview questions
        if query in ["what is lifestyle", "what is tapestry", "what are tapestry segments",
                     "tapestry overview", "tapestry 2025", "what is tapestry segmentation"]:
            return tapestry.get_all_segments_summary()

        return None

    async def process_query(
        self,
        query: str,
        session_id: str = "default",
        connection_id: str = "",
        extra_information: dict = {},
        runner: Runner = None,
    ) -> str:
        """
        Process query using in-memory session for speed, then persist to database.

        Args:
            query: User query text
            session_id: Unique session identifier
            connection_id: WebSocket connection identifier
            extra_information: Additional context (e.g., map_context)
            runner: In-memory Runner instance for fast processing
        """
        try:
            logger.info(f"Processing query: {query[:100] if query else 'None'}...")

            # Performance optimization: Instant response for simple queries
            normalized_query = query.strip().lower()

            # Check for exact match quick responses (greetings, help, capabilities)
            if normalized_query in self.QUICK_RESPONSES:
                quick_response = self.QUICK_RESPONSES[normalized_query]
                logger.info(f"Quick response for: {normalized_query}")
                stream_id = str(uuid.uuid4())
                await manager.send_streaming_message(
                    connection_id,
                    {
                        "stream_response": quick_response,
                        "stream_stop": True,
                        "stream_id": stream_id,
                    },
                )
                return quick_response

            # Check for Tapestry segment questions (e.g., "what is D2 segment", "what is segment d2")
            segment_response = self._get_segment_quick_response(normalized_query)
            if segment_response:
                logger.info(f"Quick response for segment query: {normalized_query}")
                stream_id = str(uuid.uuid4())
                await manager.send_streaming_message(
                    connection_id,
                    {
                        "stream_response": segment_response,
                        "stream_stop": True,
                        "stream_id": stream_id,
                    },
                )
                return segment_response

            # Performance optimization: Static workflow templates for common operations
            # Bypasses full AI processing for simple commands like "zoom to Dallas"
            if should_use_workflow(query):
                logger.info(f"Using workflow template for: {query[:50]}...")
                from tools.gis_tools import gis_executor

                workflow_context = {
                    "gis_executor": gis_executor,
                    "map_context": extra_information.get("map_context", {}),
                }

                workflow_result = await execute_workflow(query, workflow_context)

                if workflow_result and workflow_result.success:
                    stream_id = str(uuid.uuid4())
                    # Send operations to frontend
                    if workflow_result.operations:
                        await manager.send_message(
                            connection_id,
                            {
                                "type": "CHAT/OPERATION_DATA",
                                "payload": {"operations": workflow_result.operations},
                            },
                        )
                    # Send text response
                    await manager.send_streaming_message(
                        connection_id,
                        {
                            "stream_response": workflow_result.message,
                            "stream_stop": True,
                            "stream_id": stream_id,
                        },
                    )
                    return workflow_result.message

                # Handle special case: no layers in context
                if workflow_result and workflow_result.message == "NO_LAYERS_CONTEXT":
                    stream_id = str(uuid.uuid4())
                    error_msg = "I couldn't access the map layers. Please try again in a moment, or manually toggle the layer using the Layers panel on the right."
                    await manager.send_streaming_message(
                        connection_id,
                        {
                            "stream_response": error_msg,
                            "stream_stop": True,
                            "stream_id": stream_id,
                        },
                    )
                    return error_msg

                # If workflow failed, fall through to normal processing
                logger.info(f"Workflow template failed, falling back to AI: {workflow_result.message if workflow_result else 'None'}")

            # Use in-memory Runner (ADK best practice - initialized once at startup)
            if runner is None:
                raise ValueError("Runner must be provided (initialized at startup)")

            # Send "understanding" progress phase (immediate feedback to user)
            await manager.send_progress_message(
                connection_id, ProgressPhase.UNDERSTANDING
            )

            # Query-Adaptive Response Classification
            # Classify query type and get adaptive response instruction
            classification = classify_query(query)
            logger.info(
                f"Query classified as {classification.query_type.value} "
                f"(confidence: {classification.confidence:.2f}, style: {classification.suggested_response_style})"
            )

            # For high-confidence classifications, add response style hint to query
            # This helps the AI provide appropriately formatted responses
            adaptive_query = query
            if classification.confidence >= 0.7 and classification.query_type not in [
                QueryType.GREETING,  # Greetings handled by QUICK_RESPONSES
                QueryType.GENERAL,   # Don't modify general queries
            ]:
                style_hints = {
                    "brief": "[Respond briefly in 1-2 sentences]",
                    "action": "[Confirm action and show result concisely]",
                    "structured": "[Use tables/lists for data presentation]",
                    "detailed": "",  # No modification for detailed
                }
                hint = style_hints.get(classification.suggested_response_style, "")
                if hint:
                    # Append hint as system context (not visible to user)
                    adaptive_query = f"{query}\n\n{hint}"
                    logger.debug(f"Added response style hint: {hint}")

            # Step 1: Load existing session from database (if exists)
            db_session = await database_session_service.get_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=session_id,
            )

            # Step 2: Get or create session in in-memory service
            # First check if session already exists in memory (handles rapid consecutive queries)
            memory_session = await in_memory_session_service.get_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=session_id,
            )

            if memory_session:
                logger.info(f"Reusing existing in-memory session {session_id}")
            elif db_session:
                # Recreate session in memory from database - use shallow copy for performance
                memory_session = await in_memory_session_service.create_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=session_id,
                    state=db_session.state.copy() if db_session.state else {},
                )
                # Copy only last 20 events from DB session to in-memory session (performance optimization)
                recent_events = db_session.events[-20:] if len(db_session.events) > 20 else db_session.events
                for event in recent_events:
                    await in_memory_session_service.append_event(memory_session, event)

                logger.info(
                    f"Loaded session {session_id} from DB: {len(db_session.events)} events"
                )
            else:
                # Create new session in memory
                memory_session = await in_memory_session_service.create_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=session_id,
                )
                logger.info(f"Created new session {session_id} in memory")

            # Step 3: Add system event to in-memory session
            state_delta = {
                "connection_id": connection_id,
                "session_id": session_id,
            }
            if extra_information.get("map_context"):
                state_delta["map_context"] = extra_information.get("map_context")

            # Store user_info in session state if available (for organization-based features)
            if extra_information.get("user_info"):
                state_delta["user_info"] = extra_information.get("user_info")

            event_id = str(uuid.uuid4())
            event = Event(
                id=event_id,
                invocation_id=event_id,
                author="system",
                actions=EventActions(state_delta=state_delta),
                timestamp=datetime.now().timestamp(),
            )

            # Append event and create message in parallel
            await asyncio.gather(
                in_memory_session_service.append_event(memory_session, event),
                ChatInputOutputDAO().create_message(
                    session_id=uuid.UUID(session_id),
                    user_type=UserTypeConstants.HUMAN,
                    message=query,
                    stream_id=uuid.UUID(connection_id),
                    stop_stream=True,
                    events_data={},
                    assets_data={},
                ),
            )

            # Step 4: Process with in-memory runner (FAST!)
            # Use adaptive_query which may include response style hints
            content = types.Content(role="user", parts=[types.Part(text=adaptive_query)])
            events = runner.run_async(
                user_id=f"user_{session_id}",
                session_id=memory_session.id,
                new_message=content,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )

            event_handler = UnifiedEventHandler(
                connection_id=connection_id,
                session_id=session_id,
                manager=manager,
                mode="SSE",
            )

            async for event in events:
                await event_handler.handle_event(event)

            # Step 5: Persist final session state to database (non-blocking for faster response)
            # Using create_task() so response completes immediately while persistence happens in background
            asyncio.create_task(
                self._persist_session_to_database(
                    session_id=session_id, db_session=db_session
                )
            )

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error in process_query for session {session_id}: {error_type} - {e}",
                extra={
                    "session_id": session_id,
                    "connection_id": connection_id,
                    "error_type": error_type,
                    "traceback": traceback.format_exc(),
                },
            )

            # Send structured error to websocket for retry
            await manager.send_message(
                connection_id,
                {
                    "type": "CHAT/ERROR",
                    "payload": {
                        "error_type": "AGENT_PROCESSING_ERROR",
                        "message": str(e),
                        "retriable": True,
                        "timestamp": time.time(),
                    },
                },
            )

            raise

    async def _persist_session_to_database(self, session_id: str, db_session=None):
        """
        Persist in-memory session to database after response generation completes.

        Args:
            session_id: Session identifier
            db_session: Original DB session (if it existed)
        """
        try:
            # Get final in-memory session state
            final_memory_session = await in_memory_session_service.get_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=session_id,
            )

            if not final_memory_session:
                logger.warning(
                    f"No in-memory session found for {session_id} to persist"
                )
                return

            # Log state keys for debugging
            state_keys = (
                list(final_memory_session.state.keys())
                if final_memory_session.state
                else []
            )
            logger.info(
                f"Persisting session {session_id} with state keys: {state_keys}"
            )

            # Log marketing_context specifically if it exists
            if (
                final_memory_session.state
                and "marketing_context" in final_memory_session.state
            ):
                marketing_context = final_memory_session.state.get(
                    "marketing_context", {}
                )
                logger.info(
                    f"Marketing context found in session state with keys: {list(marketing_context.keys()) if isinstance(marketing_context, dict) else 'N/A'}"
                )
            else:
                logger.warning(
                    f"Marketing context NOT found in session state for {session_id}. "
                    f"Available keys: {state_keys}"
                )

            # Determine which events are new (not yet in DB)
            existing_event_ids = set()
            if db_session and db_session.events:
                existing_event_ids = {e.id for e in db_session.events}

            new_events = [
                e for e in final_memory_session.events if e.id not in existing_event_ids
            ]

            # Create or update session in database
            if db_session is None:
                # Create new session in DB - use shallow copy for performance
                persisted_session = await database_session_service.create_session(
                    app_name=final_memory_session.app_name,
                    user_id=final_memory_session.user_id,
                    session_id=final_memory_session.id,
                    state=(
                        final_memory_session.state.copy()
                        if final_memory_session.state
                        else {}
                    ),
                )
            else:
                # Get existing session from DB
                persisted_session = await database_session_service.get_session(
                    app_name=final_memory_session.app_name,
                    user_id=final_memory_session.user_id,
                    session_id=final_memory_session.id,
                )
                # Update state - need to actually persist it to database
                if persisted_session:
                    # Use shallow copy for performance (nested structures rarely modified concurrently)
                    persisted_session.state = (
                        final_memory_session.state.copy()
                        if final_memory_session.state
                        else {}
                    )
                    # Actually persist the state change to database
                    persisted_session = await database_session_service.update_session(
                        persisted_session
                    )

            # Append only new events to database session
            if persisted_session:
                for event in new_events:
                    await database_session_service.append_event(
                        persisted_session, event
                    )

                # Verify persisted state after update
                if (
                    persisted_session.state
                    and "marketing_context" in persisted_session.state
                ):
                    marketing_context = persisted_session.state.get(
                        "marketing_context", {}
                    )
                    logger.info(
                        f"Successfully persisted marketing_context for session {session_id} "
                        f"with keys: {list(marketing_context.keys()) if isinstance(marketing_context, dict) else 'N/A'}"
                    )
                else:
                    logger.warning(
                        f"Marketing context missing after persistence for session {session_id}. "
                        f"Persisted state keys: {list(persisted_session.state.keys()) if persisted_session.state else []}"
                    )

                logger.info(
                    f"Persisted session {session_id} to DB: "
                    f"{len(new_events)} new events, "
                    f"state keys: {list(final_memory_session.state.keys()) if final_memory_session.state else []}, "
                    f"{len(final_memory_session.events)} total events"
                )

            # database_session_service.delete_session(

            # Clean up in-memory session to free memory
            await in_memory_session_service.delete_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=session_id,
            )
            logger.debug(f"Cleaned up in-memory session {session_id}")

        except Exception as e:
            # Log error but don't raise - per user's requirement (option a: log but don't fail)
            logger.error(
                f"Failed to persist session {session_id} to database: {e}",
                extra={
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

    async def start_audio_streaming(
        self,
        session_id: str,
        connection_id: str,
        extra_information: dict = None,
        runner: Runner = None,
    ):
        """
        Initialize ADK audio streaming session with shared Runner (Google ADK best practice).

        Args:
            session_id: Unique session identifier
            connection_id: WebSocket connection identifier
            extra_information: Additional context (user_info, map_context, etc.)
            runner: Shared Runner instance (ADK best practice - reuse across requests)

        Returns:
            tuple: (live_events, live_request_queue)
                - live_events: AsyncGenerator of ADK events (audio, text, turn signals)
                - live_request_queue: Queue for sending audio/text to ADK agent
        """
        try:
            if extra_information is None:
                extra_information = {}

            logger.info(f"Initializing audio streaming for session {session_id}")

            # ADK best practice: Get session first, then create if not exists
            session = await session_service.get_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=session_id,
            )
            if session is None:
                session = await session_service.create_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=session_id,
                )

            # Build state_delta once (ADK pattern - let Event handle state updates)
            state_delta = {
                "connection_id": str(connection_id),
                "session_id": str(session_id),
                "map_context": extra_information.get("map_context"),
            }

            # Reuse single UUID for system events
            event_id = str(uuid.uuid4())
            event = Event(
                id=event_id,
                invocation_id=event_id,
                author="system",
                actions=EventActions(state_delta=state_delta),
                timestamp=datetime.now().timestamp(),
            )
            await session_service.append_event(session, event)

            # Use shared Runner (ADK best practice - initialized once at startup)
            if runner is None:
                raise ValueError("Runner must be provided (initialized at startup)")

            # Configure for BIDI audio streaming with session resumption
            run_config = RunConfig(
                session_resumption=types.SessionResumptionConfig(transparent=True),
                save_input_blobs_as_artifacts=True,
                streaming_mode=StreamingMode.BIDI,
                max_llm_calls=1000,
                output_audio_transcription=types.AudioTranscriptionConfig(),
                speech_config=types.SpeechConfig(
                    language_code="en-US",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=config("SPEAKER_VOICE_NAME", "Kore")
                        )
                    ),
                ),
            )

            # Create LiveRequestQueue for bidirectional communication
            live_request_queue = LiveRequestQueue()

            # Start live streaming with audio support
            live_events = runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
            )

            logger.info(
                f"Audio streaming initialized for connection {connection_id}, session {session_id}"
            )

            return (live_events, live_request_queue)

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error initializing audio streaming for session {session_id}: {error_type} - {e}",
                extra={
                    "session_id": session_id,
                    "connection_id": connection_id,
                    "error_type": error_type,
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    async def _cleanup_session_memory(
        self, session_service, session, keep_last_turns: int = 3, max_events: int = 50
    ):
        """
        Aggressive session memory cleanup strategy:
        1. Keep only last N turns (turn-based)
        2. Enforce maximum event count (size-based)
        3. Audio data already stripped by event loop

        Args:
            session_service: ADK session service
            session: Current session
            keep_last_turns: Number of recent turns to preserve (default: 3)
            max_events: Maximum total events to keep (default: 50)

        Returns:
            Number of events after cleanup
        """
        try:
            all_events = session.events
            initial_count = len(all_events)

            # Strategy 1: Turn-based cleanup - keep last N conversation turns
            turn_indices = [
                i
                for i, evt in enumerate(all_events)
                if hasattr(evt, "turn_complete") and evt.turn_complete
            ]

            if len(turn_indices) > keep_last_turns:
                # Keep events from last N turns
                cutoff_index = turn_indices[-(keep_last_turns + 1)]
                events_to_keep = all_events[cutoff_index:]
            else:
                events_to_keep = all_events

            # Strategy 2: Size-based enforcement (hard limit)
            if len(events_to_keep) > max_events:
                events_to_keep = events_to_keep[-max_events:]

            # Update session if pruning occurred
            if len(events_to_keep) < initial_count:
                session.events = events_to_keep
                await session_service.update_session(session)

                logger.info(
                    f"Session cleanup: {initial_count} → {len(events_to_keep)} events "
                    f"(kept last {keep_last_turns} turns, max {max_events} events)"
                )

            return len(events_to_keep)

        except Exception as e:
            logger.warning(f"Session cleanup failed: {e}")
            return len(session.events)
