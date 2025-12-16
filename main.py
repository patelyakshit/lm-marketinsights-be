"""
Pure Google ADK FastAPI Integration with Runner and Session Management

Provides WebSocket-based real-time agent execution with streaming callbacks
and proper ADK session management using Tortoise ORM.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

import jsonpatch
import litellm
from decouple import config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google.adk.events import EventActions, Event
from google.adk.runners import Runner

from agents.root_agent import RootAgent
from config.config import (
    session_service,
    database_session_service,
    in_memory_session_service,
)
from config.logging_config import setup_logging
from constants.chat_constants import (
    WEBSOCKET_TYPE_SESSION,
    WEBSOCKET_TYPE_ERROR,
    WEBSOCKET_TYPE_PONG,
    WEBSOCKET_TYPE_MAP_CONTEXT,
    WEBSOCKET_TYPE_SEND,
    WEBSOCKET_TYPE_PATCH,
    WEBSOCKET_TYPE_PING,
)
from dao.chat_history_queue_dao import ChatHistoryDAO
from dao.chat_input_output_queue_dao import ChatInputOutputDAO
from handlers import UnifiedEventHandler
from managers.websocket_manager import manager
from message_queue import init_queue_manager, close_queue_manager
from tools.layer_query_tools import set_ws_context, handle_layer_query_response
from utils.query_deduplication import query_deduplicator
from utils.context_preloader import preload_session_context, invalidate_session_context
from routers.layer_intelligence import router as layer_intelligence_router

# Initialize centralized logging configuration
setup_logging()
logger = logging.getLogger(__name__)

if config("DEBUG", False, cast=bool):
    litellm._turn_on_debug()

# Initialize FastAPI app
app = FastAPI(
    title="Google ADK Multi-Agent Service with Streaming",
    description="Real-time multi-agent system with Salesforce and GIS capabilities",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Authentication middleware
from auth.auth_middleware import AuthenticationMiddleware

# Configure authentication middleware with excluded paths
app.add_middleware(
    AuthenticationMiddleware,
    excluded_paths={
        "/",  # Root endpoint
        "/health",  # Health check endpoint for Railway
        "/api/v1/layer-intelligence/health",  # Health check endpoint
        "/api/v1/layer-intelligence/sync",  # Layer sync endpoint (for testing)
        "/api/v1/layer-intelligence/search",  # Layer search endpoint
        "/api/v1/layer-intelligence/query",  # Natural language query endpoint
        "/api/v1/layer-intelligence/layers",  # List layers endpoint
        "/api/v1/layer-intelligence/stats",  # Stats endpoint
    },
    optional_auth_paths=set(),  # No optional auth paths by default
)

# Include Layer Intelligence REST API router
app.include_router(layer_intelligence_router)


# Health check endpoint for Railway/container orchestration
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "lm-multi-agent-api"}

# Singleton agents - initialized in startup event after credentials are ready
TEXT_ROOT_AGENT = None
AUDIO_ROOT_AGENT = None

# Singleton Runners - initialized in startup event (ADK best practice)
TEXT_RUNNER = None
AUDIO_RUNNER = None
IN_MEMORY_TEXT_RUNNER = None


# Queue startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global TEXT_RUNNER, AUDIO_RUNNER, IN_MEMORY_TEXT_RUNNER, TEXT_ROOT_AGENT, AUDIO_ROOT_AGENT
    try:
        # Step 1: Check and download GCP credentials if needed
        from utils.gcp_credentials import (
            check_or_download_credentials_file,
            get_credentials_path,
        )

        logger.info("Step 1/7: Checking GCP credentials...")
        credentials_ready = await check_or_download_credentials_file()
        if not credentials_ready:
            logger.error("GCP credentials not available - cannot start service")
            raise RuntimeError("Failed to download/locate GCP credentials")

        credentials_path = str(get_credentials_path())
        logger.info(f"GCP credentials ready at: {credentials_path}")

        # Set GOOGLE_APPLICATION_CREDENTIALS to point to downloaded file
        # This ensures Google auth libraries can find the credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS={credentials_path}")

        # Step 1.5: Initialize singleton GenAI client with shared credentials
        logger.info(
            "Step 2/7: Initializing singleton GenAI client with shared credentials..."
        )
        from utils.genai_client_manager import initialize_genai_client

        initialize_genai_client(
            credentials_path=credentials_path,
            project=config("GOOGLE_CLOUD_PROJECT", default="lm-market-insights-ai"),
            location=config("GOOGLE_CLOUD_LOCATION", default="us-central1"),
        )
        logger.info("✓ Singleton GenAI client initialized with cached credentials")

        # Step 2: Initialize shared GCP client manager with downloaded credentials
        logger.info("Step 3/7: Initializing GCP client manager...")
        logger.info(
            "GCP client manager initialization with shared credentials.....Skipped"
        )

        # Step 3: Initialize database tables
        logger.info("Step 4/7: Initializing database tables...")
        await database_session_service.initialize_tables()
        logger.info("Database tables initialized")

        # Step 4: Create singleton agents with shared client (eliminates multiple token API calls)
        logger.info("Step 5/7: Creating singleton agents with shared client...")
        from utils.genai_client_manager import create_gemini_with_shared_client

        # Create Gemini instances that use the shared client
        text_model = create_gemini_with_shared_client(
            config("ROOT_MODEL", "gemini-2.5-flash-lite")
        )
        audio_model = create_gemini_with_shared_client(
            "gemini-live-2.5-flash-preview-native-audio"
        )

        TEXT_ROOT_AGENT = RootAgent(model=text_model)
        AUDIO_ROOT_AGENT = RootAgent(
            model=audio_model,
            allow_sub_agent_override=False,
        )
        logger.info(
            "✓ Singleton agents created with shared client: TEXT_ROOT_AGENT, AUDIO_ROOT_AGENT"
        )

        # Step 5: Initialize RabbitMQ queue manager
        logger.info("Step 6/7: Initializing message queue manager...")
        init_queue_manager()
        logger.info("Message queue manager initialized")

        # Step 6: Initialize singleton Runners (ADK best practice - reuse across requests)
        logger.info("Step 7/7: Initializing singleton Runners...")
        TEXT_RUNNER = Runner(
            agent=TEXT_ROOT_AGENT,
            app_name="root_agent_app",
            session_service=session_service,
        )
        AUDIO_RUNNER = Runner(
            agent=AUDIO_ROOT_AGENT,
            app_name="root_agent_app",
            session_service=session_service,
        )
        IN_MEMORY_TEXT_RUNNER = Runner(
            agent=TEXT_ROOT_AGENT,
            app_name="root_agent_app",
            session_service=in_memory_session_service,
        )
        logger.info(
            "Singleton Runners initialized: TEXT_RUNNER, AUDIO_RUNNER, IN_MEMORY_TEXT_RUNNER"
        )

        # Step 8: Initialize Layer Intelligence System (optional - graceful degradation)
        logger.info("Step 8/8: Initializing Layer Intelligence System...")
        try:
            from services.layer_intelligence import initialize_layer_intelligence
            li_status = await initialize_layer_intelligence(skip_db=True)  # Skip DB for now, uses Qdrant
            if li_status.get("overall") == "ok":
                logger.info("Layer Intelligence System initialized successfully")
            else:
                logger.warning(f"Layer Intelligence partially initialized: {li_status}")
        except ImportError as e:
            logger.warning(f"Layer Intelligence not available (missing dependencies): {e}")
        except Exception as e:
            logger.warning(f"Layer Intelligence initialization failed (non-critical): {e}")

        logger.info("=" * 80)
        logger.info("✓ Startup complete: All services initialized successfully")
        logger.info("=" * 80)
    except Exception as e:
        import traceback

        logger.error("=" * 80)
        logger.error(f"✗ Failed to initialize services: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close RabbitMQ queue connections and database pool on application shutdown."""
    try:
        # Close RabbitMQ connections
        close_queue_manager()
        logger.info("Queue connections closed successfully")

        # Close custom session storage service
        await database_session_service.close()
        logger.info("Session storage service closed successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


logger.info("Google ADK service initialized - using singleton agent pattern")

ChatHistory = ChatHistoryDAO()
ChatInputOutput = ChatInputOutputDAO()


@app.get("/")
async def root():
    """Root endpoint with ADK service information."""
    # Check if agents are initialized
    if TEXT_ROOT_AGENT is None:
        return {
            "message": "Google ADK Multi-Agent Service with Real-time Streaming",
            "status": "initializing",
            "version": "2.0.0",
            "note": "Service is starting up. Please wait for initialization to complete.",
        }

    # Use singleton agent for info (no need to create new instance)
    sample_agent = TEXT_ROOT_AGENT
    return {
        "message": "Google ADK Multi-Agent Service with Real-time Streaming",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "/ws": "SSE text streaming (ROOT_MODEL: gemini-2.5-flash-lite)",
            "/ws/audio": "BIDI audio streaming (AUDIO_MODEL: gemini-live-2.5-flash-preview-native-audio-09-2025)",
        },
        "agents": {
            "root_agent": sample_agent.name,
            "sub_agents": (
                [agent.name for agent in sample_agent.sub_agents]
                if hasattr(sample_agent, "sub_agents") and sample_agent.sub_agents
                else []
            ),
        },
    }


async def agent_to_client_messaging(
    websocket: WebSocket,
    connection_id: str,
    session_id: str,
    live_events,
):
    """
    Stream ADK agent events to client (Google's recommended pattern).

    This function processes live_events from the ADK runner and forwards
    audio, text, turn status, and thinking events to the WebSocket client.

    Args:
        websocket: FastAPI WebSocket connection
        connection_id: Unique connection identifier
        session_id: Session identifier
        live_events: AsyncGenerator of ADK events from runner.run_live()
    """
    import time
    from constants.chat_constants import (
        WEBSOCKET_TYPE_STREAM_READY,
    )

    try:
        # Send STREAM_READY signal on startup
        await manager.send_message(
            connection_id,
            {
                "type": WEBSOCKET_TYPE_STREAM_READY,
                "payload": {
                    "connection_id": connection_id,
                    "session_id": session_id,
                    "message": "Audio stream ready - you can start recording",
                    "timestamp": time.time(),
                },
            },
        )
        logger.info(f"Sent STREAM_READY signal to client {connection_id}")

        event_handler = UnifiedEventHandler(
            connection_id=connection_id,
            session_id=session_id,
            manager=manager,
            mode="BIDI",
        )

        async for event in live_events:
            await event_handler.handle_event(event)

        # Send completion message
        await event_handler.send_completion_message()

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in agent_to_client_messaging for {connection_id}: {e}")
        # Reconnect the Runner
        raise


async def client_to_agent_messaging(
    websocket: WebSocket,
    connection_id: str,
    live_request_queue,
):
    """
    Forward client audio to ADK agent (Google's recommended pattern).

    This function receives messages from the WebSocket client and forwards
    audio data to the ADK agent's LiveRequestQueue for processing.

    Args:
        websocket: FastAPI WebSocket connection
        connection_id: Unique connection identifier
        live_request_queue: ADK LiveRequestQueue for sending audio/text to agent
    """
    from constants.chat_constants import WEBSOCKET_TYPE_PONG
    from utils.audio_utils import decode_audio_base64

    try:
        # Loop to receive messages from client
        while True:
            data = await websocket.receive_text()

            try:
                session_id = websocket.query_params.get("session_id")
                message = json.loads(data)
                message_type = message.get("type", "")
                payload = message.get("payload", {})

                # logger.info(
                #     f"Received message from client {connection_id}: type={message_type}"
                # )

                # Handle different message types
                if message_type == "CHAT/SEND_AUDIO":
                    # Extract audio data and turn completion flag
                    audio_base64 = payload.get("audio_base64", "")
                    turn_complete = payload.get("turn_complete", False)

                    if audio_base64:
                        # Decode audio from Base64
                        audio_bytes = decode_audio_base64(audio_base64)

                        if audio_bytes:
                            # Send to Google ADK live queue (BIDI - keep open)
                            from google.genai import types

                            live_request_queue.send_realtime(
                                types.Blob(
                                    data=audio_bytes, mime_type="audio/pcm;rate=16000"
                                )
                            )
                            if turn_complete:
                                # live_request_queue.send_activity_end()
                                logger.info(
                                    f"Sent activity_end signal for {connection_id} (user done speaking)"
                                )

                            # Send acknowledgment
                            await manager.send_message(
                                connection_id,
                                {
                                    "type": "audio_chunk_received",
                                    "connection_id": connection_id,
                                    "turn_complete_signaled": turn_complete,
                                    "timestamp": asyncio.get_event_loop().time(),
                                },
                            )
                        else:
                            logger.warning(
                                f"Failed to decode audio data from {connection_id}"
                            )

                elif message_type == "CHAT/INTERRUPT":
                    # User triggered interrupt
                    # Note: Model detects interrupts automatically from audio stream
                    # This is just for logging/debugging
                    logger.info(
                        f"User interrupt signal received from {connection_id} (model detects automatically)"
                    )

                elif message_type == "CHAT/PING":
                    # Handle ping for connection keep-alive
                    await manager.send_message(
                        connection_id,
                        {
                            "type": WEBSOCKET_TYPE_PONG,
                            "connection_id": connection_id,
                        },
                    )

                elif message_type == WEBSOCKET_TYPE_MAP_CONTEXT:
                    try:
                        session = await session_service.create_session(
                            app_name="root_agent_app",
                            user_id=f"user_{session_id}",
                            session_id=str(session_id),
                        )
                    except Exception as e:
                        session = await session_service.get_session(
                            app_name="root_agent_app",
                            user_id=f"user_{session_id}",
                            session_id=str(session_id),
                        )
                    session.state["session_id"] = str(session_id)
                    session.state["connection_id"] = str(connection_id)
                    if payload.get("map_context"):
                        session.state["map_context"] = payload.get("map_context")

                    event = Event(
                        id=str(uuid.uuid4()),
                        invocation_id=str(uuid.uuid4()),
                        author="system",
                        actions=EventActions(
                            state_delta={
                                "connection_id": str(connection_id),
                                "session_id": str(session_id),
                                "map_context": payload.get("map_context"),
                            }
                        ),
                        timestamp=datetime.now().timestamp(),
                    )
                    await session_service.append_event(session, event)

                else:
                    logger.warning(
                        f"Unknown message type from {connection_id}: {message_type}"
                    )

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error from {connection_id}: {e}")
                await manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "connection_id": connection_id,
                        "message": "Invalid JSON format",
                    },
                )
            except Exception as e:
                logger.error(f"Error processing message from {connection_id}: {e}")
                await manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "connection_id": connection_id,
                        "message": f"Message processing error: {str(e)}",
                    },
                )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Error in client_to_agent_messaging for {connection_id}: {e}")
        raise


async def handle_patch_immediately(connection_id: str, session_id: str, message: dict):
    """Handle PATCH without blocking main agent processing loop."""
    try:
        patched_data = message.get("payload", {}).get("patched_data", [])
        if not patched_data:
            logger.warning(
                f"No patched_data found in PATCH message for {connection_id}"
            )
            return

        session = await session_service.get_session(
            app_name="root_agent_app",
            user_id=f"user_{session_id}",
            session_id=str(session_id),
        )

        if session is None:
            session = await session_service.create_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=str(session_id),
            )

        current_map_context = session.state.get("map_context", {})
        patched_map_context = jsonpatch.apply_patch(current_map_context, patched_data)

        session.state["map_context"] = patched_map_context
        event = Event(
            id=str(uuid.uuid4()),
            invocation_id=str(uuid.uuid4()),
            author="system",
            actions=EventActions(state_delta={"map_context": patched_map_context}),
            timestamp=datetime.now().timestamp(),
        )
        await session_service.append_event(session, event)
        logger.debug(
            f"Applied {len(patched_data)} patch operations to map_context for {connection_id}"
        )

    except jsonpatch.JsonPatchException as e:
        logger.error(f"JSON Patch error for {connection_id}: {str(e)[:100]}")
    except Exception as e:
        logger.error(f"Error handling PATCH for {connection_id}: {str(e)[:100]}")


async def handle_map_context_immediately(
    connection_id: str, session_id: str, message: dict
):
    """Handle MAP_CONTEXT without blocking main agent processing loop."""
    try:
        payload = message.get("payload", {})

        try:
            session = await session_service.create_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=str(session_id),
            )
        except:
            session = await session_service.get_session(
                app_name="root_agent_app",
                user_id=f"user_{session_id}",
                session_id=str(session_id),
            )

        session.state["session_id"] = str(session_id)
        session.state["connection_id"] = str(connection_id)
        if payload.get("map_context"):
            session.state["map_context"] = payload.get("map_context")

        event = Event(
            id=str(uuid.uuid4()),
            invocation_id=str(uuid.uuid4()),
            author="system",
            actions=EventActions(
                state_delta={
                    "connection_id": str(connection_id),
                    "session_id": str(session_id),
                    "map_context": payload.get("map_context"),
                }
            ),
            timestamp=datetime.now().timestamp(),
        )
        await session_service.append_event(session, event)
        logger.debug(f"Updated map_context for {connection_id}")

    except Exception as e:
        logger.error(f"Error handling MAP_CONTEXT for {connection_id}: {e}")


async def background_message_queue_processor(
    message_queue: asyncio.Queue,
    connection_id: str,
    session_id: str,
    stop_event: asyncio.Event,
):
    """
    Process PATCH and PING/PONG messages from queue in background.
    Does not block main agent processing loop.

    This allows map_context updates and keepalive pings to be processed
    immediately even when the agent is processing a query (which can take 3-7s).
    """
    try:
        while not stop_event.is_set():
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(message_queue.get(), timeout=0.1)

                message_type = message.get("type", "")

                # Handle immediately - no blocking
                if message_type == WEBSOCKET_TYPE_PATCH:
                    await handle_patch_immediately(connection_id, session_id, message)
                    logger.debug(f"Processed PATCH in background for {connection_id}")

                elif message_type == WEBSOCKET_TYPE_PING:
                    await manager.send_message(
                        connection_id,
                        {
                            "type": WEBSOCKET_TYPE_PONG,
                            "connection_id": connection_id,
                            "session_id": str(session_id),
                        },
                    )
                    logger.debug(f"Processed PING in background for {connection_id}")

                elif message_type == WEBSOCKET_TYPE_MAP_CONTEXT:
                    await handle_map_context_immediately(
                        connection_id, session_id, message
                    )
                    logger.debug(
                        f"Processed MAP_CONTEXT in background for {connection_id}"
                    )

                message_queue.task_done()

            except asyncio.TimeoutError:
                continue  # Check stop_event and continue

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(
            f"Background message queue processor error for {connection_id}: {e}"
        )


@app.websocket("/ws")
async def adk_websocket_endpoint(websocket: WebSocket):
    """
    ADK-integrated WebSocket endpoint with real-time streaming support.
    Uses text-optimized model for SSE streaming.
    """

    # Check if agents are initialized (service ready)
    if TEXT_ROOT_AGENT is None:
        await websocket.close(code=1013, reason="Service initializing - please retry")
        logger.warning("WebSocket connection rejected - service still initializing")
        return

    # Use singleton text agent (shared across all connections)
    text_root_agent = TEXT_ROOT_AGENT

    connection_id = None
    session_id = None

    try:
        widget_id = websocket.query_params.get("widget_id")
        session_id_param = websocket.query_params.get("session_id")

        from auth.websocket_auth import authenticate_websocket_connection

        await websocket.accept()
        logger.info("WebSocket connection accepted")

        connection_allowed, user_info = await authenticate_websocket_connection(
            websocket=websocket,
            require_auth=False,  # TODO: Set back to True when auth service is available
            allowed_token_params=["token", "jwt_token", "drf_token"],
        )
        print(user_info)

        if not connection_allowed:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": WEBSOCKET_TYPE_ERROR,
                        "payload": {
                            "error_type": "AUTHENTICATION_FAILED",
                            "message": "Authentication required. Please provide a valid token.",
                            "timestamp": asyncio.get_event_loop().time(),
                        },
                    }
                )
            )
            await websocket.close(code=4001, reason="Authentication failed")
            return

        if session_id_param:
            try:
                session_id = uuid.UUID(session_id_param)
            except ValueError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": WEBSOCKET_TYPE_ERROR,
                            "payload": {
                                "error_type": "INVALID_SESSION_ID",
                                "message": f"Invalid session_id format: {session_id_param}",
                                "timestamp": asyncio.get_event_loop().time(),
                            },
                        }
                    )
                )
                await websocket.close(code=1008, reason="Invalid session_id")
                return
        else:
            session_id = uuid.uuid4()

        try:
            await ChatHistory.get_or_create_session(session_id=session_id)
            logger.info(f"Database session: {session_id}")
        except Exception as e:
            logger.error(f"Database error for session {session_id}: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": WEBSOCKET_TYPE_ERROR,
                        "payload": {
                            "error_type": "DATABASE_ERROR",
                            "message": "Failed to create/retrieve session",
                            "timestamp": asyncio.get_event_loop().time(),
                        },
                    }
                )
            )
            await websocket.close(code=1011, reason="Database error")
            return

        if manager.is_session_connected(str(session_id)):
            existing_conn_id = manager.get_connection_by_session(str(session_id))
            logger.info(
                f"Session {session_id} reconnecting - disconnecting old connection {existing_conn_id}"
            )
            await manager.disconnect(existing_conn_id)

        connection_id = manager.register_connection(
            websocket=websocket, session_id=str(session_id)
        )

        # Set WebSocket context for layer query tools
        set_ws_context(manager, connection_id)

        await manager.send_message(
            connection_id,
            {
                "type": WEBSOCKET_TYPE_SESSION,
                "payload": {
                    "sessionId": str(session_id),
                    "connection_id": connection_id,
                    "streaming_enabled": True,
                    "user_id": user_info.get("id") if user_info else None,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            },
        )

        logger.info(
            f"Connection {connection_id} fully established for session {session_id}"
        )

        # Start context preloading in background (non-blocking)
        # Preloads user preferences, saved locations, etc. for faster first query
        user_id = user_info.get("id") if user_info else None
        org_id = user_info.get("organization_id") if user_info else None
        await preload_session_context(str(session_id), user_id, org_id)
        logger.info(f"Started context preloading for session {session_id}")

        # Note: Audio streaming is now handled by the dedicated /ws/audio endpoint
        # This endpoint handles text chat with background processing for PATCH/PING

        # Create message queue and stop event for background task
        message_queue = asyncio.Queue()
        stop_event = asyncio.Event()
        background_task = None

        try:
            # Start background queue processor for PATCH/PING/MAP_CONTEXT handling
            background_task = asyncio.create_task(
                background_message_queue_processor(
                    message_queue, connection_id, str(session_id), stop_event
                )
            )
            logger.info(
                f"Started background message queue processor for {connection_id}"
            )

            # Main loop - reads from websocket and routes messages
            while True:
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    message_type = message.get("type", "CHAT/SEND")

                    logger.info(
                        f"Received message from {connection_id}: type={message_type}"
                    )

                    # Route based on message type
                    if message_type == WEBSOCKET_TYPE_SEND:
                        # Process CHAT/SEND synchronously in main loop
                        try:
                            await process_adk_message(
                                connection_id,
                                session_id,
                                message,
                                user_info,
                                text_root_agent,
                            )
                        except Exception as e:
                            # Send error to websocket for retry
                            import time

                            await manager.send_message(
                                connection_id,
                                {
                                    "type": WEBSOCKET_TYPE_ERROR,
                                    "payload": {
                                        "error_type": "AGENT_PROCESSING_ERROR",
                                        "message": str(e),
                                        "retriable": True,
                                        "timestamp": time.time(),
                                    },
                                },
                            )
                            logger.error(
                                f"Agent processing error for {connection_id}: {e}"
                            )

                    elif message_type == "LAYER/QUERY_RESPONSE":
                        # Handle layer query responses from frontend
                        payload = message.get("payload", {})
                        handle_layer_query_response(payload)
                        logger.debug(f"Processed LAYER/QUERY_RESPONSE for {connection_id}")

                    elif message_type in [
                        WEBSOCKET_TYPE_PATCH,
                        WEBSOCKET_TYPE_PING,
                        WEBSOCKET_TYPE_MAP_CONTEXT,
                    ]:
                        # Queue for background processing - non-blocking
                        await message_queue.put(message)
                        logger.debug(f"Queued {message_type} for background processing")

                except json.JSONDecodeError:
                    await manager.send_message(
                        connection_id,
                        {
                            "type": "error",
                            "connection_id": connection_id,
                            "message": "Invalid JSON format",
                        },
                    )

        finally:
            # Stop background task
            stop_event.set()
            if background_task:
                background_task.cancel()
                try:
                    await background_task
                except asyncio.CancelledError:
                    pass
            logger.info(f"Stopped background queue processor for {connection_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected")
        await manager.disconnect(connection_id)
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        if connection_id:
            logger.info(f"Cleaned up connection {connection_id}")
            await manager.disconnect(connection_id)
        if session_id:
            # Clean up preloaded context when session ends
            await invalidate_session_context(str(session_id))


@app.websocket("/ws/audio")
async def adk_audio_websocket_endpoint(websocket: WebSocket):
    """
    Dedicated BIDI audio streaming endpoint (Google ADK pattern).

    Uses two concurrent tasks for bidirectional communication:
    - agent_to_client_messaging: Streams ADK events to client
    - client_to_agent_messaging: Forwards client audio to ADK

    This endpoint follows Google's recommended pattern from the official
    ADK documentation for optimal audio streaming performance.
    Uses audio-optimized model for BIDI streaming.
    """

    # Check if agents are initialized (service ready)
    if AUDIO_ROOT_AGENT is None:
        await websocket.close(code=1013, reason="Service initializing - please retry")
        logger.warning(
            "Audio WebSocket connection rejected - service still initializing"
        )
        return

    # Use singleton audio agent (shared across all connections)
    audio_root_agent = AUDIO_ROOT_AGENT

    connection_id = None
    session_id = None

    try:
        # ==================== Auth & Session Setup ====================
        # Extract query parameters
        widget_id = websocket.query_params.get("widget_id")
        session_id_param = websocket.query_params.get("session_id")

        from auth.websocket_auth import authenticate_websocket_connection

        # Authenticate connection
        connection_allowed, user_info = await authenticate_websocket_connection(
            websocket=websocket,
            require_auth=False,  # TODO: Set back to True when auth service is available
            allowed_token_params=["token", "jwt_token", "drf_token"],
        )

        # Accept WebSocket
        await websocket.accept()
        logger.info("Audio WebSocket connection accepted")

        # Handle authentication failure
        if not connection_allowed:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": WEBSOCKET_TYPE_ERROR,
                        "payload": {
                            "error_type": "AUTHENTICATION_FAILED",
                            "message": "Authentication required. Please provide a valid token.",
                            "timestamp": asyncio.get_event_loop().time(),
                        },
                    }
                )
            )
            await websocket.close(code=4001, reason="Authentication failed")
            return

        # Validate or create session_id
        if session_id_param:
            try:
                session_id = uuid.UUID(session_id_param)
            except ValueError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": WEBSOCKET_TYPE_ERROR,
                            "payload": {
                                "error_type": "INVALID_SESSION_ID",
                                "message": f"Invalid session_id format: {session_id_param}",
                                "timestamp": asyncio.get_event_loop().time(),
                            },
                        }
                    )
                )
                await websocket.close(code=1008, reason="Invalid session_id")
                return
        else:
            session_id = uuid.uuid4()

        # Create/retrieve database session
        try:
            await ChatHistory.get_or_create_session(session_id=session_id)
            logger.info(f"Audio session: {session_id}")
        except Exception as e:
            logger.error(f"Database error for session {session_id}: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": WEBSOCKET_TYPE_ERROR,
                        "payload": {
                            "error_type": "DATABASE_ERROR",
                            "message": "Failed to create/retrieve session",
                            "timestamp": asyncio.get_event_loop().time(),
                        },
                    }
                )
            )
            await websocket.close(code=1011, reason="Database error")
            return

        # Check for duplicate session
        if manager.is_session_connected(str(session_id)):
            existing_conn_id = manager.get_connection_by_session(str(session_id))
            logger.warning(
                f"Audio session {session_id} already connected with {existing_conn_id}"
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": WEBSOCKET_TYPE_ERROR,
                        "payload": {
                            "error_type": "SESSION_ALREADY_CONNECTED",
                            "message": f"Session {session_id} is already connected. Disconnect existing session first.",
                            "session_id": str(session_id),
                            "existing_connection_id": existing_conn_id,
                            "timestamp": asyncio.get_event_loop().time(),
                        },
                    }
                )
            )
            await websocket.close(code=1008, reason="Session already connected")
            return

        # Register connection
        connection_id = manager.register_connection(
            websocket=websocket, session_id=str(session_id)
        )

        # Send session info
        await manager.send_message(
            connection_id,
            {
                "type": WEBSOCKET_TYPE_SESSION,
                "payload": {
                    "sessionId": str(session_id),
                    "connection_id": connection_id,
                    "streaming_enabled": True,
                    "audio_mode": True,
                    "user_id": user_info.get("user_id") if user_info else None,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            },
        )

        logger.info(
            f"Audio connection {connection_id} fully established for session {session_id}"
        )

        # ==================== Initialize Audio Streaming ====================
        # Prepare extra information
        extra_information = {}
        if user_info:
            extra_information["user_info"] = user_info

        # Get live_events and live_request_queue from audio_root_agent with shared Runner
        live_events, live_request_queue = await audio_root_agent.start_audio_streaming(
            session_id=str(session_id),
            connection_id=connection_id,
            extra_information=extra_information,
            runner=AUDIO_RUNNER,
        )

        # Register live_queue with manager
        manager.set_live_queue(connection_id, live_request_queue)
        logger.info(f"LiveRequestQueue registered for audio connection {connection_id}")

        # ==================== Launch Two Concurrent Tasks ====================
        # Following Google ADK's recommended pattern
        agent_to_client_task = asyncio.create_task(
            agent_to_client_messaging(
                websocket, connection_id, str(session_id), live_events
            )
        )
        client_to_agent_task = asyncio.create_task(
            client_to_agent_messaging(websocket, connection_id, live_request_queue)
        )

        tasks = [agent_to_client_task, client_to_agent_task]
        logger.info(
            f"Started two concurrent tasks for audio connection {connection_id}"
        )

        # Wait for first exception or completion (Google ADK recommended pattern)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        # Extract and log exceptions from completed tasks
        for task in done:
            exception = task.exception()
            if exception:
                logger.error(
                    f"Audio streaming task failed for {connection_id}: {exception}"
                )

        # Cancel any pending tasks to prevent resource leaks
        for task in pending:
            task.cancel()
            logger.info(f"Cancelled pending task for {connection_id}")

        logger.info(f"Audio streaming tasks completed for {connection_id}")

    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected: {connection_id}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Audio WebSocket error for {connection_id}: {e}")
    finally:
        # Cleanup
        if connection_id:
            logger.info(f"Cleaning up audio connection {connection_id}")
            await manager.disconnect(connection_id)


async def process_adk_message(
    connection_id: str,
    session_id: uuid.UUID,
    message: dict,
    user_info: dict = None,
    root_agent=None,
) -> dict:
    """
    Process user message through Google ADK Runner with streaming support.

    Args:
        connection_id: WebSocket connection identifier
        session_id: Session UUID
        message: Message dictionary
        user_info: User information dictionary
        root_agent: RootAgent instance (required)
    """
    user_query = message.get("payload", {}).get("message")
    message_type = message.get("type", "query")
    extra_information = message.get("payload", {})

    # Add user information to extra_information
    if user_info:
        extra_information["user_info"] = user_info

    # Handle echo messages for testing
    if message_type == WEBSOCKET_TYPE_PING:
        return await manager.send_message(
            connection_id,
            {
                "type": WEBSOCKET_TYPE_PONG,
                "connection_id": connection_id,
                "session_id": str(session_id),
            },
        )

    try:
        # Handle CHAT/PATCH - Update map_context state via JSON Patch operations
        if message_type == WEBSOCKET_TYPE_PATCH:
            logger.info(f"Processing CHAT/PATCH for session {session_id}")

            try:
                # Extract patch operations from payload
                patched_data = message.get("payload", {}).get("patched_data", [])

                if not patched_data:
                    logger.warning(
                        f"No patched_data found in CHAT/PATCH message for session {session_id}"
                    )
                    return None

                # Get session from DatabaseSessionService

                # Try to get existing session first
                session = await session_service.get_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=str(session_id),
                )

                if session is None:
                    logger.info(
                        f"Session not found for {session_id}, creating new session"
                    )
                    session = await session_service.create_session(
                        app_name="root_agent_app",
                        user_id=f"user_{session_id}",
                        session_id=str(session_id),
                    )

                # Get current map_context or initialize empty dict
                current_map_context = session.state.get("map_context", {})

                # Apply JSON Patch operations
                patched_map_context = jsonpatch.apply_patch(
                    current_map_context, patched_data
                )

                # Update session state
                session.state["map_context"] = patched_map_context
                event = Event(
                    id=str(uuid.uuid4()),
                    invocation_id=str(uuid.uuid4()),
                    author="system",
                    actions=EventActions(
                        state_delta={
                            "map_context": patched_map_context,
                        }
                    ),
                    timestamp=datetime.now().timestamp(),
                )
                await session_service.append_event(session, event)
                logger.info(
                    f"Applied {len(patched_data)} patch operations to map_context for session {session_id}"
                )
                logger.debug(f"Updated map_context")
                return None  # No response needed for PATCH operations

            except jsonpatch.JsonPatchException as e:
                logger.error(f"Patched Data:{patched_data}")
                logger.error(
                    f"JSON Patch error for session {session_id}: {str(e)[:100]}"
                )
                return None
            except Exception as e:
                logger.error(f"Patched Data:{patched_data}")
                logger.error(
                    f"Error processing CHAT/PATCH for session {session_id}: {str(e)[:100]}"
                )
                import traceback

                traceback.print_exc()
                return None
        if message_type == WEBSOCKET_TYPE_MAP_CONTEXT:
            try:
                session = await session_service.create_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=str(session_id),
                )
            except Exception as e:
                session = await session_service.get_session(
                    app_name="root_agent_app",
                    user_id=f"user_{session_id}",
                    session_id=str(session_id),
                )
            session.state["session_id"] = str(session_id)
            session.state["connection_id"] = str(connection_id)
            if extra_information.get("map_context"):
                session.state["map_context"] = extra_information.get("map_context")

            event = Event(
                id=str(uuid.uuid4()),
                invocation_id=str(uuid.uuid4()),
                author="system",
                actions=EventActions(
                    state_delta={
                        "connection_id": str(connection_id),
                        "session_id": str(session_id),
                        "map_context": extra_information.get("map_context"),
                    }
                ),
                timestamp=datetime.now().timestamp(),
            )
            await session_service.append_event(session, event)
        if message_type == WEBSOCKET_TYPE_SEND:
            logger.info(
                f"Processing ADK query in session {session_id}: {user_query[:100] if user_query else 'None'}..."
            )

            # Query Deduplication: Prevent duplicate API calls for same query
            # This handles double-clicks or rapid resubmissions
            is_duplicate, in_flight = await query_deduplicator.check_or_register(
                str(session_id), user_query
            )

            if is_duplicate:
                # Wait for existing query's result instead of processing again
                logger.info(
                    f"Duplicate query detected for session {session_id}, waiting for existing result"
                )
                try:
                    await in_flight.wait_for_result(timeout=60.0)
                    logger.info(f"Duplicate query got result from original for session {session_id}")
                except TimeoutError:
                    logger.warning(f"Timeout waiting for duplicate query result in session {session_id}")
                except Exception as e:
                    logger.error(f"Error waiting for duplicate query: {e}")
                return  # Result already streamed to client by original query

            # Process query through in-memory ADK Runner for faster performance
            # The Runner uses in-memory session service during processing
            # Session is persisted to database after response completes
            try:
                await root_agent.process_query(
                    user_query,
                    str(session_id),
                    connection_id,
                    extra_information,
                    IN_MEMORY_TEXT_RUNNER,
                )
                # Mark query as complete so waiting duplicates get notified
                query_deduplicator.complete(in_flight, {"status": "success"})
            except Exception as e:
                # Mark query as failed so waiting duplicates get the error
                query_deduplicator.fail(in_flight, e)
                raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error processing ADK message: {e}")
        return {
            "type": "error",
            "connection_id": connection_id,
            "session_id": session_id,
            "message": f"ADK processing error: {str(e)}",
            "timestamp": asyncio.get_event_loop().time(),
        }
