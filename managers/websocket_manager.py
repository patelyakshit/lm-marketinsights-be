"""
Simple WebSocket connection manager with Google ADK integration.

Handles WebSocket connections and audio streaming with minimal complexity.
Simple is better than complex.
"""

import json
import logging
import uuid
from typing import Dict, Optional

from fastapi import WebSocket
from google.adk.agents import LiveRequestQueue

from config.audio_config import audio_config
from constants.chat_constants import (
    WEBSOCKET_TYPE_THINKING,
    WEBSOCKET_TYPE_STREAM,
    WEBSOCKET_TYPE_OPERATION_DATA,
    WEBSOCKET_TYPE_STREAM_AUDIO,
    WEBSOCKET_TYPE_PROGRESS,
    WEBSOCKET_TYPE_TASK_PROGRESS,
)
from utils.audio_utils import encode_audio_base64
from utils.json_parser import SocketJSONEncoder
import asyncio

logger = logging.getLogger(__name__)


class WebsocketConnectionManager:
    """
    WebSocket connection manager with Google ADK integration and streaming support.
    """

    def __init__(self):
        # Core connection management
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id

        # Audio streaming (Google ADK live request queues)
        self.connection_live_queues: Dict[str, LiveRequestQueue] = {}

    def register_connection(
        self, websocket: WebSocket, session_id: str, connection_id: Optional[str] = None
    ) -> str:
        """
        Register an already-accepted WebSocket connection.

        Args:
            websocket: Already-accepted WebSocket
            session_id: Session ID to associate with this connection
            connection_id: Optional connection ID (generated if not provided)

        Returns:
            connection_id: The connection identifier
        """
        if not connection_id:
            connection_id = str(uuid.uuid4())

        self.active_connections[connection_id] = websocket
        self.connection_sessions[connection_id] = session_id
        self.session_connections[session_id] = connection_id

        logger.info(f"Connection {connection_id} registered with session {session_id}")
        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect WebSocket and unregister streaming session."""
        try:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

                if connection_id in self.connection_sessions:
                    session_id = self.connection_sessions[connection_id]
                    del self.connection_sessions[connection_id]

                    # Remove from session_connections mapping
                    if session_id in self.session_connections:
                        del self.session_connections[session_id]

                # Cleanup live queues (prevent memory leak)
                if connection_id in self.connection_live_queues:
                    try:
                        # Close the queue gracefully before deleting
                        live_queue = self.connection_live_queues[connection_id]
                        live_queue.close()
                        logger.info(f"Closed LiveRequestQueue for {connection_id}")
                    except Exception as e:
                        logger.warning(
                            f"Error closing LiveRequestQueue for {connection_id}: {e}"
                        )
                    finally:
                        del self.connection_live_queues[connection_id]

                logger.info(f"Connection {connection_id} disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting connection {connection_id}: {e}")

    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:

                # await asyncio.sleep(1)
                await websocket.send_text(json.dumps(message, cls=SocketJSONEncoder))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)

    async def send_operations_data(self, connection_id: str, message: dict):
        """Send operation data to client."""
        message = {"type": WEBSOCKET_TYPE_OPERATION_DATA, "payload": message}
        await self.send_message(connection_id, message)

    async def send_streaming_message(self, connection_id: str, message: dict):
        """Send streaming message to client."""
        message = {"type": WEBSOCKET_TYPE_STREAM, "payload": message}
        await self.send_message(connection_id, message)

    async def send_thinking_message(self, connection_id: str, message):
        """Send thinking message to client."""
        message = {"type": WEBSOCKET_TYPE_THINKING, "payload": message}
        await self.send_message(connection_id, message)

    async def send_progress_message(
        self, connection_id: str, phase: str, message: str = None, agent: str = None
    ):
        """
        Send progress phase update to client.

        Args:
            connection_id: WebSocket connection ID
            phase: Progress phase (understanding, routing, executing, generating)
            message: Optional custom message for the phase
            agent: Optional agent name (for routing/executing phases)
        """
        # Default messages for each phase
        phase_messages = {
            "understanding": "Understanding your request...",
            "routing": f"Routing to {agent}..." if agent else "Determining best approach...",
            "executing": f"Executing {agent} tools..." if agent else "Executing tools...",
            "generating": "Generating response...",
        }

        payload = {
            "phase": phase,
            "message": message or phase_messages.get(phase, "Processing..."),
            "timestamp": asyncio.get_event_loop().time(),
        }

        if agent:
            payload["agent"] = agent

        await self.send_message(
            connection_id, {"type": WEBSOCKET_TYPE_PROGRESS, "payload": payload}
        )

    async def send_task_progress(
        self,
        connection_id: str,
        tasks: list,
        current_task_id: str = None,
        action_type: str = None,
        action_detail: str = None,
    ):
        """
        Send detailed task progress update (Manus-style).

        Args:
            connection_id: WebSocket connection ID
            tasks: List of task objects with id, label, status, etc.
            current_task_id: ID of the currently active task
            action_type: Current action type (thinking, searching, browsing, etc.)
            action_detail: Detail about current action (search query, URL, etc.)
        """
        payload = {
            "tasks": tasks,
            "timestamp": asyncio.get_event_loop().time(),
        }

        if current_task_id:
            payload["current_task_id"] = current_task_id

        if action_type:
            payload["current_action"] = {
                "type": action_type,
                "detail": action_detail,
            }

        await self.send_message(
            connection_id, {"type": WEBSOCKET_TYPE_TASK_PROGRESS, "payload": payload}
        )

    def is_session_connected(self, session_id: str) -> bool:
        """Check if a session is already connected."""
        return session_id in self.session_connections

    def get_connection_by_session(self, session_id: str) -> Optional[str]:
        """Get connection ID for a session."""
        return self.session_connections.get(session_id)

    # ==================== Audio Streaming Methods ====================

    async def send_audio_to_client(
        self, connection_id: str, audio_bytes: bytes, is_final: bool = False
    ):
        """
        Send audio data to client.

        Args:
            connection_id: Connection to send audio to
            audio_bytes: Raw audio bytes (PCM)
            is_final: Whether this is the final audio chunk
        """
        try:
            if connection_id not in self.active_connections:
                logger.warning(
                    f"Attempted to send audio to non-existent connection {connection_id}"
                )
                return

            # Encode to Base64 for transmission
            audio_base64 = encode_audio_base64(audio_bytes)

            message = {
                "type": WEBSOCKET_TYPE_STREAM_AUDIO,
                "payload": {
                    "connection_id": connection_id,
                    "session_id": self.connection_sessions.get(connection_id),
                    "audio_chunk": audio_base64,
                    "sample_rate": audio_config.PLAYBACK_SAMPLE_RATE,
                    "stream_stop": is_final,
                },
            }

            await self.send_message(connection_id, message)

        except Exception as e:
            logger.error(f"Error sending audio to client {connection_id}: {e}")

    def set_live_queue(self, connection_id: str, live_queue: LiveRequestQueue):
        """
        Set the ADK LiveRequestQueue for a connection.

        Args:
            connection_id: Connection ID
            live_queue: LiveRequestQueue instance
        """
        self.connection_live_queues[connection_id] = live_queue
        logger.info(f"LiveRequestQueue set for connection {connection_id}")


# Global manager instance
manager = WebsocketConnectionManager()
