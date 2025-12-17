"""
Unified event handler for ADK streaming events.

Consolidates event processing logic for both SSE and BIDI streaming modes.
"""

import asyncio
import datetime
import json
import logging
import time
import uuid
from typing import Any

from config.config import STREAM_CHUNK_DELAY
from constants.chat_constants import (
    WEBSOCKET_TYPE_STREAM,
    WEBSOCKET_TYPE_STREAM_INFO,
    WEBSOCKET_TYPE_TURN_STATUS,
    ProgressPhase,
)
from dao import ChatInputOutputDAO
from utils.enums import UserTypeConstants

logger = logging.getLogger(__name__)


class UnifiedEventHandler:
    """
    Unified event handler for both SSE and BIDI streaming modes.

    Modes:
    - SSE: Text-only streaming with tool execution and 3-state completion detection
    - BIDI: Audio + text streaming with turn status and interrupts

    Architecture:
    - Eliminates ~215 lines of duplicate code between root_agent.py and main.py
    - Provides consistent event processing across both streaming modes
    - Maintains all functionality while improving code readability
    """

    def __init__(
        self, connection_id: str, session_id: str, manager: Any, mode: str = "SSE"
    ):
        """
        Initialize the unified event handler.

        Args:
            connection_id: WebSocket connection ID
            session_id: User session ID
            manager: WebSocket manager instance
            mode: Streaming mode - "SSE" or "BIDI"
        """
        self.connection_id = connection_id
        self.session_id = session_id
        self.manager = manager
        self.mode = mode.upper()
        self.response_text = ""
        self.last_tool_name = None  # Track last tool for sequence detection
        self.invocation_tool_cache = (
            {}
        )  # Track tools per invocation: {invocation_id: [(tool_name, tool_result), ...]}

        # Progress phase tracking (send each phase only once per session)
        self._progress_phases_sent = set()
        self._current_agent = None  # Track which sub-agent is active

        if self.mode not in ["SSE", "BIDI"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'SSE' or 'BIDI'")

        logger.info(
            f"Initialized UnifiedEventHandler in {self.mode} mode for connection {connection_id}"
        )

    async def _send_progress_phase(self, phase: str, agent: str = None) -> None:
        """
        Send a progress phase update (only once per phase).

        Args:
            phase: Progress phase to send
            agent: Optional agent name for context
        """
        # Only send each phase once per event handler lifetime
        phase_key = f"{phase}:{agent or 'root'}"
        if phase_key in self._progress_phases_sent:
            return

        self._progress_phases_sent.add(phase_key)
        await self.manager.send_progress_message(
            self.connection_id, phase, agent=agent
        )

    async def handle_event(self, event: Any) -> None:
        """
        Main event dispatcher - routes events to appropriate handlers.

        Args:
            event: ADK streaming event
        """
        try:
            # BIDI mode: Skip user echoes AND sub-agent responses
            if self.mode == "BIDI":
                is_model_response = hasattr(event, "author") and event.author != "user"
                if not is_model_response:
                    return

            # Detect sub-agent routing and send progress phase
            if hasattr(event, "author") and event.author:
                author = event.author
                # Sub-agents have names like "gis_agent", "salesforce_agent", etc.
                if author != "user" and "_agent" in author and author != "root_agent":
                    if self._current_agent != author:
                        self._current_agent = author
                        # Format agent name for display (e.g., "gis_agent" -> "GIS Agent")
                        display_name = author.replace("_", " ").title()
                        await self._send_progress_phase(ProgressPhase.ROUTING, display_name)

            if hasattr(event, "content") and event.content:
                await self._process_content(event)

            # Process non-content events (thinking, turn status)
            await self._process_non_content_events(event)

        except Exception as e:
            logger.error(
                f"Error handling event in {self.mode} mode for {self.connection_id}: {e}"
            )

    async def _process_content(self, event: Any) -> None:
        """
        Process content events (text, audio, tool execution).

        Args:
            event: Event with content
        """
        if not hasattr(event.content, "parts") or not event.content.parts:
            return

        # SSE mode: Process all parts (tool responses and text)
        if self.mode == "SSE":
            for part in event.content.parts:
                # Tool execution results
                if hasattr(part, "function_response") and part.function_response:
                    await self._handle_tool_execution(event, part)

                # Text streaming
                if hasattr(part, "text") and part.text:
                    await self._handle_text_content(event, part)

        # BIDI mode: Process first part only
        elif self.mode == "BIDI":
            part = event.content.parts[0]

            # Text streaming
            if hasattr(part, "text") and part.text:
                await self._handle_text_content(event, part)

            if hasattr(part, "function_response") and part.function_response:
                await self._handle_tool_execution(event, part)

            # Audio streaming
            if hasattr(part, "inline_data") and part.inline_data:
                await self._handle_audio_content(event, part)

    async def _process_non_content_events(self, event: Any) -> None:
        """
        Process non-content events (thinking, turn status).

        Args:
            event: Event to process
        """
        # BIDI mode: Handle thinking events
        if self.mode == "BIDI":
            if hasattr(event, "thinking") and event.thinking:
                await self.manager.send_thinking_message(
                    self.connection_id, f"Processing: {event.thinking[:100]}..."
                )

            # Handle audio transcription (text representation of audio response)
            if hasattr(event, "output_transcription") and event.output_transcription:
                if (
                    hasattr(event.output_transcription, "text")
                    and event.output_transcription.text
                ):
                    await self._handle_audio_transcription(event)

        # BIDI mode: Handle turn status
        if self.mode == "BIDI":
            if hasattr(event, "turn_complete") or hasattr(event, "interrupted"):
                await self._handle_turn_status(event)

    def _is_last_tool_in_turn(self, event: Any) -> bool:
        """
        Detect if this tool is the last in the current turn.

        Uses multiple heuristics to determine if more tools will be called:
        1. Check turn_complete flag (most reliable)
        2. Check finish_reason (model done with generation)
        3. Look ahead in event parts for subsequent function_calls

        Args:
            event: Current event being processed

        Returns:
            True if this appears to be the last tool, False otherwise
        """
        # Method 1: Check turn_complete flag (BIDI mode)
        if hasattr(event, "turn_complete") and event.turn_complete:
            return True

        # Method 2: Check finish_reason (model completed generation)
        if hasattr(event, "finish_reason") and event.finish_reason:
            return True

        # Method 3: Look ahead in parts for subsequent function calls
        if hasattr(event, "content") and event.content:
            parts = event.content.parts
            if len(parts) > 1:
                # Find current function_response index
                current_idx = None
                for i, p in enumerate(parts):
                    if hasattr(p, "function_response") and p.function_response:
                        current_idx = i
                        break

                if current_idx is not None and current_idx < len(parts) - 1:
                    # Check subsequent parts
                    for next_part in parts[current_idx + 1 :]:
                        # If next part is another function_call, this is NOT last
                        if (
                            hasattr(next_part, "function_call")
                            and next_part.function_call
                        ):
                            return False
                        # If next part is text, this tool is likely last
                        if hasattr(next_part, "text") and next_part.text:
                            return True

        # Method 4: Check if this is a function_response event from agent
        # Function responses come in separate events, so we need to be conservative
        if hasattr(event, "content") and event.content:
            if hasattr(event.content, "parts") and event.content.parts:
                part = event.content.parts[0]
                if hasattr(part, "function_response") and part.function_response:
                    # This is a function_response event - default to NOT last
                    # unless we have clear indicators above
                    return False

        # Default: assume last only if we have positive indicators
        # For model responses (text), default to True
        return True

    # ========================================================================
    # Tool Execution Handlers (SSE Mode Only)
    # ========================================================================

    async def _handle_tool_execution(self, event: Any, part: Any) -> None:
        """
        Handle tool execution results from sub-agents.

        Uses invocation tracking to cache tool results and send only the last tool
        when the turn completes (detected by model response or new invocation).

        Args:
            event: Event containing tool execution
            part: Part with function_response
        """
        agent_name = (
            event.author
        )  # e.g., "gis_agent", "salesforce_agent"
        tool_name = part.function_response.name
        tool_result = part.function_response.response.get("result", "")
        invocation_id = getattr(event, "invocation_id", None)

        logger.info(f"Tool execution detected - Agent: {agent_name}, Tool: {tool_name}")

        # Send "executing" progress phase
        display_name = agent_name.replace("_", " ").title() if agent_name else None
        await self._send_progress_phase(ProgressPhase.EXECUTING, display_name)

        # Cache tool for this invocation
        if invocation_id not in self.invocation_tool_cache:
            self.invocation_tool_cache[invocation_id] = []

        self.invocation_tool_cache[invocation_id].append(
            {
                "agent_name": agent_name,
                "tool_name": tool_name,
                "tool_result": tool_result,
                "event": event,
            }
        )

        logger.debug(
            f"Cached tool {tool_name} for invocation {invocation_id} "
            f"(total: {len(self.invocation_tool_cache[invocation_id])})"
        )

    async def _handle_gis_tool(
        self, tool_result: Any, tool_name: str, is_last_tool: bool
    ) -> None:
        """
        Handle GIS agent tool execution with context-aware filtering.

        Uses dynamic tool configuration to determine if this tool's result
        should be sent to frontend based on:
        - Tool's send_mode ("always", "never", "auto")
        - Whether this is the last tool in the turn

        Args:
            tool_result: Tool execution result (already in operations format)
            tool_name: Name of the executed tool
            is_last_tool: Whether this is the last tool in the current turn
        """
        from config.tool_config import should_send_tool_to_frontend

        # Dynamic decision based on tool config and context
        should_send = should_send_tool_to_frontend(tool_name, is_last_tool)

        if not should_send:
            logger.debug(
                f"Skipping tool result (mode check): {tool_name} "
                f"(last={is_last_tool})"
            )
            return

        try:
            # Tool result is already in operations format
            operation_data = (
                json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            )

            await self.manager.send_operations_data(self.connection_id, operation_data)

            logger.info(f"Sent GIS operation data for tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error sending GIS operation data: {e}")

    async def _handle_salesforce_tool(
        self, tool_name: str, tool_result: Any, is_last_tool: bool
    ) -> None:
        """
        Handle Salesforce agent tool execution.

        Sends PLOT_GEOJSON operation for mapping tools:
        - export_query_to_geojson_tool
        - geocode_existing_records_tool
        - export_query_to_geojson

        Args:
            tool_name: Name of the executed tool
            tool_result: Tool execution result
            is_last_tool: Whether this is the last tool in the turn (for future filtering)
        """
        if tool_name not in [
            "export_query_to_geojson_tool",
            "geocode_existing_records_tool",
            "export_query_to_geojson",
        ]:
            return

        try:
            tool_response_json = (
                json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            )

            # Extract file_information from the tool response
            file_info = tool_response_json.get("file_information", {})

            if file_info:
                operation_data = {
                    "operations": [
                        {
                            "type": "PLOT_GEOJSON",
                            "payload": file_info,
                        }
                    ]
                }

                await self.manager.send_operations_data(
                    self.connection_id, operation_data
                )

                logger.info(
                    f"Sent Salesforce PLOT_GEOJSON operation for tool: {tool_name}"
                )

        except Exception as e:
            logger.error(f"Error sending Salesforce operation data: {e}")

    async def _flush_cached_tools(self, invocation_id: str) -> None:
        """
        Flush cached tool results for an invocation.

        Sends only the last tool in the sequence (or tools with "always" mode).
        This is called when we detect a turn completion (model text response).

        Args:
            invocation_id: Invocation ID to flush
        """
        if invocation_id not in self.invocation_tool_cache:
            return

        cached_tools = self.invocation_tool_cache[invocation_id]
        if not cached_tools:
            return

        logger.debug(
            f"Flushing {len(cached_tools)} cached tools for invocation {invocation_id}"
        )

        # Performance optimization: Process tools in parallel using asyncio.gather
        handler_tasks = []
        for i, tool_data in enumerate(cached_tools):
            is_last_tool = i == len(cached_tools) - 1  # Last in sequence
            agent_name = tool_data["agent_name"]
            tool_name = tool_data["tool_name"]
            tool_result = tool_data["tool_result"]

            # Collect handler tasks for parallel execution
            if agent_name == "gis_agent" and tool_result:
                handler_tasks.append(self._handle_gis_tool(tool_result, tool_name, is_last_tool))

            if agent_name == "salesforce_agent":
                handler_tasks.append(self._handle_salesforce_tool(tool_name, tool_result, is_last_tool))

            # Marketing agent returns text directly, no special handling needed

        # Execute all tool handlers in parallel
        if handler_tasks:
            await asyncio.gather(*handler_tasks, return_exceptions=True)

        # Clear cache for this invocation
        del self.invocation_tool_cache[invocation_id]
        logger.debug(f"Cleared tool cache for invocation {invocation_id}")

    async def _handle_text_content(self, event: Any, part: Any) -> None:
        """
        Handle text streaming - routes to mode-specific handler.

        Also flushes cached tools when we see model text (turn completion).

        Args:
            event: Event containing text
            part: Part with text content
        """
        # Send "generating" progress phase when text starts streaming
        await self._send_progress_phase(ProgressPhase.GENERATING)

        # Flush cached tools when we see model response (turn complete)
        if hasattr(event, "invocation_id") and event.invocation_id:
            await self._flush_cached_tools(event.invocation_id)

        if self.mode == "SSE":
            await self._handle_sse_text(event, part.text)
        elif self.mode == "BIDI":
            await self._handle_bidi_text(event, part.text)

    async def _handle_sse_text(self, event: Any, text: str) -> None:
        """
        Handle SSE text streaming with 3-state detection.

        States:
        1. Streaming chunk: partial=True, finish_reason=None
        2. Last streaming chunk: finish_reason="STOP"
        3. Final consolidated: partial=None, usage_metadata populated

        Args:
            event: Event containing text
            text: Text content
        """
        # Detect streaming states
        # Note
        # For Future Use
        # is_streaming_chunk = event.partial == True and event.finish_reason is None
        # is_last_streaming_chunk = event.finish_reason == "STOP"
        is_final_consolidated = (
            event.partial is None
            and event.usage_metadata
            and event.usage_metadata.total_token_count
        )

        if is_final_consolidated:
            # Final event with complete text + metadata + token usage
            # This is the typical Gemini response pattern
            await self._send_sse_completion(text, event.usage_metadata)

        else:
            words = text.split(" ")
            for word in words:
                await self.manager.send_streaming_message(
                    self.connection_id,
                    {
                        "stream_response": f"{word} ",
                        "stream_stop": False,
                        "stream_id": event.id,
                    },
                )
                await asyncio.sleep(STREAM_CHUNK_DELAY)

    async def _handle_bidi_text(self, event: Any, text: str) -> None:
        """
        Handle BIDI text streaming (simple mode).

        Args:
            event: Event containing text
            text: Text content
        """
        self.response_text += text

        await self.manager.send_message(
            self.connection_id,
            {
                "type": WEBSOCKET_TYPE_STREAM,
                "payload": {
                    "connection_id": self.connection_id,
                    "session_id": self.session_id,
                    "text": text,
                    "is_partial": getattr(event, "partial", True),
                    "timestamp": time.time(),
                },
            },
        )

    # ========================================================================
    # Audio Streaming Handlers (BIDI Mode Only)
    # ========================================================================

    async def _handle_audio_content(self, event: Any, part: Any) -> None:
        """
        Handle audio streaming in BIDI mode.

        - Sends audio chunks to client (non-blocking)
        - Strips audio data from event to save memory
        - Detects final chunk based on turn_complete or finish_reason

        Args:
            event: Event containing audio
            part: Part with inline_data (audio)
        """
        if not hasattr(part.inline_data, "data"):
            return

        audio_data = part.inline_data.data

        # Determine if this is the final audio chunk
        is_final = (
            getattr(event, "turn_complete", False)
            or getattr(event, "finish_reason", None) is not None
        )

        # Send audio to client (non-blocking)
        asyncio.create_task(
            self.manager.send_audio_to_client(
                self.connection_id,
                audio_data,
                is_final=is_final,
            )
        )

        # Strip audio data to save memory
        part.inline_data.data = b""

    async def _handle_audio_transcription(self, event: Any) -> None:
        """
        Handle audio transcription events (text representation of audio response).

        This provides accessibility support by sending text transcription of
        audio responses to the client for display.

        Args:
            event: Event with output_transcription
        """
        transcription_text = event.output_transcription.text

        logger.debug(f"[AUDIO_TRANSCRIPTION] : {transcription_text}")

        await self.manager.send_message(
            self.connection_id,
            {
                "type": "CHAT/AUDIO_TRANSCRIPTION",
                "payload": {
                    "text": transcription_text,
                    "timestamp": time.time(),
                },
            },
        )

        # logger.debug(f"Sent audio transcription: {transcription_text[:50]}...")

    # ========================================================================
    # Turn Status Handlers (BIDI Mode Only)
    # ========================================================================

    async def _handle_turn_status(self, event: Any) -> None:
        """
        Handle turn status events (turn_complete, interrupted).

        Following official ADK pattern: both flags in one message.

        Args:
            event: Event with turn status
        """
        turn_complete = getattr(event, "turn_complete", False)
        interrupted = getattr(event, "interrupted", False)

        if turn_complete or interrupted:
            await self.manager.send_message(
                self.connection_id,
                {
                    "type": WEBSOCKET_TYPE_TURN_STATUS,
                    "payload": {
                        "turn_complete": turn_complete,
                        "interrupted": interrupted,
                        "final_text": self.response_text if turn_complete else "",
                        "timestamp": time.time(),
                    },
                },
            )

            logger.info(
                f"Sent turn status: turn_complete={turn_complete}, "
                f"interrupted={interrupted}"
            )

    async def _send_sse_completion(self, final_text: str, usage_metadata: Any) -> None:
        """
        Send SSE completion message with token usage and save to database.

        Args:
            final_text: Complete response text
            usage_metadata: Token usage metadata from ADK
        """
        # Send completion message to client
        await self.manager.send_streaming_message(
            self.connection_id,
            {"stream_response": "", "stream_stop": True, "stream_id": None},
        )
        await self.manager.send_message(
            self.connection_id,
            {
                "type": WEBSOCKET_TYPE_STREAM_INFO,
                "payload": {
                    "final_text": final_text,
                    "stream_stop": True,
                    "token_usage": {
                        "total": usage_metadata.total_token_count,
                        "prompt": usage_metadata.prompt_token_count,
                        "completion": usage_metadata.candidates_token_count,
                    },
                    "finish_reason": "STOP",
                },
            },
        )

        # Save to database
        await ChatInputOutputDAO().create_message(
            session_id=uuid.UUID(self.session_id),
            user_type=UserTypeConstants.AI,
            message=final_text,
            stream_id=uuid.UUID(self.connection_id),
            stop_stream=True,
            events_data={
                "total": usage_metadata.total_token_count,
                "prompt": usage_metadata.prompt_token_count,
                "completion": usage_metadata.candidates_token_count,
            },
            assets_data={},
        )

    async def send_completion_message(self) -> None:
        """
        Send completion message (BIDI mode only).

        Called after event loop ends naturally.
        """
        if self.mode != "BIDI":
            return

        await self.manager.send_message(
            self.connection_id,
            {
                "type": "audio_streaming_complete",
                "connection_id": self.connection_id,
                "session_id": self.session_id,
                "final_text": self.response_text,
                "timestamp": time.time(),
            },
        )

        logger.info(f"Audio streaming completed for {self.connection_id}")
