"""
Streaming Callbacks System for Real-time Agent Execution Visibility

This module implements Google ADK callbacks for streaming tool execution,
thinking tokens, and agent coordination in real-time via WebSocket.
"""

import logging
from typing import Dict, Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from managers.websocket_manager import manager

logger = logging.getLogger(__name__)


def after_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    """
    Callback executed after tool execution completes.

    Tool execution data is extracted directly from events in root_agent.py,
    so this callback only performs logging.
    """
    agent_name = tool_context.agent_name
    tool_name = tool.name
    logger.info(f"Tool executed: {tool_name} by {agent_name}")
    return None


async def before_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    # agent_name = tool_context.agent_name
    # tool_name = tool.name
    connection_id = tool_context.state.get("connection_id")

    await manager.send_thinking_message(
        connection_id,
        {"response": f"Retrieving the requested information…"},
    )
    return None


async def after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    print(f"[Callback] After model call for agent: {agent_name}")
    if agent_name == "marketing_agent":
        await manager.send_message(
            callback_context.state.get("connection_id"),
            {"response": f"Generating marketing posts..."},
        )


async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    last_user_message = ""
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")
    if agent_name == "marketing_agent":
        await manager.send_message(
            callback_context.state.get("connection_id"),
            {"response": f"Generating marketing posts... 🚀"},
        )
