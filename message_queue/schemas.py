"""
Message schemas for queue operations.

This module defines data structures and validation for queue messages.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

from pydantic import BaseModel, Field


class MessageMetadata(BaseModel):
    """
    Metadata for queue messages.
    """

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "bumblebee_app"
    version: str = "2.0.0"
    environment: str = "development"
    user_id: Optional[int] = None
    session_id: Optional[str] = None


class QueueMessage(BaseModel):
    """
    Base queue message structure.
    """

    operation: str = Field(..., description="Database operation: CREATE/UPDATE/DELETE")
    model_name: str = Field(..., description="Database table name")
    payload: Dict[str, Any] = Field(..., description="Operation data")
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)


# Chat History Message Schemas


class ChatHistoryCreatePayload(BaseModel):
    """
    Payload for creating a chat session.
    """

    session_id: str
    user_id: Optional[int] = None
    session_title: Optional[str] = None
    session_type: str = "CHATBOT"
    widget_id: Optional[int] = None
    widget_history: bool = False
    chat_summary: str = ""
    weather_data: list = Field(default_factory=list)
    chat: list = Field(default_factory=list)


class ChatHistoryUpdatePayload(BaseModel):
    """
    Payload for updating a chat session.
    """

    session_id: str
    data: Dict[str, Any]  # Fields to update


class ChatHistoryDeletePayload(BaseModel):
    """
    Payload for deleting a chat session.
    """

    session_id: str


# Chat Message Message Schemas


class ChatMessageCreatePayload(BaseModel):
    """
    Payload for creating a chat message.
    """

    session_id: str
    user_type: str
    message: str
    stream_id: Optional[str] = None
    stop_stream: bool = False
    events_data: Dict[str, Any] = Field(default_factory=dict)
    assets_data: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageUpdatePayload(BaseModel):
    """
    Payload for updating a chat message.
    """

    message_id: int
    data: Dict[str, Any]  # Fields to update


class ChatMessageDeletePayload(BaseModel):
    """
    Payload for deleting chat messages.
    """

    session_id: Optional[str] = None  # Delete all messages for session
    message_id: Optional[int] = None  # Delete specific message


# Message Factory Functions


def create_chat_history_message(
    operation: str,
    payload: Union[
        ChatHistoryCreatePayload, ChatHistoryUpdatePayload, ChatHistoryDeletePayload
    ],
    metadata: Optional[Dict[str, Any]] = None,
) -> QueueMessage:
    """
    Create a chat history queue message.

    Args:
        operation: CREATE/UPDATE/DELETE
        payload: Operation-specific payload
        metadata: Additional metadata

    Returns:
        Formatted queue message
    """
    message_metadata = MessageMetadata()
    if metadata:
        for key, value in metadata.items():
            if hasattr(message_metadata, key):
                setattr(message_metadata, key, value)

    return QueueMessage(
        operation=operation,
        model_name="user_app_chathistory",
        payload=payload.dict() if hasattr(payload, "dict") else payload,
        metadata=message_metadata,
    )


def create_chat_message_message(
    operation: str,
    payload: Union[
        ChatMessageCreatePayload, ChatMessageUpdatePayload, ChatMessageDeletePayload
    ],
    metadata: Optional[Dict[str, Any]] = None,
) -> QueueMessage:
    """
    Create a chat message queue message.

    Args:
        operation: CREATE/UPDATE/DELETE
        payload: Operation-specific payload
        metadata: Additional metadata

    Returns:
        Formatted queue message
    """
    message_metadata = MessageMetadata()
    if metadata:
        for key, value in metadata.items():
            if hasattr(message_metadata, key):
                setattr(message_metadata, key, value)

    return QueueMessage(
        operation=operation,
        model_name="user_app_chatinputoutput",
        payload=payload.dict() if hasattr(payload, "dict") else payload,
        metadata=message_metadata,
    )


# Helper Functions


def validate_message(message_data: Dict[str, Any]) -> QueueMessage:
    """
    Validate and parse a queue message.

    Args:
        message_data: Raw message data

    Returns:
        Validated queue message

    Raises:
        ValidationError: If message is invalid
    """
    return QueueMessage(**message_data)


def get_payload_schema(model_name: str, operation: str) -> type:
    """
    Get the appropriate payload schema class.

    Args:
        model_name: Database table name
        operation: Database operation

    Returns:
        Payload schema class

    Raises:
        ValueError: If no schema found
    """
    schema_map = {
        "user_app_chathistory": {
            "CREATE": ChatHistoryCreatePayload,
            "UPDATE": ChatHistoryUpdatePayload,
            "DELETE": ChatHistoryDeletePayload,
        },
        "user_app_chatinputoutput": {
            "CREATE": ChatMessageCreatePayload,
            "UPDATE": ChatMessageUpdatePayload,
            "DELETE": ChatMessageDeletePayload,
        },
    }

    if model_name not in schema_map:
        raise ValueError(f"Unknown model: {model_name}")

    if operation not in schema_map[model_name]:
        raise ValueError(f"Unknown operation for {model_name}: {operation}")

    return schema_map[model_name][operation]
