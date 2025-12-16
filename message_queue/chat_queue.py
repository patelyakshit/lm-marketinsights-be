"""
Chat-specific queue operations.

This module provides queue-based operations for chat history and chat messages,
replacing the traditional DAO pattern with RabbitMQ message publishing.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Union

from config.config import RoutingKeys
from .base_queue import BaseQueueDAO
from .schemas import (
    ChatHistoryCreatePayload,
    ChatMessageCreatePayload,
)

logger = logging.getLogger(__name__)


class ChatHistoryQueueDAO(BaseQueueDAO):
    """
    Queue-based operations for chat history.

    Replaces ChatHistoryDAO with RabbitMQ message publishing.
    """

    def __init__(self):
        super().__init__("user_app", "chathistory")

    def create_session(
        self,
        session_id: Union[str, uuid.UUID],
        user_id: Optional[int] = None,
        session_title: Optional[str] = None,
        session_type: str = "CHATBOT",
        widget_id: Optional[int] = None,
        widget_history: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a create session operation.

        Args:
            session_id: Session UUID
            user_id: User ID
            session_title: Session title
            session_type: Session type
            widget_id: Widget ID
            widget_history: Whether this is widget history
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        payload = ChatHistoryCreatePayload(
            session_id=str(session_id),
            user_id=user_id,
            session_title=session_title,
            session_type=session_type,
            widget_id=widget_id,
            widget_history=widget_history,
            chat_summary="",
            weather_data=[],
            chat=[],
        )

        return self.create_operation(
            routing_key=RoutingKeys.CHAT_HISTORY_CREATE,
            data=payload.dict(),
            metadata=metadata,
        )

    def get_or_create_session(
        self,
        session_id: Union[str, uuid.UUID],
        defaults: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], bool]:
        """
        Queue a get or create session operation.

        Note: This always queues a CREATE operation since we don't do reads.
        The consumer will handle the get-or-create logic.

        Args:
            session_id: Session UUID
            defaults: Default values for creation
            metadata: Additional metadata

        Returns:
            Tuple of (session data dict, True) - always reports as created
        """
        create_data = defaults or {}
        create_data.update(
            {
                "session_id": str(session_id),
                "user_id": create_data.get("user_id", 3),  # Default from original DAO
                "chat_summary": create_data.get("chat_summary", ""),
                "weather_data": create_data.get("weather_data", []),
                "chat": create_data.get("chat", []),
            }
        )

        success = self.create_session(
            session_id=session_id,
            user_id=create_data.get("user_id"),
            session_title=create_data.get("session_title"),
            session_type=create_data.get("session_type", "CHATBOT"),
            widget_id=create_data.get("widget_id"),
            widget_history=create_data.get("widget_history", False),
            metadata=metadata,
        )

        # Return a basic session structure
        session_data = {"session_id": str(session_id), **create_data}

        return session_data, success

    def update_chat_data(
        self,
        session_id: Union[str, uuid.UUID],
        chat_data: list,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update chat data operation.

        Args:
            session_id: Session UUID
            chat_data: Chat messages data
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_HISTORY_UPDATE,
            identifier={"session_id": str(session_id)},
            data={"chat": chat_data},
            metadata=metadata,
        )

    def update_session_title(
        self,
        session_id: Union[str, uuid.UUID],
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update session title operation.

        Args:
            session_id: Session UUID
            title: New session title
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_HISTORY_UPDATE,
            identifier={"session_id": str(session_id)},
            data={"session_title": title},
            metadata=metadata,
        )

    def update_chat_summary(
        self,
        session_id: Union[str, uuid.UUID],
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update chat summary operation.

        Args:
            session_id: Session UUID
            summary: Chat summary
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_HISTORY_UPDATE,
            identifier={"session_id": str(session_id)},
            data={"chat_summary": summary},
            metadata=metadata,
        )

    def delete_session(
        self,
        session_id: Union[str, uuid.UUID],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a delete session operation.

        Args:
            session_id: Session UUID to delete
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.delete_operation(
            routing_key=RoutingKeys.CHAT_HISTORY_DELETE,
            identifier={"session_id": str(session_id)},
            metadata=metadata,
        )


class ChatInputOutputQueueDAO(BaseQueueDAO):
    """
    Queue-based operations for chat input/output messages.

    Replaces ChatInputOutputDAO with RabbitMQ message publishing.
    """

    def __init__(self):
        super().__init__("user_app", "chatinputoutput")

    def create_message(
        self,
        session_id: Union[str, uuid.UUID],
        user_type: str,
        message: str,
        stream_id: Optional[Union[str, uuid.UUID]] = None,
        stop_stream: bool = False,
        events_data: Optional[Dict[str, Any]] = None,
        assets_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a create message operation.

        Args:
            session_id: Session UUID
            user_type: Type of user (Human or AI)
            message: Message content
            stream_id: Optional stream ID
            stop_stream: Whether to stop streaming
            events_data: Additional event data
            assets_data: Additional assets data
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        payload = ChatMessageCreatePayload(
            session_id=str(session_id),
            user_type=user_type,
            message=message,
            stream_id=str(stream_id) if stream_id else None,
            stop_stream=stop_stream,
            events_data=events_data or {},
            assets_data=assets_data or {},
        )

        return self.create_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_CREATE,
            data=payload.dict(),
            metadata=metadata,
        )

    def update_message_content(
        self,
        message_id: int,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update message content operation.

        Args:
            message_id: Message ID
            new_content: New message content
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_UPDATE,
            identifier={"id": message_id},
            data={"message": new_content},
            metadata=metadata,
        )

    def update_stream_status(
        self,
        message_id: int,
        stop_stream: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update stream status operation.

        Args:
            message_id: Message ID
            stop_stream: Whether to stop streaming
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_UPDATE,
            identifier={"id": message_id},
            data={"stop_stream": stop_stream},
            metadata=metadata,
        )

    def add_events_data(
        self,
        message_id: int,
        events_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update events data operation.

        Args:
            message_id: Message ID
            events_data: Events data to add
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_UPDATE,
            identifier={"id": message_id},
            data={"events_data": events_data},
            metadata=metadata,
        )

    def add_assets_data(
        self,
        message_id: int,
        assets_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an update assets data operation.

        Args:
            message_id: Message ID
            assets_data: Assets data to add
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.update_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_UPDATE,
            identifier={"id": message_id},
            data={"assets_data": assets_data},
            metadata=metadata,
        )

    def delete_session_messages(
        self,
        session_id: Union[str, uuid.UUID],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a delete session messages operation.

        Args:
            session_id: Session UUID
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self.delete_operation(
            routing_key=RoutingKeys.CHAT_MESSAGE_DELETE,
            identifier={"session_id": str(session_id)},
            metadata=metadata,
        )
