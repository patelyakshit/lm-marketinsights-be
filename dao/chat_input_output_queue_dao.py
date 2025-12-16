"""
Queue-based ChatInputOutput DAO replacement.

This module provides the same interface as ChatInputOutputDAO but uses
RabbitMQ message queuing instead of direct database operations.
"""

import uuid
import logging
from typing import Dict, Any, Optional, Union

from message_queue.chat_queue import ChatInputOutputQueueDAO

logger = logging.getLogger(__name__)


class ChatInputOutputDAO:
    """
    Queue-based replacement for ChatInputOutputDAO.

    Maintains the same interface as the original DAO but publishes
    operations to RabbitMQ queues instead of executing database operations.
    """

    def __init__(self):
        self.queue_dao = ChatInputOutputQueueDAO()

    async def create_message(
        self,
        session_id: Union[str, uuid.UUID],
        user_type: str,
        message: str,
        stream_id: Optional[Union[str, uuid.UUID]] = None,
        stop_stream: bool = False,
        events_data: Optional[Dict[str, Any]] = None,
        assets_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat message (via queue).

        Args:
            session_id: The session UUID this message belongs to
            user_type: Type of user (Human or AI)
            message: The message content
            stream_id: Optional stream ID for streaming messages
            stop_stream: Whether to stop streaming
            events_data: Additional event data
            assets_data: Additional assets data

        Returns:
            Mock ChatInputOutput instance data
        """
        success = self.queue_dao.create_message(
            session_id=session_id,
            user_type=user_type,
            message=message,
            stream_id=stream_id,
            stop_stream=stop_stream,
            events_data=events_data,
            assets_data=assets_data
        )

        if success:
            logger.info(f"Queued message creation for session: {session_id}")
        else:
            logger.error(f"Failed to queue message creation for session: {session_id}")

        # Return mock message data for compatibility
        return {
            "id": None,  # Will be assigned by consumer
            "session_id": str(session_id),
            "user_type": user_type,
            "message": message,
            "stream_id": str(stream_id) if stream_id else None,
            "stop_stream": stop_stream,
            "events_data": events_data or {},
            "assets_data": assets_data or {},
            "timestamp": None  # Will be assigned by consumer
        }

    async def update_message_content(
        self,
        message_id: int,
        new_content: str
    ) -> bool:
        """
        Update the content of a message (via queue).

        Args:
            message_id: The message ID
            new_content: New message content

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.update_message_content(
            message_id=message_id,
            new_content=new_content
        )

        if success:
            logger.info(f"Queued message content update: {message_id}")
        else:
            logger.error(f"Failed to queue message content update: {message_id}")

        return success

    async def update_stream_status(
        self,
        message_id: int,
        stop_stream: bool
    ) -> bool:
        """
        Update the streaming status of a message (via queue).

        Args:
            message_id: The message ID
            stop_stream: Whether to stop streaming

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.update_stream_status(
            message_id=message_id,
            stop_stream=stop_stream
        )

        if success:
            logger.info(f"Queued stream status update: {message_id}")
        else:
            logger.error(f"Failed to queue stream status update: {message_id}")

        return success

    async def add_events_data(
        self,
        message_id: int,
        events_data: Dict[str, Any]
    ) -> bool:
        """
        Add or update events data for a message (via queue).

        Args:
            message_id: The message ID
            events_data: Events data to add

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.add_events_data(
            message_id=message_id,
            events_data=events_data
        )

        if success:
            logger.info(f"Queued events data update: {message_id}")
        else:
            logger.error(f"Failed to queue events data update: {message_id}")

        return success

    async def add_assets_data(
        self,
        message_id: int,
        assets_data: Dict[str, Any]
    ) -> bool:
        """
        Add or update assets data for a message (via queue).

        Args:
            message_id: The message ID
            assets_data: Assets data to add

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.add_assets_data(
            message_id=message_id,
            assets_data=assets_data
        )

        if success:
            logger.info(f"Queued assets data update: {message_id}")
        else:
            logger.error(f"Failed to queue assets data update: {message_id}")

        return success

    async def delete_session_messages(
        self,
        session_id: Union[str, uuid.UUID]
    ) -> bool:
        """
        Delete all messages for a session (via queue).

        Args:
            session_id: The session UUID

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.delete_session_messages(session_id=session_id)

        if success:
            logger.info(f"Queued session messages deletion: {session_id}")
        else:
            logger.error(f"Failed to queue session messages deletion: {session_id}")

        return success