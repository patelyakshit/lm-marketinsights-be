"""
Queue-based ChatHistory DAO replacement.

This module provides the same interface as ChatHistoryDAO but uses
RabbitMQ message queuing instead of direct database operations.
"""

import uuid
import logging
from typing import Dict, Any, Optional, Union

from message_queue.chat_queue import ChatHistoryQueueDAO

logger = logging.getLogger(__name__)


class ChatHistoryDAO:
    """
    Queue-based replacement for ChatHistoryDAO.

    Maintains the same interface as the original DAO but publishes
    operations to RabbitMQ queues instead of executing database operations.
    """

    def __init__(self):
        self.queue_dao = ChatHistoryQueueDAO()

    async def create_session(
        self,
        user_id: Optional[int] = None,
        session_title: Optional[str] = None,
        session_type: str = "CHATBOT",
        widget_id: Optional[int] = None,
        widget_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new chat session (via queue).

        Args:
            user_id: ID of the user (from other project)
            session_title: Title for the session
            session_type: Type of session
            widget_id: ID of the widget (from other project)
            widget_history: Whether this is widget history

        Returns:
            Mock ChatHistory instance data
        """
        session_id = uuid.uuid4()

        success = self.queue_dao.create_session(
            session_id=session_id,
            user_id=user_id,
            session_title=session_title,
            session_type=session_type,
            widget_id=widget_id,
            widget_history=widget_history
        )

        if success:
            logger.info(f"Queued session creation: {session_id}")
        else:
            logger.error(f"Failed to queue session creation: {session_id}")

        # Return mock session data for compatibility
        return {
            "session_id": session_id,
            "user_id": user_id,
            "session_title": session_title,
            "session_type": session_type,
            "widget_id": widget_id,
            "widget_history": widget_history,
            "chat": [],
            "chat_summary": "",
            "weather_data": []
        }

    async def get_or_create_session(
        self,
        session_id: Union[str, uuid.UUID],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], bool]:
        """
        Get an existing session or create a new one (via queue).

        Args:
            session_id: The session UUID
            defaults: Default values to use when creating a new session

        Returns:
            Tuple of (session data dict, created boolean)
        """
        session_data, created = self.queue_dao.get_or_create_session(
            session_id=session_id,
            defaults=defaults
        )

        logger.info(f"Queued get_or_create_session: {session_id} (created: {created})")
        return session_data, created

    async def update_chat_data(
        self,
        session_id: Union[str, uuid.UUID],
        chat_data: list
    ) -> bool:
        """
        Update the chat data for a session (via queue).

        Args:
            session_id: The session UUID
            chat_data: List of chat messages/data

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.update_chat_data(
            session_id=session_id,
            chat_data=chat_data
        )

        if success:
            logger.info(f"Queued chat data update: {session_id}")
        else:
            logger.error(f"Failed to queue chat data update: {session_id}")

        return success

    async def update_session_title(
        self,
        session_id: Union[str, uuid.UUID],
        title: str
    ) -> bool:
        """
        Update the session title (via queue).

        Args:
            session_id: The session UUID
            title: New title for the session

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.update_session_title(
            session_id=session_id,
            title=title
        )

        if success:
            logger.info(f"Queued session title update: {session_id}")
        else:
            logger.error(f"Failed to queue session title update: {session_id}")

        return success

    async def update_chat_summary(
        self,
        session_id: Union[str, uuid.UUID],
        summary: str
    ) -> bool:
        """
        Update the chat summary (via queue).

        Args:
            session_id: The session UUID
            summary: Summary of the chat session

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.update_chat_summary(
            session_id=session_id,
            summary=summary
        )

        if success:
            logger.info(f"Queued chat summary update: {session_id}")
        else:
            logger.error(f"Failed to queue chat summary update: {session_id}")

        return success

    async def delete_session(
        self,
        session_id: Union[str, uuid.UUID]
    ) -> bool:
        """
        Delete a chat session (via queue).

        Args:
            session_id: The session UUID to delete

        Returns:
            True if operation was queued successfully, False otherwise
        """
        success = self.queue_dao.delete_session(session_id=session_id)

        if success:
            logger.info(f"Queued session deletion: {session_id}")
        else:
            logger.error(f"Failed to queue session deletion: {session_id}")

        return success