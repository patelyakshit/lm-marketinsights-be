"""
Data Access Object (DAO) package for the bumblebee application.

This package contains queue-based DAO classes that publish database operations
to RabbitMQ instead of executing them directly.

Usage Examples:

    # ChatHistory operations
    from dao.chat_history_queue_dao import ChatHistoryDAO

    chat_dao = ChatHistoryDAO()

    # Create a new session
    session = await chat_dao.create_session(
        user_id=123,
        session_title="Customer Support Chat"
    )

    # Update chat data
    await chat_dao.update_chat_data(session.session_id, chat_messages)

    # ChatInputOutput operations
    from dao.chat_input_output_queue_dao import ChatInputOutputDAO

    message_dao = ChatInputOutputDAO()

    # Create a new message
    message = await message_dao.create_message(
        session_id=session.session_id,
        user_type=UserTypeConstants.HUMAN,
        message="Hello, I need help with my account"
    )
"""

from .chat_history_queue_dao import ChatHistoryDAO
from .chat_input_output_queue_dao import ChatInputOutputDAO

__all__ = [
    # Queue-based DAOs
    "ChatHistoryDAO",
    "ChatInputOutputDAO",
]

# Convenience instances for direct usage
chat_history_dao = ChatHistoryDAO()
chat_input_output_dao = ChatInputOutputDAO()

__all__.extend([
    "chat_history_dao",
    "chat_input_output_dao",
])