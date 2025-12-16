"""
Queue package for RabbitMQ-based database operations.

This package provides message queuing functionality to replace direct
database operations with asynchronous message publishing.
"""

from config.config import (
    RoutingKeys,
    CHAT_HISTORY_QUEUE,
    CHAT_INPUT_OUTPUT_QUEUE,
    DATABASE_OPERATIONS_EXCHANGE,
)
from .base_queue import BaseQueueDAO
from .chat_queue import ChatHistoryQueueDAO, ChatInputOutputQueueDAO
from .manager import (
    QueueManager,
    init_queue_manager,
    close_queue_manager,
    publish_queue_message,
    queue_manager,
)

__all__ = [
    # Manager
    "QueueManager",
    "init_queue_manager",
    "close_queue_manager",
    "publish_queue_message",
    "queue_manager",
    # Configuration
    "RoutingKeys",
    "CHAT_HISTORY_QUEUE",
    "CHAT_INPUT_OUTPUT_QUEUE",
    "DATABASE_OPERATIONS_EXCHANGE",
    # Queue DAOs
    "ChatHistoryQueueDAO",
    "ChatInputOutputQueueDAO",
    "BaseQueueDAO",
]
