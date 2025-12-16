"""
RabbitMQ Queue Manager.

This module manages RabbitMQ connections, queue declarations, and message publishing.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel

from config.config import (
    CONNECTION_PARAMS,
    DATABASE_OPERATIONS_EXCHANGE,
    EXCHANGE_TYPE,
    QUEUE_CONFIG,
    MESSAGE_PROPERTIES,
    APP_SETTINGS,
)

logger = logging.getLogger(__name__)

# Queue to Celery task name mapping for Django application
QUEUE_TO_TASK_MAP = {
    "chat_history_operations": "chatbot_app.tasks.consume_chat_history_operations",
    "chat_input_output_operations": "chatbot_app.tasks.consume_chat_input_output_operations",
    "dlq_operations": "chatbot_app.tasks.consume_dlq_operations",
}


class QueueManager:
    """
    RabbitMQ connection and queue management.

    Handles connection lifecycle, queue declarations, and message publishing.
    """

    def __init__(self):
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[BlockingChannel] = None
        self._is_connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3

    def connect(self) -> None:
        """
        Establish connection to RabbitMQ.

        Raises:
            Exception: If connection fails
        """
        try:
            # Create connection parameters
            connection_kwargs = {
                "host": CONNECTION_PARAMS["host"],
                "port": CONNECTION_PARAMS["port"],
                "virtual_host": CONNECTION_PARAMS["virtual_host"],
                "connection_attempts": CONNECTION_PARAMS["connection_attempts"],
                "retry_delay": CONNECTION_PARAMS["retry_delay"],
                "heartbeat": CONNECTION_PARAMS["heartbeat"],
                "blocked_connection_timeout": CONNECTION_PARAMS[
                    "blocked_connection_timeout"
                ],
                "socket_timeout": CONNECTION_PARAMS.get("socket_timeout", 10),
            }

            # Only add credentials if both username and password are provided
            if (
                CONNECTION_PARAMS["credentials"]["username"]
                and CONNECTION_PARAMS["credentials"]["password"]
            ):
                connection_kwargs["credentials"] = pika.PlainCredentials(
                    CONNECTION_PARAMS["credentials"]["username"],
                    CONNECTION_PARAMS["credentials"]["password"],
                )

            parameters = pika.ConnectionParameters(**connection_kwargs)

            # Establish connection
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self._is_connected = True

            logger.info("Successfully connected to RabbitMQ")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._is_connected = False
            raise

    def disconnect(self) -> None:
        """
        Close RabbitMQ connection.
        """
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()

            if self.connection and not self.connection.is_closed:
                self.connection.close()

            self._is_connected = False
            self._reconnect_attempts = 0
            logger.info("Disconnected from RabbitMQ")

        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    def _ensure_connection(self) -> bool:
        """
        Ensure connection and channel are open, reconnect if needed.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Check if connection and channel are open
            if (
                self._is_connected
                and self.connection
                and not self.connection.is_closed
                and self.channel
                and self.channel.is_open
            ):
                # Connection is healthy
                return True

            # Connection or channel is closed, attempt reconnection
            logger.warning(
                "RabbitMQ connection or channel is closed. Attempting reconnection..."
            )

            # Close existing resources
            try:
                if self.channel:
                    self.channel.close()
                if self.connection:
                    self.connection.close()
            except Exception:
                pass  # Ignore errors when closing already-closed resources

            # Reset state
            self._is_connected = False

            # Attempt reconnection
            if self._reconnect_attempts < self._max_reconnect_attempts:
                self._reconnect_attempts += 1
                logger.info(
                    f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}"
                )

                # Reconnect
                self.connect()
                self.setup_queues()

                logger.info("Successfully reconnected to RabbitMQ")
                self._reconnect_attempts = 0  # Reset on success
                return True
            else:
                logger.error(
                    f"Failed to reconnect after {self._max_reconnect_attempts} attempts"
                )
                return False

        except Exception as e:
            logger.error(f"Error ensuring connection: {e}")
            return False

    def setup_queues(self) -> None:
        """
        Declare exchange, queues, and bindings.

        Raises:
            Exception: If queue setup fails
        """
        if not self._is_connected or not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            # Declare exchange
            self.channel.exchange_declare(
                exchange=DATABASE_OPERATIONS_EXCHANGE,
                exchange_type=EXCHANGE_TYPE,
                durable=True,
            )

            # Declare queues and bindings
            for queue_name, config in QUEUE_CONFIG.items():
                # Declare queue
                self.channel.queue_declare(
                    queue=queue_name,
                    durable=config["durable"],
                    exclusive=config["exclusive"],
                    auto_delete=config["auto_delete"],
                )

                # Bind queue to exchange with routing keys
                for routing_key in config["routing_keys"]:
                    self.channel.queue_bind(
                        exchange=DATABASE_OPERATIONS_EXCHANGE,
                        queue=queue_name,
                        routing_key=routing_key,
                    )

            logger.info("Successfully set up queues and bindings")

        except Exception as e:
            logger.error(f"Failed to set up queues: {e}")
            raise

    def publish_message(
        self,
        routing_key: str,
        operation: str,
        model_name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a message to the queue with automatic reconnection and retry.

        Args:
            routing_key: RabbitMQ routing key
            operation: Database operation (CREATE/UPDATE/DELETE)
            model_name: Database table name
            payload: Operation data
            metadata: Additional metadata

        Returns:
            True if message was published successfully, False otherwise
        """
        # Try publishing with one retry on failure
        for attempt in range(2):
            try:
                # Ensure connection is healthy (auto-reconnect if needed)
                if not self._ensure_connection():
                    logger.error(
                        f"Cannot publish message: failed to establish connection (attempt {attempt + 1}/2)"
                    )
                    if attempt == 0:
                        continue  # Retry once
                    return False

                # Build message
                message = self._build_message(
                    routing_key, operation, model_name, payload, metadata
                )

                # Publish message
                self.channel.basic_publish(
                    exchange=DATABASE_OPERATIONS_EXCHANGE,
                    routing_key=routing_key,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(**MESSAGE_PROPERTIES),
                )

                logger.info(
                    f"Published message: {routing_key} - {operation} {model_name}"
                )
                return True

            except pika.exceptions.AMQPChannelError as e:
                logger.error(
                    f"Channel error publishing message (attempt {attempt + 1}/2): {e}"
                )
                self._is_connected = False  # Mark as disconnected
                if attempt == 0:
                    continue  # Retry once
                return False

            except pika.exceptions.AMQPConnectionError as e:
                logger.error(
                    f"Connection error publishing message (attempt {attempt + 1}/2): {e}"
                )
                self._is_connected = False  # Mark as disconnected
                if attempt == 0:
                    continue  # Retry once
                return False

            except Exception as e:
                logger.error(
                    f"Failed to publish message (attempt {attempt + 1}/2): {e}"
                )
                if attempt == 0:
                    continue  # Retry once
                return False

        return False

    def _get_task_name_from_routing_key(self, routing_key: str) -> str:
        """
        Map routing key to appropriate Celery task name.

        Args:
            routing_key: RabbitMQ routing key

        Returns:
            Celery task name for Django consumer
        """
        # Map routing keys to queues based on existing patterns
        if routing_key.startswith("chat.history"):
            queue_name = "chat_history_operations"
        elif routing_key.startswith("chat.message"):
            queue_name = "chat_input_output_operations"
        else:
            # Default to DLQ for unknown routing keys
            queue_name = "dlq_operations"

        # Map queue to task name
        return QUEUE_TO_TASK_MAP.get(
            queue_name, "chatbot_app.tasks.consume_dlq_operations"
        )

    def _build_message(
        self,
        routing_key: str,
        operation: str,
        model_name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build Celery task message format for Django consumer.

        Args:
            routing_key: RabbitMQ routing key
            operation: Database operation
            model_name: Database table name
            payload: Operation data
            metadata: Additional metadata

        Returns:
            Celery-formatted message dictionary
        """
        # Get correct task name based on routing key
        task_name = self._get_task_name_from_routing_key(routing_key)

        # Build metadata with existing logic
        full_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4()),
            "source": APP_SETTINGS["source"],
            "version": APP_SETTINGS["version"],
            "environment": APP_SETTINGS["environment"],
        }

        # Add custom metadata if provided
        if metadata:
            full_metadata.update(metadata)

        # Return Celery task format
        return {
            "task": task_name,
            "id": str(uuid.uuid4()),
            "kwargs": {
                "operation": operation,
                "model_name": model_name,
                "payload": payload,
                "metadata": full_metadata,
            },
            "retries": 0,
            "eta": datetime.now(timezone.utc).isoformat(),
        }

    def is_connected(self) -> bool:
        """
        Check if connected to RabbitMQ with healthy channel.

        Returns:
            True if connection and channel are healthy, False otherwise
        """
        return (
            self._is_connected
            and self.connection
            and not self.connection.is_closed
            and self.channel
            and self.channel.is_open
        )

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        self.setup_queues()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Global queue manager instance
queue_manager = QueueManager()


def init_queue_manager() -> None:
    """
    Initialize the global queue manager.

    Raises:
        Exception: If initialization fails
    """
    try:
        queue_manager.connect()
        queue_manager.setup_queues()
        logger.info("Queue manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize queue manager: {e}")
        raise


def close_queue_manager() -> None:
    """
    Close the global queue manager.
    """
    try:
        queue_manager.disconnect()
        logger.info("Queue manager closed successfully")
    except Exception as e:
        logger.error(f"Error closing queue manager: {e}")


def publish_queue_message(
    routing_key: str,
    operation: str,
    model_name: Dict[str, Any],
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish a message using the global queue manager.

    Args:
        routing_key: RabbitMQ routing key
        operation: Database operation (CREATE/UPDATE/DELETE)
        model_name: Database table name
        payload: Operation data
        metadata: Additional metadata

    Returns:
        True if message was published successfully, False otherwise
    """
    return queue_manager.publish_message(
        routing_key=routing_key,
        operation=operation,
        model_name=model_name,
        payload=payload,
        metadata=metadata,
    )
