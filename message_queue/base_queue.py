"""
Base queue operations class.

This module provides base functionality for queue-based database operations,
replacing the traditional DAO pattern with message queuing.
"""

import logging
from typing import Dict, Any, Optional

from .manager import publish_queue_message

logger = logging.getLogger(__name__)


class BaseQueueDAO:
    """
    Base class for queue-based data access operations.

    Replaces traditional database operations with RabbitMQ message publishing.
    """

    def __init__(self, app_name: str, model_name: str):
        """
        Initialize the queue DAO.

        Args:
            model_name: Database table name
        """
        self.model_name = model_name
        self.app_name = app_name

    def _publish_operation(
        self,
        routing_key: str,
        operation: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a database operation to the queue.

        Args:
            routing_key: RabbitMQ routing key
            operation: Database operation (CREATE/UPDATE/DELETE)
            payload: Operation data
            metadata: Additional metadata

        Returns:
            True if message was published successfully, False otherwise
        """
        try:
            success = publish_queue_message(
                routing_key=routing_key,
                operation=operation,
                model_name={"app_name": self.app_name, "model_name": self.model_name},
                payload=payload,
                metadata=metadata,
            )

            if success:
                logger.info(f"Queued {operation} operation for {self.model_name}")
            else:
                logger.error(
                    f"Failed to queue {operation} operation for {self.model_name}"
                )

            return success

        except Exception as e:
            logger.error(
                f"Error publishing {operation} operation for {self.model_name}: {e}"
            )
            return False

    def create_operation(
        self,
        routing_key: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a CREATE operation.

        Args:
            routing_key: RabbitMQ routing key
            data: Data for the new record
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self._publish_operation(
            routing_key=routing_key,
            operation="CREATE",
            payload={"data": data},
            metadata=metadata,
        )

    def update_operation(
        self,
        routing_key: str,
        identifier: Dict[str, Any],
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue an UPDATE operation.

        Args:
            routing_key: RabbitMQ routing key
            identifier: Fields to identify the record(s) to update
            data: Data to update
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self._publish_operation(
            routing_key=routing_key,
            operation="UPDATE",
            payload={"identifier": identifier, "data": data},
            metadata=metadata,
        )

    def delete_operation(
        self,
        routing_key: str,
        identifier: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue a DELETE operation.

        Args:
            routing_key: RabbitMQ routing key
            identifier: Fields to identify the record(s) to delete
            metadata: Additional metadata

        Returns:
            True if operation was queued successfully, False otherwise
        """
        return self._publish_operation(
            routing_key=routing_key,
            operation="DELETE",
            payload={"identifier": identifier},
            metadata=metadata,
        )

    def batch_operation(
        self,
        operations: list[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue multiple operations as a batch.

        Args:
            operations: List of operations, each containing:
                       - routing_key: str
                       - operation: str (CREATE/UPDATE/DELETE)
                       - payload: Dict[str, Any]
            metadata: Additional metadata

        Returns:
            True if all operations were queued successfully, False otherwise
        """
        success_count = 0
        total_operations = len(operations)

        for op in operations:
            success = self._publish_operation(
                routing_key=op["routing_key"],
                operation=op["operation"],
                payload=op["payload"],
                metadata=metadata,
            )
            if success:
                success_count += 1

        if success_count == total_operations:
            logger.info(
                f"Successfully queued {total_operations} batch operations for {self.model_name}"
            )
            return True
        else:
            logger.warning(
                f"Only {success_count}/{total_operations} batch operations queued for {self.model_name}"
            )
            return False

    def get_model_name(self) -> str:
        """
        Get the model name for this DAO.

        Returns:
            Database table name
        """
        return self.model_name
