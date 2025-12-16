import json
import logging
import traceback
from functools import wraps
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors"""

    def __init__(
        self,
        message: str,
        error_type: str = "AGENT_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


class SalesforceError(AgentError):
    """Exception for Salesforce-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SALESFORCE_ERROR", details)


class LLMError(AgentError):
    """Exception for LLM-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LLM_ERROR", details)


class SessionError(AgentError):
    """Exception for session-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SESSION_ERROR", details)


class WebSocketError(AgentError):
    """Exception for WebSocket-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "WEBSOCKET_ERROR", details)


class GISError(AgentError):
    """Exception for GIS-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "GIS_ERROR", details)


def handle_agent_errors(func):
    """Decorator to handle agent errors gracefully"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AgentError as e:
            logger.error(f"Agent error in {func.__name__}: {e.message}")
            return create_error_response(e.error_type, e.message, e.details)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return create_error_response(
                "INTERNAL_ERROR", f"Unexpected error: {str(e)}"
            )

    return wrapper


def handle_sync_agent_errors(func):
    """Decorator to handle agent errors gracefully for synchronous functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AgentError as e:
            logger.error(f"Agent error in {func.__name__}: {e.message}")
            return create_error_response(e.error_type, e.message, e.details)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return create_error_response(
                "INTERNAL_ERROR", f"Unexpected error: {str(e)}"
            )

    return wrapper


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    connection_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "type": "error",
        "error_type": error_type,
        "message": message,
        "details": details or {},
        "connection_id": connection_id,
        "timestamp": None,
    }


def log_error_to_session(
    session_service,
    session_id: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
):
    """Log error to session service"""
    try:
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()

        if hasattr(error, "error_type"):
            error_type = error.error_type

        session_service.log_error(
            session_id=session_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            severity="ERROR",
        )

        if context:
            logger.info(f"Error context: {json.dumps(context, indent=2)}")

    except Exception as e:
        logger.error(f"Failed to log error to session: {e}")


class ErrorRecovery:
    """Handles error recovery strategies"""

    @staticmethod
    def should_retry(
        error: Exception, attempt_count: int, max_retries: int = 3
    ) -> bool:
        """Determine if an operation should be retried"""
        if attempt_count >= max_retries:
            return False

        # Retry on network errors, rate limits, etc.
        retry_on = [
            "ConnectionError",
            "TimeoutError",
            "RateLimitError",
            "ServiceUnavailable",
        ]

        error_name = type(error).__name__
        return any(retry_type in error_name for retry_type in retry_on)

    @staticmethod
    def get_retry_delay(attempt_count: int) -> float:
        """Calculate exponential backoff delay"""
        base_delay = 1.0
        max_delay = 60.0
        delay = base_delay * (2**attempt_count)
        return min(delay, max_delay)
