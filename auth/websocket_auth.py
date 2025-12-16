"""
WebSocket authentication utilities.

Provides reusable function for authenticating WebSocket connections
against external authentication services.
"""

import logging
from typing import Optional, Dict, Any, Tuple

from fastapi import WebSocket

from auth.microservice_auth import (
    auth_service,
    AuthServiceValidationError,
    AuthServiceConnectionError,
)

logger = logging.getLogger(__name__)


async def authenticate_websocket_connection(
    websocket: WebSocket,
    require_auth: bool = True,
    allowed_token_params: Optional[list] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Authenticate WebSocket connection using tokens from query parameters.

    Args:
        websocket: The WebSocket connection
        require_auth: Whether authentication is required (default: True)
        allowed_token_params: List of query parameter names to check for tokens

    Returns:
        Tuple of (is_authenticated: bool, user_info: Optional[Dict])
    """
    if allowed_token_params is None:
        allowed_token_params = ["token", "jwt_token", "drf_token", "access_token"]

    # Extract token from query parameters
    auth_token = None
    token_type = "bearer"

    for param_name in allowed_token_params:
        token_value = websocket.query_params.get(param_name)
        if token_value:
            auth_token = token_value
            # Auto-detect token type: JWT (3 parts) vs DRF token
            if len(token_value.split(".")) == 3:
                token_type = "jwt"
            elif (
                param_name.lower() in ["drf_token", "token"] and len(token_value) == 40
            ):
                # DRF tokens are typically 40 characters
                token_type = "token"
            else:
                token_type = "bearer"  # Default fallback
            break

    # Handle missing token
    if not auth_token:
        if require_auth:
            logger.warning(
                "WebSocket connection rejected: No authentication token provided"
            )
            return False, None
        else:
            return True, None

    # Validate token
    try:
        user_info = await auth_service.validate_token(auth_token, token_type)

        if not user_info:
            if require_auth:
                logger.warning("WebSocket connection rejected: Invalid token")
                return False, None
            else:
                return True, None

        logger.info(
            f"WebSocket authentication successful for user: {user_info.get('user_id', 'unknown')}"
        )
        return True, user_info.get("user", {})

    except AuthServiceValidationError as e:
        if require_auth:
            logger.warning(
                f"WebSocket connection rejected: Token validation failed - {e}"
            )
            return False, None
        else:
            return True, None

    except AuthServiceConnectionError as e:
        logger.error(
            f"WebSocket authentication error: Auth service connection failed - {e}"
        )
        if require_auth:
            return False, None
        else:
            return True, None

    except Exception as e:
        logger.error(f"WebSocket authentication error: Unexpected error - {e}")
        if require_auth:
            return False, None
        else:
            return True, None
