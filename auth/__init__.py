"""
Authentication package for external microservice authentication.

This package provides authentication middleware and service client
for validating tokens against external authentication services.
"""

from .auth_middleware import AuthenticationMiddleware
from .microservice_auth import (
    AuthService,
    AuthServiceError,
    AuthServiceConnectionError,
    AuthServiceValidationError,
    auth_service,
    extract_token_from_header,
    extract_token_from_query,
)
from .websocket_auth import authenticate_websocket_connection

__all__ = [
    # Service client
    "AuthService",
    "AuthServiceError",
    "AuthServiceConnectionError",
    "AuthServiceValidationError",
    "auth_service",
    "extract_token_from_header",
    "extract_token_from_query",
    # Middleware
    "AuthenticationMiddleware",
    # WebSocket Authentication
    "authenticate_websocket_connection",
]
