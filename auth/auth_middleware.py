"""
FastAPI authentication middleware for token validation.

This module provides middleware that validates tokens for all incoming requests
against an external authentication service, with flexible configuration options.
"""

import logging
from typing import Optional, Dict, Any, Set
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from auth.microservice_auth import (
    auth_service,
    extract_token_from_header,
    extract_token_from_query,
    AuthServiceValidationError,
    AuthServiceConnectionError
)

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for token-based authentication.

    Validates tokens against external authentication service for all requests
    except those in the excluded paths list.
    """

    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: Optional[Set[str]] = None,
        optional_auth_paths: Optional[Set[str]] = None,
        auth_failure_response: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize authentication middleware.

        Args:
            app: The ASGI application
            excluded_paths: Set of paths that don't require authentication
            optional_auth_paths: Set of paths where authentication is optional
            auth_failure_response: Custom response for authentication failures
        """
        super().__init__(app)

        # Default excluded paths (public endpoints)
        self.excluded_paths = excluded_paths or {
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        }

        # Paths where authentication is optional
        self.optional_auth_paths = optional_auth_paths or set()

        # Default authentication failure response
        self.auth_failure_response = auth_failure_response or {
            "error": "Authentication required",
            "message": "Valid token required to access this resource",
            "status_code": 401
        }

        logger.info(f"AuthenticationMiddleware initialized with {len(self.excluded_paths)} excluded paths")

    async def dispatch(self, request: Request, call_next):
        """
        Process incoming request and validate authentication.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            HTTP response
        """
        # Extract request path
        path = request.url.path

        # Skip authentication for excluded paths
        if self._is_excluded_path(path):
            logger.debug(f"Skipping authentication for excluded path: {path}")
            return await call_next(request)

        # Check if this is an optional auth path
        optional_auth = path in self.optional_auth_paths

        # Extract token from request
        token = await self._extract_token(request)

        if not token:
            if optional_auth:
                logger.debug(f"No token provided for optional auth path: {path}")
                # Continue without authentication
                request.state.user = None
                request.state.authenticated = False
                return await call_next(request)
            else:
                logger.warning(f"No token provided for protected path: {path}")
                return self._create_auth_failure_response("No authentication token provided")

        # Validate token
        try:
            user_info = await self._validate_token(token, request)

            if user_info:
                # Store user information in request state
                request.state.user = user_info
                request.state.authenticated = True
                request.state.token = token

                logger.info(f"Authentication successful for user: {user_info.get('user_id', 'unknown')} on path: {path}")
                return await call_next(request)
            else:
                if optional_auth:
                    logger.debug(f"Invalid token for optional auth path: {path}")
                    request.state.user = None
                    request.state.authenticated = False
                    return await call_next(request)
                else:
                    logger.warning(f"Token validation failed for path: {path}")
                    return self._create_auth_failure_response("Invalid or expired token")

        except AuthServiceConnectionError as e:
            logger.error(f"Auth service connection error: {e}")
            if optional_auth:
                # Allow access if auth service is down and auth is optional
                request.state.user = None
                request.state.authenticated = False
                return await call_next(request)
            else:
                return self._create_auth_failure_response("Authentication service unavailable", status_code=503)

        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            if optional_auth:
                request.state.user = None
                request.state.authenticated = False
                return await call_next(request)
            else:
                return self._create_auth_failure_response("Authentication error", status_code=500)

    async def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract authentication token from request headers or query parameters.

        Args:
            request: The HTTP request

        Returns:
            Extracted token or None
        """
        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header:
            token = extract_token_from_header(auth_header)
            if token:
                logger.debug("Token extracted from Authorization header")
                return token

        # Try X-Auth-Token header
        auth_token_header = request.headers.get("X-Auth-Token")
        if auth_token_header:
            logger.debug("Token extracted from X-Auth-Token header")
            return auth_token_header

        # Try query parameters
        query_params = dict(request.query_params)
        token = extract_token_from_query(query_params)
        if token:
            logger.debug("Token extracted from query parameters")
            return token

        return None

    async def _validate_token(self, token: str, request: Request) -> Optional[Dict[str, Any]]:
        """
        Validate token against authentication service.

        Args:
            token: The token to validate
            request: The HTTP request (for context)

        Returns:
            User information if valid, None otherwise
        """
        try:
            # Determine token type based on format
            token_type = "jwt" if self._looks_like_jwt(token) else "bearer"

            # Validate token
            user_info = await auth_service.validate_token(token, token_type)
            return user_info

        except AuthServiceValidationError:
            return None

    def _is_excluded_path(self, path: str) -> bool:
        """
        Check if a path should be excluded from authentication.

        Supports exact matches and prefix patterns.

        Args:
            path: The request path to check

        Returns:
            True if path should be excluded from authentication
        """
        # Check exact matches
        if path in self.excluded_paths:
            return True

        # Check prefix patterns (for paths like /media/*)
        for excluded_path in self.excluded_paths:
            if path.startswith(excluded_path + "/"):
                return True

        return False

    def _looks_like_jwt(self, token: str) -> bool:
        """
        Check if token looks like a JWT (has 3 parts separated by dots).

        Args:
            token: The token to check

        Returns:
            True if token looks like JWT
        """
        return len(token.split('.')) == 3

    def _create_auth_failure_response(self, message: str, status_code: int = 401) -> JSONResponse:
        """
        Create standardized authentication failure response.

        Args:
            message: Error message
            status_code: HTTP status code

        Returns:
            JSON response with error details
        """
        response_data = {
            "error": "Authentication failed",
            "message": message,
            "status_code": status_code
        }

        return JSONResponse(
            status_code=status_code,
            content=response_data
        )

    def add_excluded_path(self, path: str) -> None:
        """
        Add a path to the excluded paths set.

        Args:
            path: Path to exclude from authentication
        """
        self.excluded_paths.add(path)
        logger.info(f"Added excluded path: {path}")

    def add_optional_auth_path(self, path: str) -> None:
        """
        Add a path to the optional authentication paths set.

        Args:
            path: Path where authentication is optional
        """
        self.optional_auth_paths.add(path)
        logger.info(f"Added optional auth path: {path}")

    def remove_excluded_path(self, path: str) -> None:
        """
        Remove a path from the excluded paths set.

        Args:
            path: Path to remove from exclusion list
        """
        self.excluded_paths.discard(path)
        logger.info(f"Removed excluded path: {path}")


