"""
Authentication service client for external microservice authentication.

This module provides an async client for validating tokens against an external
authentication service with configurable endpoints and retry logic.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

import httpx
from decouple import config

logger = logging.getLogger(__name__)


class AuthServiceError(Exception):
    """Base exception for authentication service errors."""

    pass


class AuthServiceConnectionError(AuthServiceError):
    """Raised when unable to connect to authentication service."""

    pass


class AuthServiceValidationError(AuthServiceError):
    """Raised when token validation fails."""

    pass


class AuthService:
    """
    Async client for external authentication service.

    Provides token validation against a configurable authentication microservice
    with retry logic and timeout handling.
    """

    def __init__(self):
        """Initialize AuthService with environment configuration."""
        self.auth_service_url = config(
            "AUTH_SERVICE_URL", default="http://localhost:8001"
        )
        self.timeout = config("AUTH_SERVICE_TIMEOUT", default=5, cast=int)
        self.max_retries = config("AUTH_SERVICE_MAX_RETRIES", default=3, cast=int)
        self.api_key = config(
            "AUTH_SERVICE_API_KEY_1", default="&zwP9Gm*fs$6BHboq69xL7GWH85jEhx4"
        )

        # Ensure URL doesn't end with slash for consistent endpoint construction
        self.auth_service_url = self.auth_service_url.rstrip("/")

        # HTTP client configuration
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }

        logger.info(f"AuthService initialized with endpoint: {self.auth_service_url}")

    async def validate_token(
        self, token: str, token_type: str = "bearer"
    ) -> Dict[str, Any]:
        """
        Validate a token against the external authentication service.

        Args:
            token: The token to validate (JWT or DRF token)
            token_type: Type of token ("bearer"/"jwt" for JWT, "token" for DRF)

        Returns:
            Dict containing user information and token validation status

        Raises:
            AuthServiceConnectionError: When unable to connect to auth service
            AuthServiceValidationError: When token validation fails
        """
        if not token:
            raise AuthServiceValidationError("Token is required")

        # Prepare headers based on token type
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Bumblebee-AuthClient/1.0",
        }

        # Add Authorization header based on token type
        if token_type.lower() in ["jwt", "bearer"]:
            headers["Authorization"] = f"Bearer {token}"
        elif token_type.lower() == "token":
            headers["Authorization"] = f"Token {token}"
        else:
            headers["Authorization"] = f"Bearer {token}"  # Default to Bearer

        # Add X-API-Key header if configured
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        # Attempt validation with retry logic
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(**self.client_config) as client:
                    response = await client.post(self.auth_service_url, headers=headers)

                    # Handle response
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("valid", False):
                            logger.info(
                                f"Token validation successful for user: {result.get('user_id', 'unknown')}"
                            )
                            return result
                        else:
                            raise AuthServiceValidationError(
                                f"Token validation failed: {result.get('message', 'Invalid token')}"
                            )

                    elif response.status_code == 401:
                        raise AuthServiceValidationError("Invalid or expired token")

                    elif response.status_code == 403:
                        raise AuthServiceValidationError("Token validation forbidden")

                    else:
                        raise AuthServiceConnectionError(
                            f"Auth service returned {response.status_code}: {response.text}"
                        )

            except httpx.TimeoutException:
                last_exception = AuthServiceConnectionError(
                    f"Timeout connecting to auth service (attempt {attempt + 1})"
                )
                logger.warning(f"Auth service timeout on attempt {attempt + 1}")

            except httpx.ConnectError:
                last_exception = AuthServiceConnectionError(
                    f"Unable to connect to auth service (attempt {attempt + 1})"
                )
                logger.warning(
                    f"Auth service connection error on attempt {attempt + 1}"
                )

            except AuthServiceValidationError:
                raise

            except Exception as e:
                last_exception = AuthServiceError(
                    f"Unexpected error during token validation: {str(e)}"
                )
                logger.error(
                    f"Unexpected auth service error on attempt {attempt + 1}: {e}"
                )

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2**attempt
                logger.info(f"Retrying auth service call in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise AuthServiceConnectionError("Max retries exceeded")

    async def validate_jwt_token(self, jwt_token: str) -> Dict[str, Any]:
        """
        Validate a JWT token.

        Args:
            jwt_token: The JWT token to validate

        Returns:
            Dict containing user information and token validation status
        """
        return await self.validate_token(jwt_token, "jwt")

    async def validate_drf_token(self, drf_token: str) -> Dict[str, Any]:
        """
        Validate a Django REST Framework token.

        Args:
            drf_token: The DRF token to validate

        Returns:
            Dict containing user information and token validation status
        """
        return await self.validate_token(drf_token, "token")


# Global auth service instance
auth_service = AuthService()


def extract_token_from_header(authorization_header: str) -> Optional[str]:
    """
    Extract token from Authorization header.

    Args:
        authorization_header: The Authorization header value

    Returns:
        Extracted token or None if invalid format
    """
    if not authorization_header:
        return None

    parts = authorization_header.split()
    if len(parts) != 2:
        return None

    scheme, token = parts
    if scheme.lower() not in ["bearer", "token"]:
        return None

    return token


def extract_token_from_query(query_params: Dict[str, str]) -> Optional[str]:
    """
    Extract token from query parameters.

    Args:
        query_params: Dictionary of query parameters

    Returns:
        Extracted token or None if not found
    """
    # Try different token parameter names
    for param_name in ["token", "jwt_token", "access_token"]:
        if param_name in query_params:
            return query_params[param_name]

    return None
