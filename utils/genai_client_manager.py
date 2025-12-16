"""
GenAI Client Manager with Shared Credential Caching

Provides a singleton manager that creates and caches a single genai.Client instance
shared across all Gemini model instances, eliminating repeated token API calls
and credential checks on every LLM call.
"""

import logging
import os
import threading
from functools import cached_property
from pathlib import Path
from typing import Optional

from decouple import config
from google import genai
from google.auth.transport.requests import Request
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Singleton state
_genai_client: Optional[genai.Client] = None
_cached_credentials: Optional[service_account.Credentials] = None
_initialized: bool = False
_lock = threading.Lock()


def initialize_genai_client(
    credentials_path: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """
    Initialize the singleton genai.Client() instance with shared credentials.
    
    This should be called once during application startup, after GCP credentials
    have been verified and GOOGLE_APPLICATION_CREDENTIALS environment variable
    has been set.
    
    By explicitly loading and caching credentials, we ensure:
    1. OAuth token is fetched once during startup
    2. Token is cached and reused across all clients
    3. No repeated credential checks on every LLM call
    
    Args:
        credentials_path: Path to service account JSON (defaults to GOOGLE_APPLICATION_CREDENTIALS env var)
        project: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        location: GCP location (defaults to GOOGLE_CLOUD_LOCATION env var)
    
    Returns:
        genai.Client: The initialized singleton client instance
        
    Raises:
        RuntimeError: If client initialization fails
    """
    global _genai_client, _cached_credentials, _initialized
    
    if _initialized and _genai_client is not None:
        logger.debug("GenAI client already initialized, returning existing instance")
        return _genai_client
    
    with _lock:
        if _initialized and _genai_client is not None:  # Double-check locking
            return _genai_client
        
        try:
            # Use provided values or fall back to env vars/defaults
            credentials_path = credentials_path or os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS", "/tmp/credentials.json"
            )
            project = project or config(
                "GOOGLE_CLOUD_PROJECT", default="lm-market-insights-ai"
            )
            location = location or config(
                "GOOGLE_CLOUD_LOCATION", default="us-central1"
            )
            
            logger.info("=" * 80)
            logger.info("Initializing GenAI Client Manager with shared credentials")
            logger.info(f"  Credentials path: {credentials_path}")
            logger.info(f"  Project: {project}")
            logger.info(f"  Location: {location}")
            logger.info("=" * 80)
            
            # Validate credentials file
            creds_file = Path(credentials_path)
            if not creds_file.exists():
                raise FileNotFoundError(
                    f"GCP credentials file not found: {credentials_path}"
                )
            if not creds_file.is_file():
                raise ValueError(f"Credentials path is not a file: {credentials_path}")
            if creds_file.stat().st_size == 0:
                raise ValueError(f"Credentials file is empty: {credentials_path}")
            
            logger.info(
                f"✓ Credentials file validated ({creds_file.stat().st_size} bytes)"
            )
            
            # Ensure environment variables are set for ADC
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
            
            logger.info("✓ ADC environment variables configured")
            
            # Load service account credentials explicitly
            logger.info("Loading service account credentials...")
            _cached_credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info("✓ Service account credentials loaded")
            
            # Pre-warm: Force initial token fetch to populate cache
            if not _cached_credentials.valid:
                logger.info("Pre-warming credentials: Fetching initial access token...")
                _cached_credentials.refresh(Request())
                logger.info("✓ Initial token fetched and cached")
                logger.debug(f"  Token expiry: {_cached_credentials.expiry}")
            else:
                logger.info("✓ Existing token is still valid")
            
            # Create shared client - this will use the cached credentials
            # The genai.Client() will use ADC which will find our cached token
            logger.info("Creating shared genai.Client() instance...")
            _genai_client = genai.Client()
            logger.info("✓ GenAI client created with shared credentials")
            
            _initialized = True
            logger.info("=" * 80)
            logger.info("✓ GenAI Client Manager initialized successfully")
            logger.info("=" * 80)
            
            return _genai_client
            
        except Exception as e:
            import traceback
            logger.error("=" * 80)
            logger.error(f"✗ Failed to initialize GenAI client: {e}")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            raise RuntimeError(f"Failed to initialize GenAI client: {e}") from e


def get_genai_client() -> genai.Client:
    """
    Get the singleton genai.Client() instance.
    
    Returns:
        genai.Client: The singleton client instance
        
    Raises:
        RuntimeError: If the client has not been initialized yet
    """
    if _genai_client is None:
        raise RuntimeError(
            "GenAI client not initialized. "
            "Call initialize_genai_client() during application startup first."
        )
    return _genai_client


def get_cached_credentials() -> Optional[service_account.Credentials]:
    """
    Get the cached service account credentials.
    
    Returns:
        Cached Credentials instance, or None if not initialized
    """
    return _cached_credentials


def create_gemini_with_shared_client(model: str):
    """
    Create a Gemini model instance that uses the shared genai.Client().
    
    This ensures ADK agents use the singleton client with cached credentials,
    preventing repeated credential checks on every LLM call.
    
    Args:
        model: Model identifier (e.g., "gemini-2.5-flash-lite")
    
    Returns:
        Gemini instance configured to use shared client
    
    Usage:
        from utils.genai_client_manager import create_gemini_with_shared_client
        gemini_model = create_gemini_with_shared_client("gemini-2.5-flash-lite")
        agent = LlmAgent(model=gemini_model, ...)
    """
    from google.adk.models import Gemini
    
    # Ensure client is initialized
    if not _initialized:
        raise RuntimeError(
            "GenAI client not initialized. "
            "Call initialize_genai_client() during application startup first."
        )
    
    # Create a dynamic subclass that uses the shared client
    class _GeminiWithSharedClient(Gemini):
        """Gemini subclass that uses shared genai.Client() instance."""
        
        @cached_property
        def api_client(self) -> genai.Client:
            """Override to use shared singleton client."""
            return get_genai_client()
        
        @cached_property
        def _live_api_client(self) -> genai.Client:
            """Override live API client to use shared singleton client."""
            return get_genai_client()
    
    # Create and return instance
    gemini_instance = _GeminiWithSharedClient(model=model)
    logger.debug(f"Created GeminiWithSharedClient for model: {model}")
    return gemini_instance

