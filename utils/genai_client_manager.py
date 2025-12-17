"""
GenAI Client Manager with API Key Support

Provides a singleton manager that creates and caches a single genai.Client instance
shared across all Gemini model instances. Supports both:
- Google AI Studio (API key) - Default, uses personal API key
- Vertex AI (service account) - Enterprise, uses GCP credentials
"""

import logging
import os
import threading
from functools import cached_property
from pathlib import Path
from typing import Optional

from decouple import config
from google import genai

logger = logging.getLogger(__name__)

# Singleton state
_genai_client: Optional[genai.Client] = None
_initialized: bool = False
_lock = threading.Lock()
_using_api_key: bool = False

# API key should be set via GOOGLE_API_KEY environment variable
# Do NOT hardcode API keys - they will be flagged as leaked on GitHub
DEFAULT_API_KEY = None  # Must be set via env var


def initialize_genai_client(
    api_key: Optional[str] = None,
    use_vertex_ai: bool = False,
    credentials_path: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """
    Initialize the singleton genai.Client() instance.

    By default, uses Google AI Studio with personal API key for safety.
    Set use_vertex_ai=True to use Vertex AI with service account credentials.

    Args:
        api_key: Google AI Studio API key (defaults to GOOGLE_API_KEY env var or built-in default)
        use_vertex_ai: If True, use Vertex AI instead of Google AI Studio
        credentials_path: Path to service account JSON (only for Vertex AI)
        project: GCP project ID (only for Vertex AI)
        location: GCP location (only for Vertex AI)

    Returns:
        genai.Client: The initialized singleton client instance

    Raises:
        RuntimeError: If client initialization fails
    """
    global _genai_client, _initialized, _using_api_key

    if _initialized and _genai_client is not None:
        logger.debug("GenAI client already initialized, returning existing instance")
        return _genai_client

    with _lock:
        if _initialized and _genai_client is not None:  # Double-check locking
            return _genai_client

        try:
            # Check if we should use Vertex AI (from env var or parameter)
            use_vertex_ai = use_vertex_ai or config("GOOGLE_GENAI_USE_VERTEXAI", default="1") == "1"

            if use_vertex_ai:
                # Vertex AI mode (service account)
                return _initialize_vertex_ai_client(credentials_path, project, location)
            else:
                # Google AI Studio mode (API key) - DEFAULT
                return _initialize_api_key_client(api_key)

        except Exception as e:
            import traceback
            logger.error("=" * 80)
            logger.error(f"✗ Failed to initialize GenAI client: {e}")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            raise RuntimeError(f"Failed to initialize GenAI client: {e}") from e


def _initialize_api_key_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialize client with Google AI Studio API key."""
    global _genai_client, _initialized, _using_api_key

    # Get API key from parameter or env var (REQUIRED)
    api_key = api_key or config("GOOGLE_API_KEY", default="")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get a key from https://aistudio.google.com/apikey and add to .env file."
        )

    logger.info("=" * 80)
    logger.info("Initializing GenAI Client with Google AI Studio (API Key)")
    logger.info(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
    logger.info(f"  Mode: Google AI Studio (NOT Vertex AI)")
    logger.info("=" * 80)

    # Create client with API key (explicitly disable Vertex AI)
    logger.info("Creating genai.Client() with API key...")
    _genai_client = genai.Client(api_key=api_key, vertexai=False)

    _initialized = True
    _using_api_key = True

    logger.info("✓ GenAI client created with personal API key")
    logger.info("=" * 80)
    logger.info("✓ GenAI Client Manager initialized successfully (API Key mode)")
    logger.info("=" * 80)

    return _genai_client


def _initialize_vertex_ai_client(
    credentials_path: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """Initialize client with Vertex AI service account."""
    global _genai_client, _initialized, _using_api_key

    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    # Use provided values or fall back to env vars/defaults
    credentials_path = credentials_path or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"
    )
    project = project or config(
        "GOOGLE_CLOUD_PROJECT", default="lm-market-insights-ai"
    )
    location = location or config(
        "GOOGLE_CLOUD_LOCATION", default="us-central1"
    )

    logger.info("=" * 80)
    logger.info("Initializing GenAI Client with Vertex AI (Service Account)")
    logger.info(f"  Credentials path: {credentials_path}")
    logger.info(f"  Project: {project}")
    logger.info(f"  Location: {location}")
    logger.info("=" * 80)

    # Validate credentials file
    creds_file = Path(credentials_path)
    if not creds_file.exists():
        raise FileNotFoundError(f"GCP credentials file not found: {credentials_path}")
    if not creds_file.is_file():
        raise ValueError(f"Credentials path is not a file: {credentials_path}")
    if creds_file.stat().st_size == 0:
        raise ValueError(f"Credentials file is empty: {credentials_path}")

    logger.info(f"✓ Credentials file validated ({creds_file.stat().st_size} bytes)")

    # Ensure environment variables are set for ADC
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    logger.info("✓ ADC environment variables configured")

    # Load and cache credentials
    logger.info("Loading service account credentials...")
    cached_credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    logger.info("✓ Service account credentials loaded")

    # Pre-warm token
    if not cached_credentials.valid:
        logger.info("Pre-warming credentials: Fetching initial access token...")
        cached_credentials.refresh(Request())
        logger.info("✓ Initial token fetched and cached")

    # Create shared client
    logger.info("Creating shared genai.Client() instance...")
    _genai_client = genai.Client()

    _initialized = True
    _using_api_key = False

    logger.info("✓ GenAI client created with Vertex AI credentials")
    logger.info("=" * 80)
    logger.info("✓ GenAI Client Manager initialized successfully (Vertex AI mode)")
    logger.info("=" * 80)

    return _genai_client


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


def is_using_api_key() -> bool:
    """Check if client is using API key mode (vs Vertex AI)."""
    return _using_api_key


def create_gemini_with_shared_client(model: str):
    """
    Create a Gemini model instance that uses the shared genai.Client().

    This ensures ADK agents use the singleton client with cached credentials,
    preventing repeated credential checks on every LLM call.

    Args:
        model: Model identifier (e.g., "gemini-2.5-flash-lite")

    Returns:
        Gemini instance configured to use shared client
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
