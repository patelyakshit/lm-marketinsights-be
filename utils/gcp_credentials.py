"""
GCP Credentials Management Utility

Automatically checks for and downloads GCP service account credentials
from Azure Blob Storage if not present locally.

Supports multiple credential sources:
1. GOOGLE_APPLICATION_CREDENTIALS_JSON env var (for Railway/Heroku - JSON as string)
2. GOOGLE_APPLICATION_CREDENTIALS file path (for local/Docker)
3. Azure Blob Storage download (fallback)
"""

import json
import logging
import os
from pathlib import Path

import httpx
from decouple import config

logger = logging.getLogger(__name__)

# Credentials directory and file path
# Using /tmp for container environments (Docker/K8s/Railway)
CREDENTIALS_DIR = Path("/tmp")
CREDENTIALS_FILE = CREDENTIALS_DIR / "gcp_credentials.json"


def create_credentials_from_env() -> bool:
    """
    Create credentials file from GOOGLE_APPLICATION_CREDENTIALS_JSON env var.
    This is used for Railway/Heroku where you can't mount files.

    Returns:
        bool: True if credentials were created from env var, False otherwise
    """
    json_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_creds:
        return False

    try:
        # Validate it's valid JSON
        creds_dict = json.loads(json_creds)

        # Ensure directory exists
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

        # Write to file
        CREDENTIALS_FILE.write_text(json.dumps(creds_dict, indent=2))
        CREDENTIALS_FILE.chmod(0o600)

        # Set the env var to point to this file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)

        logger.info(f"✓ Created GCP credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON env var")
        logger.info(f"  File: {CREDENTIALS_FILE}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error creating credentials from env var: {e}")
        return False


# Check for existing GOOGLE_APPLICATION_CREDENTIALS path
env_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if env_creds_path and Path(env_creds_path).exists():
    CREDENTIALS_FILE = Path(env_creds_path)
    CREDENTIALS_DIR = CREDENTIALS_FILE.parent

# Azure Blob Storage URL for GCP credentials
AZURE_BLOB_URL = config(
    "AZURE_GCP_SERVICE_ACCOUNT",
    default="https://lmproductwigetsdev.blob.core.windows.net/lmwidget-agent-dev/gcp-cred/Location Matters.json",
)


async def check_or_download_credentials_file() -> bool:
    """
    Check if credentials.json exists, create from env var, or download from Azure.

    Priority order:
    1. GOOGLE_APPLICATION_CREDENTIALS_JSON env var (Railway/Heroku)
    2. Existing file at GOOGLE_APPLICATION_CREDENTIALS path
    3. Download from Azure Blob Storage

    Returns:
        bool: True if file exists or was created/downloaded, False otherwise
    """
    global CREDENTIALS_FILE, CREDENTIALS_DIR

    try:
        # First, try to create from env var (Railway/Heroku deployment)
        if create_credentials_from_env():
            CREDENTIALS_FILE = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            CREDENTIALS_DIR = CREDENTIALS_FILE.parent
            return True

        # Ensure credentials directory exists and is writable
        if not CREDENTIALS_DIR.exists():
            logger.info(f"Creating credentials directory: {CREDENTIALS_DIR}")
            CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

        # Verify directory is writable
        if not os.access(CREDENTIALS_DIR, os.W_OK):
            logger.error(f"Credentials directory is not writable: {CREDENTIALS_DIR}")
            return False

        # Check if credentials file already exists
        if CREDENTIALS_FILE.exists():
            # Validate it's a non-empty file
            if CREDENTIALS_FILE.stat().st_size > 0:
                logger.info(
                    f"✓ GCP credentials file found: {CREDENTIALS_FILE} ({CREDENTIALS_FILE.stat().st_size} bytes)"
                )
                return True
            else:
                logger.warning(f"Existing credentials file is empty, will re-download")
                CREDENTIALS_FILE.unlink()

        logger.info(
            f"GCP credentials file not found, downloading from Azure Blob Storage..."
        )
        logger.info(f"Source URL: {AZURE_BLOB_URL[:80]}...")

        # Download credentials from Azure Blob Storage
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(AZURE_BLOB_URL)
            response.raise_for_status()

            # Validate response has content
            if not response.content or len(response.content) == 0:
                logger.error("Downloaded credentials file is empty")
                return False

            # Write credentials to file with proper permissions
            CREDENTIALS_FILE.write_bytes(response.content)

            # Set file permissions to be readable only by owner (security best practice)
            CREDENTIALS_FILE.chmod(0o600)

            logger.info(
                f"✓ Successfully downloaded GCP credentials to: {CREDENTIALS_FILE}"
            )
            logger.info(f"  File size: {CREDENTIALS_FILE.stat().st_size} bytes")
            logger.info(f"  Permissions: {oct(CREDENTIALS_FILE.stat().st_mode)[-3:]}")

        return True

    except httpx.HTTPStatusError as e:
        logger.error(
            f"✗ Failed to download GCP credentials from Azure (HTTP {e.response.status_code})"
        )
        logger.error(f"  URL: {AZURE_BLOB_URL}")
        logger.error(f"  Error: {e}")
        return False

    except httpx.TimeoutException:
        logger.error(f"✗ Timeout downloading GCP credentials from Azure Blob Storage")
        logger.error(f"  URL: {AZURE_BLOB_URL}")
        return False

    except PermissionError as e:
        logger.error(f"✗ Permission denied when writing credentials file: {e}")
        logger.error(f"  Directory: {CREDENTIALS_DIR}")
        logger.error(f"  File: {CREDENTIALS_FILE}")
        return False

    except Exception as e:
        import traceback

        logger.error(f"✗ Unexpected error checking/downloading GCP credentials: {e}")
        logger.error(traceback.format_exc())
        return False


def get_credentials_path() -> Path:
    """
    Get the path to the GCP credentials file.

    Returns:
        Path: Absolute path to credentials.json
    """
    return CREDENTIALS_FILE


def credentials_exist() -> bool:
    """
    Check if GCP credentials file exists.

    Returns:
        bool: True if credentials.json exists, False otherwise
    """
    return CREDENTIALS_FILE.exists()
