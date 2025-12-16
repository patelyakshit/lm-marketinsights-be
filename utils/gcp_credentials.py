"""
GCP Credentials Management Utility

Automatically checks for and downloads GCP service account credentials
from Azure Blob Storage if not present locally.
"""

import logging
import os
from pathlib import Path

import httpx
from decouple import config

logger = logging.getLogger(__name__)

# Credentials directory and file path
# Check if GOOGLE_APPLICATION_CREDENTIALS is set, use that path if it exists
# Otherwise default to /tmp/credentials.json
env_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if env_creds_path:
    CREDENTIALS_FILE = Path(env_creds_path)
    CREDENTIALS_DIR = CREDENTIALS_FILE.parent
else:
    # Using /tmp for container environments (Docker/K8s)
    CREDENTIALS_DIR = Path("/tmp")
    CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

# Azure Blob Storage URL for GCP credentials
AZURE_BLOB_URL = config(
    "AZURE_GCP_SERVICE_ACCOUNT",
    default="https://lmproductwigetsdev.blob.core.windows.net/lmwidget-agent-dev/gcp-cred/Location Matters.json",
)


async def check_or_download_credentials_file() -> bool:
    """
    Check if credentials.json exists, download from Azure if missing.

    The GCP credentials are required for Google ADK and other Google Cloud services.
    If the file doesn't exist, it's automatically downloaded from Azure Blob Storage.

    Returns:
        bool: True if file exists or download succeeded, False otherwise
    """
    try:
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
