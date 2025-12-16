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


def fix_private_key_newlines(private_key: str) -> str:
    """
    Fix private_key newlines that may be escaped in various ways.

    When JSON is pasted into Railway/Heroku env vars, newlines can be:
    - Literal \\n (needs to become actual newline)
    - Already correct actual newlines
    - Mixed escaping

    Returns:
        str: Private key with proper newlines
    """
    original = private_key

    # Pattern 1: Literal backslash-n (most common in Railway)
    # In Python, "\\n" represents a literal backslash followed by 'n'
    if "\\n" in private_key:
        private_key = private_key.replace("\\n", "\n")
        logger.info("  Fixed \\\\n -> newline")

    # Pattern 2: Double-escaped (\\\\n in the raw string)
    if "\\\\n" in private_key:
        private_key = private_key.replace("\\\\n", "\n")
        logger.info("  Fixed \\\\\\\\n -> newline")

    # Pattern 3: Raw backslash followed by 'n' character (appears as r'\n')
    # Check if the key has the proper PEM structure after fixes
    if "-----BEGIN" in private_key and "\n" not in private_key:
        # No newlines at all - try to fix common patterns
        private_key = private_key.replace("\\n", "\n")
        logger.info("  Forced newline conversion (no newlines found)")

    # Validate PEM structure
    if "-----BEGIN PRIVATE KEY-----" in private_key:
        # Count newlines - a valid PEM should have multiple
        newline_count = private_key.count("\n")
        logger.info(f"  Private key has {newline_count} newlines")

        # Log first part of key for debugging (safe - just the header)
        key_preview = private_key[:50].replace("\n", "\\n")
        logger.info(f"  Key preview: {key_preview}...")
    else:
        logger.warning("  Private key doesn't contain expected PEM header!")
        # Log what we see at the start
        key_start = repr(private_key[:60])
        logger.warning(f"  Key starts with: {key_start}")

    if private_key != original:
        logger.info("  Private key newlines were fixed")

    return private_key


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

    # Log raw input info for debugging
    logger.info(f"  Raw env var length: {len(json_creds)} chars")

    try:
        # Validate it's valid JSON
        creds_dict = json.loads(json_creds)

        logger.info(f"  Parsed JSON keys: {list(creds_dict.keys())}")

        # Fix private_key newlines
        if "private_key" in creds_dict:
            creds_dict["private_key"] = fix_private_key_newlines(creds_dict["private_key"])
        else:
            logger.error("  No 'private_key' found in credentials!")
            return False

        # Ensure directory exists
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

        # Log what we're about to write
        private_key = creds_dict.get("private_key", "")
        logger.info(f"  Private key length: {len(private_key)} chars")
        logger.info(f"  Total fields in creds: {len(creds_dict)}")

        # Write to file
        json_content = json.dumps(creds_dict, indent=2)
        logger.info(f"  JSON content length: {len(json_content)} chars")
        CREDENTIALS_FILE.write_text(json_content)
        CREDENTIALS_FILE.chmod(0o600)

        # Verify written file
        written_size = CREDENTIALS_FILE.stat().st_size
        logger.info(f"  Written file size: {written_size} bytes")

        # Read back and verify private_key
        with open(CREDENTIALS_FILE, "r") as f:
            verified_creds = json.load(f)
            verified_pk = verified_creds.get("private_key", "")
            verified_newlines = verified_pk.count("\n")
            logger.info(f"  Verified private_key length: {len(verified_pk)} chars")
            logger.info(f"  Verified private_key newlines: {verified_newlines}")
            # Check PEM structure
            if "-----BEGIN PRIVATE KEY-----" in verified_pk and "-----END PRIVATE KEY-----" in verified_pk:
                logger.info("  ✓ PEM structure verified")
            else:
                logger.error("  ✗ PEM structure MISSING!")
                logger.error(f"    Key starts with: {repr(verified_pk[:80])}")

        # Set the env var to point to this file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)

        logger.info(f"✓ Created GCP credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON env var")
        logger.info(f"  File: {CREDENTIALS_FILE}")
        logger.info(f"  File size: {CREDENTIALS_FILE.stat().st_size} bytes")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        # Log a preview of the raw value for debugging
        preview = json_creds[:100] if len(json_creds) > 100 else json_creds
        logger.error(f"  JSON preview: {repr(preview)}...")
        return False
    except Exception as e:
        logger.error(f"✗ Error creating credentials from env var: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
