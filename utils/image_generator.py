"""
Image Generation and Upload Utility for PlaceStory

Handles AI image generation via Gemini and uploads to Azure Blob Storage via the Manager.
Recursively processes placestory JSON to replace image_prompt fields with URLs.
"""

import hashlib
import asyncio
import logging
import os
from io import BytesIO
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from decouple import config
from PIL import Image
from google.genai import types as genai_types

from utils.genai_client_manager import get_genai_client
from utils.azure_storage import AzureBlobStorageManager
from tools.placestory_tools import _send_placestory_status

logger = logging.getLogger(__name__)


PLACESTORY_AZURE_ACCOUNT_KEY = config(
    "PLACESTORY_AZURE_STORAGE_ACCOUNT_KEY", default=""
)
PLACESTORY_AZURE_CONTAINER_NAME = config(
    "PLACESTORY_AZURE_CONTAINER_NAME", default="placestory-images"
)
PLACESTORY_AZURE_STORAGE_URL = config("PLACESTORY_AZURE_STORAGE_URL", default="")
PLACESTORY_AZURE_STORAGE_ACCOUNT_NAME = config(
    "PLACESTORY_AZURE_STORAGE_ACCOUNT_NAME", default=""
)

_placestory_storage_manager = AzureBlobStorageManager(
    account_url=PLACESTORY_AZURE_STORAGE_URL,
    account_key=PLACESTORY_AZURE_ACCOUNT_KEY,
    container_name=PLACESTORY_AZURE_CONTAINER_NAME,
)


async def upload_image_via_manager(
    image_bytes: bytes, filename: str, content_type: str = "image/png"
) -> str:
    """
    Uploads using the configured AzureBlobStorageManager.
    """
    try:
        result = await _placestory_storage_manager.upload_file(
            file_content=image_bytes, filename=filename, content_type=content_type
        )

        if result.get("success"):
            # Prefer SAS URL for secure access, fallback to blob_url
            return result.get("sas_url") or result.get("blob_url")
        else:
            logger.error(f"Manager upload failed: {result.get('error')}")
            return await upload_to_local_fallback(image_bytes, filename)

    except Exception as e:
        logger.error(f"❌ Error using Azure Manager: {e}")
        return await upload_to_local_fallback(image_bytes, filename)


async def upload_to_local_fallback(image_bytes: bytes, filename: str) -> str:
    """
    Fallback to local storage if Azure upload fails.
    """
    try:
        os.makedirs("static", exist_ok=True)
        filepath = os.path.join("static", filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return f"/static/{filename}"
    except Exception as e:
        logger.error(f"❌ Error with local fallback: {e}")
        return "https://via.placeholder.com/800x600?text=Generation+Failed"


async def generate_and_save_image_async(
    prompt: str, prompt_index: int = 0, context_image: Optional[Image.Image] = None
) -> str:
    """
    Generate an image from a prompt using Gemini and upload to Azure.

    Supports multimodal input: can include a context image (e.g., logo) along with text prompt
    for image-to-image generation.

    Args:
        prompt: Text prompt describing the image to generate
        prompt_index: Index for naming generated images
        context_image: Optional PIL Image to use as context/reference for generation

    Returns:
        URL of the generated and uploaded image
    """
    logger.info(f"🎨 Generating image {prompt_index + 1}: '{prompt[:60]}...'")

    try:
        client = get_genai_client()

        # Build contents list for multimodal input
        parts = [genai_types.Part(text=prompt)]

        # Add context image if provided (multimodal input)
        if context_image is not None:
            # Convert PIL Image to bytes
            img_bytes_io = BytesIO()
            # Save in PNG format for best compatibility
            context_image.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()

            # Determine MIME type based on image format
            mime_type = "image/png"

            # Create inline data part for the image
            parts.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        data=img_bytes,
                        mime_type=mime_type,
                    )
                )
            )
            logger.info(
                f"Added context image to multimodal input (size: {len(img_bytes)} bytes)"
            )

        # Create content with parts
        contents = [
            genai_types.Content(
                role="user",
                parts=parts,
            )
        ]

        # Configure for image output
        config = genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=config,
        )

        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                break

        if not image_bytes:
            logger.warning(f"No image bytes found for prompt: {prompt[:40]}...")
            return "https://via.placeholder.com/800x600?text=Image+Generation+Failed"

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        filename = f"image_{prompt_hash}_{prompt_index}.png"

        # Validate filename has extension
        if not filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            logger.warning(f"Filename missing image extension: {filename}, adding .png")
            filename = f"{filename}.png"

        # Upload to Azure Blob Storage
        try:
            azure_blob_storage_manager = AzureBlobStorageManager(
                account_name=PLACESTORY_AZURE_STORAGE_ACCOUNT_NAME,
                account_key=PLACESTORY_AZURE_ACCOUNT_KEY,
                container_name=PLACESTORY_AZURE_CONTAINER_NAME,
            )
            logger.debug(
                f"Uploading image to Azure: container={PLACESTORY_AZURE_CONTAINER_NAME}, filename={filename}, size={len(image_bytes)} bytes"
            )
            result = await azure_blob_storage_manager.upload_file(
                image_bytes, filename, content_type="image/png"
            )

            if result.get("success"):
                # Prefer SAS URL for secure access, fallback to blob_url
                image_url = result.get("sas_url") or result.get("blob_url")
                if image_url:
                    # Verify the URL contains the filename
                    if filename not in image_url:
                        logger.warning(
                            f"Generated URL does not contain filename '{filename}': {image_url[:100]}"
                        )
                    # Verify URL has proper format
                    if not image_url.startswith("http"):
                        logger.error(f"Invalid URL format returned: {image_url}")
                        return await upload_to_local_fallback(image_bytes, filename)

                    logger.info(
                        f"✅ Successfully uploaded image to Azure: {filename} "
                        f"(URL: {image_url[:80]}..., filename in URL: {filename in image_url})"
                    )
                    return image_url
                else:
                    logger.warning(
                        f"Azure upload succeeded but no URL returned, falling back to local storage for {filename}"
                    )
                    return await upload_to_local_fallback(image_bytes, filename)
            else:
                # Fallback to local storage if Azure upload fails
                logger.warning(
                    f"Azure upload failed, falling back to local storage for {filename}"
                )
                return await upload_to_local_fallback(image_bytes, filename)
        except Exception as upload_error:
            # If Azure upload raises an exception, fall back to local storage
            logger.error(
                f"❌ Error uploading to Azure for image {prompt_index + 1}: {upload_error}"
            )
            return await upload_to_local_fallback(image_bytes, filename)

    except Exception as e:
        logger.error(f"❌ Error generating image {prompt_index + 1}: {e}")
        return "https://via.placeholder.com/800x600?text=Error"


def find_image_jobs_recursively(data: Any, path: str = "root") -> List[Dict[str, Any]]:
    """
    Recursively find all image blocks with prompts in the placestory JSON.
    """
    jobs = []

    if isinstance(data, dict):
        if data.get("type") == "image" and data.get("payload", {}).get("image_prompt"):

            jobs.append(
                {
                    "prompt": data["payload"]["image_prompt"],
                    "source_obj": data["payload"]["source"],
                    "payload_obj": data["payload"],
                    "block_id": data.get("id", "unknown"),
                    "path": path,
                }
            )
            logger.debug(f"Found image prompt at {path}: {data.get('id')}")

        for key, value in data.items():
            jobs.extend(find_image_jobs_recursively(value, f"{path}.{key}"))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            jobs.extend(find_image_jobs_recursively(item, f"{path}[{i}]"))

    return jobs


async def process_placestory_images(
    placestory_json: Dict[str, Any], tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Find prompts, generate, upload via Manager, and strip prompts.
    """
    logger.info("🔍 Searching for image prompts in placestory...")

    image_jobs = find_image_jobs_recursively(placestory_json)

    if not image_jobs:
        logger.info("No image prompts found to process.")
        return placestory_json

    logger.info(f"Found {len(image_jobs)} image(s) to generate and upload")

    await _send_placestory_status(
        tool_context,
        step_id="process_images",
        label="Generating PlaceStory images",
        status="in_progress",
        details=f"Generating {len(image_jobs)} images for the PlaceStory.",
    )

    tasks = [
        generate_and_save_image_async(job["prompt"], i)
        for i, job in enumerate(image_jobs)
    ]

    generated_urls = await asyncio.gather(*tasks)

    for i, job in enumerate(image_jobs):
        url = generated_urls[i]
        logger.info(f"🔄 Updating URL for {job['block_id']}: {url}")

        job["source_obj"]["url"] = url

        if "image_prompt" in job["payload_obj"]:
            del job["payload_obj"]["image_prompt"]

    await _send_placestory_status(
        tool_context,
        step_id="process_images",
        status="done",
    )

    logger.info(f"✅ Successfully processed {len(image_jobs)} images")
    return placestory_json
