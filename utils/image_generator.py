"""
Image Generation and Upload Utility

Handles AI image generation via Gemini and uploads to Azure Blob Storage.
"""

import hashlib
import logging
import os
from io import BytesIO
from typing import Optional
from decouple import config
from PIL import Image
from google import genai
from google.genai import types as genai_types

from utils.azure_storage import AzureBlobStorageManager

logger = logging.getLogger(__name__)

# Separate API key for image generation (Gemini 3 Pro Image)
IMAGE_GEN_API_KEY = config(
    "IMAGE_GEN_API_KEY",
    default="AIzaSyBiDiReSuCV_ZvCD7YzqHlHkBF2lvXLCPw"
)

# Create a separate client for image generation using API key (NOT Vertex AI)
_image_gen_client: Optional[genai.Client] = None

def get_image_gen_client() -> genai.Client:
    """Get or create a client specifically for image generation using API key.

    IMPORTANT: Must use vertexai=False to force Google AI Studio endpoint.
    The GOOGLE_GENAI_USE_VERTEXAI=1 env var would otherwise force Vertex AI,
    which doesn't accept API keys.
    """
    global _image_gen_client
    if _image_gen_client is None:
        # Explicitly disable Vertex AI to use Google AI Studio with API key
        _image_gen_client = genai.Client(api_key=IMAGE_GEN_API_KEY, vertexai=False)
        logger.info("Created image generation client with API key (Google AI Studio, not Vertex AI)")
    return _image_gen_client


# Azure Storage Configuration for marketing images
AZURE_ACCOUNT_KEY = config(
    "PLACESTORY_AZURE_STORAGE_ACCOUNT_KEY", default=""
)
AZURE_CONTAINER_NAME = config(
    "PLACESTORY_AZURE_CONTAINER_NAME", default="placestory-images"
)
AZURE_STORAGE_URL = config("PLACESTORY_AZURE_STORAGE_URL", default="")
AZURE_STORAGE_ACCOUNT_NAME = config(
    "PLACESTORY_AZURE_STORAGE_ACCOUNT_NAME", default=""
)


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
        logger.error(f"Error with local fallback: {e}")
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
    logger.info(f"Generating image {prompt_index + 1}: '{prompt[:60]}...'")

    try:
        # Use dedicated image generation client with API key for Gemini 3
        client = get_image_gen_client()

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
        gen_config = genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        # Use Gemini 3 Pro Image Preview for image generation with API key
        # Must use vertexai=False in client to access Google AI Studio (not Vertex AI)
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=gen_config,
        )

        # Extract image bytes from Gemini response
        image_bytes = None
        if response.candidates and len(response.candidates) > 0:
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
                account_name=AZURE_STORAGE_ACCOUNT_NAME,
                account_key=AZURE_ACCOUNT_KEY,
                container_name=AZURE_CONTAINER_NAME,
            )
            logger.debug(
                f"Uploading image to Azure: container={AZURE_CONTAINER_NAME}, filename={filename}, size={len(image_bytes)} bytes"
            )
            result = await azure_blob_storage_manager.upload_file(
                image_bytes, filename, content_type="image/png"
            )

            if result.get("success"):
                # Prefer SAS URL for secure access, fallback to blob_url
                image_url = result.get("sas_url") or result.get("blob_url")
                if image_url:
                    # Verify URL has proper format
                    if not image_url.startswith("http"):
                        logger.error(f"Invalid URL format returned: {image_url}")
                        return await upload_to_local_fallback(image_bytes, filename)

                    logger.info(
                        f"Successfully uploaded image to Azure: {filename}"
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
                f"Error uploading to Azure for image {prompt_index + 1}: {upload_error}"
            )
            return await upload_to_local_fallback(image_bytes, filename)

    except Exception as e:
        logger.error(f"Error generating image {prompt_index + 1}: {e}")
        return "https://via.placeholder.com/800x600?text=Error"
