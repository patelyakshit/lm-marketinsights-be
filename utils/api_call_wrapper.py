"""
API Call Wrapper with Rate Limiting and 429 Error Handling

Provides a convenient wrapper for Gemini API calls that automatically handles:
- Rate limiting
- 429 error retries with proper backoff
- Token estimation
"""

import logging
from typing import Optional, Any
from google import genai
from google.genai import types as genai_types

from utils.rate_limiter import get_rate_limiter, estimate_tokens
from utils.error_handlers import ErrorRecovery

logger = logging.getLogger(__name__)


async def generate_content_with_rate_limit(
    client: genai.Client,
    model: str,
    prompt: str,
    max_retries: int = 3,
    config: Optional[genai_types.GenerateContentConfig] = None,
) -> genai_types.GenerateContentResponse:
    """
    Generate content with automatic rate limiting and 429 error handling.
    
    Args:
        client: GenAI client instance
        model: Model name (e.g., "gemini-2.5-flash")
        prompt: Prompt text
        max_retries: Maximum number of retries on 429 errors (default: 3)
        config: Optional GenerateContentConfig
        
    Returns:
        GenerateContentResponse
        
    Raises:
        Exception: If all retries are exhausted or non-429 error occurs
    """
    # Estimate tokens and acquire rate limiter permission
    estimated_tokens = estimate_tokens(prompt)
    rate_limiter = get_rate_limiter()
    await rate_limiter.acquire(estimated_tokens)
    
    # Make API call with 429 error handling
    response = None
    for attempt in range(max_retries):
        try:
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)],
                )
            ]
            
            if config:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
            
            return response  # Success
            
        except Exception as e:
            # Check if it's a 429 error
            if ErrorRecovery.is_429_error(e):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"429 error on attempt {attempt + 1}/{max_retries}, "
                        f"handling backoff... Error: {str(e)[:200]}"
                    )
                    await ErrorRecovery.handle_429_error(e)
                    continue
                else:
                    logger.error(
                        f"429 error after {max_retries} attempts, giving up. "
                        f"Error: {str(e)}"
                    )
                    raise
            else:
                # Not a 429 error, re-raise immediately
                logger.error(f"Non-429 error during API call: {str(e)}")
                raise
    
    # Should never reach here, but just in case
    if response is None:
        raise Exception(f"Failed to generate content after {max_retries} attempts")
    
    return response

