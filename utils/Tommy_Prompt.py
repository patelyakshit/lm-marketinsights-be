"""
Tommy's Car Wash Context for Marketing Post Generation

This module provides context information about Tommy's Car Wash
to be dynamically injected into marketing post and image generation prompts.
"""

import os
import logging
from typing import Optional
from decouple import config
from PIL import Image

logger = logging.getLogger(__name__)


def get_tommy_context() -> str:
    """
    Get comprehensive context about Tommy's Car Wash for marketing content generation.

    Returns:
        Formatted string containing Tommy's Car Wash business information,
        brand voice, services, and marketing guidelines.
    """
    return """
# TOMMY'S CAR WASH CONTEXT

**Business Overview:**
Tommy's Car Wash is a premium car wash service focused on delivering exceptional quality and customer satisfaction. We pride ourselves on providing fast, efficient, and thorough car cleaning services that keep vehicles looking their best.

**Brand Voice & Personality:**
- Friendly, approachable, and community-focused
- Professional yet personable
- Emphasizes quality, speed, and convenience
- Values customer relationships and local community connections

**Services:**
- Basic Wash: Quick and affordable exterior cleaning
- Premium Wash: Comprehensive exterior and interior detailing
- Unlimited Monthly Plans: Cost-effective subscription for regular customers
- Premium Services: Protective wax/coating, interior deep cleaning, and specialty treatments

**Visual Style Guidelines:**
- Clean, bright, and modern aesthetic
- Showcase sparkling clean vehicles
- Professional, welcoming atmosphere
- Emphasis on the transformation from dirty to clean
- Include visual elements that convey speed and efficiency

When generating marketing content for Tommy's Car Wash, incorporate these brand elements naturally while maintaining the campaign's specific goals, tone, and target audience alignment.
""".strip()


def get_tommy_logo_image() -> Optional[Image.Image]:
    """
    Load and return Tommy Terific's Car Wash logo image for multimodal image generation.

    Reads TOMMY_LOGO_FILE environment variable which should contain
    the filename or relative path to the logo file at the root of the application.

    Returns:
        PIL Image object if logo file exists and can be loaded,
        None if logo file not configured or cannot be loaded.
    """
    logo_file = config("TOMMY_LOGO_FILE", default=None)

    if not logo_file:
        return None

    try:
        # Logo file is at root of application
        # Get the root directory (parent of utils directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        logo_path = os.path.join(root_dir, logo_file)

        if not os.path.exists(logo_path):
            logger.warning(f"Tommy logo file not found at: {logo_path}")
            return None

        # Load the image using PIL
        logo_image = Image.open(logo_path)
        logger.info(f"Successfully loaded Tommy logo from: {logo_path}")
        return logo_image

    except Exception as e:
        logger.error(f"Error loading Tommy logo from {logo_file}: {e}")
        return None
