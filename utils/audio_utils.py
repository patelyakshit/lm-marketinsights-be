"""
Simple audio utilities for BIDI streaming.

Provides only essential base64 encoding/decoding for audio transmission.
Simple is better than complex.
"""

import base64


def encode_audio_base64(audio_bytes: bytes) -> str:
    """
    Encode audio bytes to Base64 string for WebSocket transmission.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(audio_bytes).decode("utf-8")


def decode_audio_base64(audio_base64: str) -> bytes:
    """
    Decode Base64 string to audio bytes.

    Args:
        audio_base64: Base64 encoded audio string

    Returns:
        Raw audio bytes
    """
    return base64.b64decode(audio_base64)
