"""
Simple audio configuration for BIDI streaming.

Provides only essential sample rate settings.
Simple is better than complex.
"""

from decouple import config


class AudioConfig:
    """Minimal audio streaming configuration."""

    # Sample rates (only essential settings)
    RECORDING_SAMPLE_RATE = config("AUDIO_RECORDING_SAMPLE_RATE", default=16000, cast=int)
    PLAYBACK_SAMPLE_RATE = config("AUDIO_PLAYBACK_SAMPLE_RATE", default=24000, cast=int)


# Global audio configuration instance
audio_config = AudioConfig()
