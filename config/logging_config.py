"""
Centralized Logging Configuration for Bumblebee Application.

This module provides a single source of truth for application-wide logging
configuration with STDIO-only output and environment-based log levels.
"""

import logging
import sys

from decouple import config


def setup_logging():
    """
    Configure application-wide logging with STDIO output only.

    Features:
    - Environment-based log level via LOG_LEVEL env var
    - Consistent formatting across all modules
    - STDIO-only output (no file handlers)
    - Suppresses noisy third-party loggers
    """
    # Get log level from environment (default: DEBUG for better visibility)
    log_level = config("LOG_LEVEL", default="DEBUG").upper()

    # Map string to logging level
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level_value = level_mapping.get(log_level, logging.INFO)

    # Configure root logger with STDIO handler only
    logging.basicConfig(
        level=log_level_value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("message_queue").setLevel(logging.WARNING)
    logging.getLogger(r"google_adk").setLevel(logging.WARNING)
    logging.getLogger(r"google_adk.google.adk.flows.llm_flows.base_llm_flo").setLevel(
        logging.DEBUG
    )

    # Log the configuration
    root_logger = logging.getLogger(__name__)
    root_logger.info(f"Logging initialized with level: {log_level}")
