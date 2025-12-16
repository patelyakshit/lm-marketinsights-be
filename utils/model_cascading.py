"""
Model Cascading System

Automatically selects the appropriate model tier based on query complexity:
- FAST tier: Simple queries (greetings, navigation, confirmations)
- STANDARD tier: Normal queries (exploration, general questions)
- POWERFUL tier: Complex queries (analysis, multi-step tasks)

This saves costs by using lighter models for simple tasks while
ensuring complex queries get the processing power they need.
"""

import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from decouple import config

from utils.query_classifier import classify_query, QueryType, QueryClassification

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model performance tiers."""
    FAST = "fast"           # Lightweight, low-cost, quick responses
    STANDARD = "standard"   # Balanced performance and cost
    POWERFUL = "powerful"   # Maximum capability for complex tasks


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    tier: ModelTier
    model_name: str
    description: str
    max_tokens: int = 2048


# Default model configurations
# These can be overridden via environment variables
DEFAULT_MODELS = {
    ModelTier.FAST: ModelConfig(
        tier=ModelTier.FAST,
        model_name=config("FAST_MODEL", default="gemini-2.5-flash-lite"),
        description="Fast responses for simple queries",
        max_tokens=1024,
    ),
    ModelTier.STANDARD: ModelConfig(
        tier=ModelTier.STANDARD,
        model_name=config("ROOT_MODEL", default="gemini-2.5-flash-lite"),
        description="Standard model for general queries",
        max_tokens=2048,
    ),
    ModelTier.POWERFUL: ModelConfig(
        tier=ModelTier.POWERFUL,
        model_name=config("POWERFUL_MODEL", default="gemini-2.0-flash"),
        description="Powerful model for complex analysis",
        max_tokens=4096,
    ),
}


# Query type to model tier mapping
QUERY_TYPE_TIERS = {
    QueryType.GREETING: ModelTier.FAST,
    QueryType.CLARIFICATION: ModelTier.FAST,
    QueryType.NAVIGATION: ModelTier.FAST,
    QueryType.EXPLORATION: ModelTier.STANDARD,
    QueryType.GENERAL: ModelTier.STANDARD,
    QueryType.TASK: ModelTier.STANDARD,
    QueryType.ANALYSIS: ModelTier.POWERFUL,
}


class ModelCascader:
    """
    Determines appropriate model tier based on query complexity.

    Usage:
        cascader = ModelCascader()
        model_config = cascader.get_model_for_query("What is the population of Dallas?")
        # Use model_config.model_name for the request
    """

    def __init__(self, models: dict = None):
        """
        Initialize the cascader.

        Args:
            models: Optional custom model configurations
        """
        self.models = models or DEFAULT_MODELS

    def get_tier_for_query(self, query: str) -> ModelTier:
        """
        Determine the appropriate model tier for a query.

        Args:
            query: User query text

        Returns:
            Appropriate ModelTier
        """
        classification = classify_query(query)
        return self.get_tier_for_classification(classification)

    def get_tier_for_classification(
        self, classification: QueryClassification
    ) -> ModelTier:
        """
        Get model tier based on query classification.

        Args:
            classification: Query classification result

        Returns:
            Appropriate ModelTier
        """
        # Use mapped tier or default to STANDARD
        tier = QUERY_TYPE_TIERS.get(classification.query_type, ModelTier.STANDARD)

        # Override to POWERFUL for low-confidence complex queries
        # (we want to be safe with uncertain classifications)
        if classification.confidence < 0.6 and classification.query_type in [
            QueryType.ANALYSIS,
            QueryType.TASK,
        ]:
            tier = ModelTier.POWERFUL

        logger.debug(
            f"Query classified as {classification.query_type.value} -> {tier.value} tier"
        )
        return tier

    def get_model_for_query(self, query: str) -> ModelConfig:
        """
        Get the full model configuration for a query.

        Args:
            query: User query text

        Returns:
            ModelConfig for the appropriate tier
        """
        tier = self.get_tier_for_query(query)
        return self.models[tier]

    def get_model_name_for_query(self, query: str) -> str:
        """
        Get just the model name for a query.

        Args:
            query: User query text

        Returns:
            Model name string
        """
        return self.get_model_for_query(query).model_name


# Global singleton instance
_cascader = ModelCascader()


def get_model_for_query(query: str) -> str:
    """
    Convenience function to get appropriate model for a query.

    Args:
        query: User query text

    Returns:
        Model name string
    """
    return _cascader.get_model_name_for_query(query)


def get_tier_for_query(query: str) -> ModelTier:
    """
    Convenience function to get model tier for a query.

    Args:
        query: User query text

    Returns:
        ModelTier enum value
    """
    return _cascader.get_tier_for_query(query)


def should_use_powerful_model(query: str) -> bool:
    """
    Check if a query should use the powerful model tier.

    Args:
        query: User query text

    Returns:
        True if query needs powerful model
    """
    return get_tier_for_query(query) == ModelTier.POWERFUL
