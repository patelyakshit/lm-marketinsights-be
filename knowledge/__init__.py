"""
Knowledge Base Module

Provides fast access to domain knowledge including:
- ArcGIS Tapestry 2025 segmentation data
- LifeMode group information
- Industry thresholds and recommendations
"""

from knowledge.tapestry_service import (
    TapestryKnowledgeService,
    get_tapestry_service,
)

__all__ = [
    "TapestryKnowledgeService",
    "get_tapestry_service",
]
