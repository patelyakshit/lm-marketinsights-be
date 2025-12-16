"""
Tapestry Knowledge Service

Provides fast access to ArcGIS Tapestry 2025 segmentation data.
Loads from local JSON cache with optional API fallback.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Path to the knowledge base
KNOWLEDGE_DIR = Path(__file__).parent
TAPESTRY_JSON = KNOWLEDGE_DIR / "tapestry_2025.json"


class TapestryKnowledgeService:
    """
    Fast access to Tapestry 2025 segment information.

    Features:
    - Instant lookups from JSON cache (< 1ms)
    - All 60 segments with detailed metadata
    - LifeMode group information
    - Business recommendations per segment
    """

    def __init__(self, json_path: Path = TAPESTRY_JSON):
        """Initialize the service with the JSON knowledge base."""
        self._data: Dict[str, Any] = {}
        self._segments: Dict[str, Dict] = {}
        self._lifemode_groups: Dict[str, Dict] = {}
        self._loaded = False
        self._json_path = json_path

        # Load data on init
        self._load_data()

    def _load_data(self) -> None:
        """Load the Tapestry data from JSON."""
        try:
            if self._json_path.exists():
                with open(self._json_path, "r") as f:
                    self._data = json.load(f)

                self._segments = self._data.get("segments", {})
                self._lifemode_groups = self._data.get("lifemode_groups", {})
                self._loaded = True

                logger.info(
                    f"Loaded Tapestry {self._data.get('version', 'unknown')} data: "
                    f"{len(self._segments)} segments, {len(self._lifemode_groups)} LifeMode groups"
                )
            else:
                logger.warning(f"Tapestry JSON not found at {self._json_path}")
        except Exception as e:
            logger.error(f"Error loading Tapestry data: {e}")

    def get_segment(self, code: str) -> Optional[Dict]:
        """
        Get segment details by code.

        Args:
            code: Segment code (e.g., "A1", "D2", "L3")

        Returns:
            Segment dict with all metadata, or None if not found
        """
        # Normalize code (uppercase)
        code = code.upper().strip()
        return self._segments.get(code)

    def get_segment_response(self, code: str) -> str:
        """
        Get a formatted response for a segment query.

        Args:
            code: Segment code (e.g., "A1", "D2")

        Returns:
            Formatted markdown response
        """
        segment = self.get_segment(code)

        if not segment:
            # Return helpful message for unknown segments
            return self._unknown_segment_response(code)

        # Build formatted response
        name = segment.get("name", "Unknown")
        lifemode = segment.get("lifemode", "")
        lifemode_name = segment.get("lifemode_name", "")
        description = segment.get("description", "")
        median_age = segment.get("median_age", "N/A")
        median_income = segment.get("median_income", "N/A")
        housing = segment.get("housing", "N/A")
        characteristics = segment.get("characteristics", [])
        consumer_behavior = segment.get("consumer_behavior", [])
        best_for = segment.get("best_for", [])

        response = f"""**{name} ({code})** - LifeMode {lifemode}: {lifemode_name}

{description}

**Demographics:**
- Median Age: {median_age}
- Median Income: {median_income}
- Housing: {housing}

**Key Characteristics:**
{self._format_list(characteristics)}

**Consumer Behavior:**
{self._format_list(consumer_behavior)}

**Best Business Types:**
{self._format_list(best_for)}"""

        return response

    def get_lifemode_group(self, letter: str) -> Optional[Dict]:
        """
        Get LifeMode group details by letter.

        Args:
            letter: LifeMode letter (A-L)

        Returns:
            LifeMode dict with metadata and segment list
        """
        letter = letter.upper().strip()
        return self._lifemode_groups.get(letter)

    def get_lifemode_response(self, letter: str) -> str:
        """
        Get a formatted response for a LifeMode query.

        Args:
            letter: LifeMode letter (A-L)

        Returns:
            Formatted markdown response
        """
        group = self.get_lifemode_group(letter)

        if not group:
            return self._unknown_lifemode_response(letter)

        name = group.get("name", "Unknown")
        description = group.get("description", "")
        lifestage = group.get("lifestage", "")
        segments = group.get("segments", [])

        # Get segment names
        segment_details = []
        for seg_code in segments:
            seg = self.get_segment(seg_code)
            if seg:
                segment_details.append(f"- **{seg_code}**: {seg.get('name', 'Unknown')}")

        response = f"""**LifeMode {letter}: {name}**

{description}

**LifeStage:** {lifestage}

**Segments in this group ({len(segments)}):**
{chr(10).join(segment_details)}"""

        return response

    def get_all_segments_summary(self) -> str:
        """Get a summary of all 60 segments."""
        if not self._loaded:
            return "Tapestry data not loaded."

        response = f"""**ArcGIS Tapestry 2025 Overview**

Version: {self._data.get('version', 'Unknown')}
Total Segments: {self._data.get('total_segments', 60)}
LifeMode Groups: {self._data.get('total_lifemode_groups', 12)}

**LifeMode Groups (A-L):**
"""
        for letter in "ABCDEFGHIJKL":
            group = self._lifemode_groups.get(letter)
            if group:
                name = group.get("name", "")
                seg_count = len(group.get("segments", []))
                response += f"- **{letter}: {name}** ({seg_count} segments)\n"

        response += "\nAsk about any segment (e.g., 'what is A1 segment') for details!"
        return response

    def search_segments(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search segments by keyword.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching segments
        """
        query_lower = query.lower()
        results = []

        for code, segment in self._segments.items():
            score = 0

            # Check name
            if query_lower in segment.get("name", "").lower():
                score += 10

            # Check description
            if query_lower in segment.get("description", "").lower():
                score += 5

            # Check characteristics
            for char in segment.get("characteristics", []):
                if query_lower in char.lower():
                    score += 3

            # Check best_for
            for biz in segment.get("best_for", []):
                if query_lower in biz.lower():
                    score += 4

            if score > 0:
                results.append({"code": code, "segment": segment, "score": score})

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_segments_for_business(self, business_type: str) -> List[Dict]:
        """
        Find segments that are good matches for a business type.

        Args:
            business_type: Type of business (e.g., "coffee shop", "fitness")

        Returns:
            List of recommended segments
        """
        return self.search_segments(business_type, limit=10)

    def _format_list(self, items: List[str]) -> str:
        """Format a list as bullet points."""
        if not items:
            return "- N/A"
        return "\n".join(f"- {item}" for item in items[:5])

    def _unknown_segment_response(self, code: str) -> str:
        """Response for unknown segment codes."""
        return f"""I don't have information for segment **{code}** in my knowledge base.

**ArcGIS Tapestry 2025** has 60 segments organized in 12 LifeMode groups (A-L).

**Segment code format:** Letter + Number (e.g., A1, B2, C3)

**LifeMode Groups:**
- A: Urban Threads (A1-A6)
- B: Books and Boots (B1-B3)
- C: Metro Vibes (C1-C6)
- D: Tech Trailblazers (D1-D5)
- E: Community Connections (E1-E6)
- F: Urban Harmony (F1-F5)
- G: Family Fabric (G1-G3)
- H: Family Prosperity (H1-H4)
- I: Countryscapes (I1-I7)
- J: Mature Reflections (J1-J4)
- K: Suburban Shine (K1-K8)
- L: Premier Estates (L1-L3)

Try asking: "what is A1 segment" or "what is LifeMode D"
"""

    def _unknown_lifemode_response(self, letter: str) -> str:
        """Response for unknown LifeMode letters."""
        return f"""I don't have information for LifeMode **{letter}**.

**ArcGIS Tapestry 2025** has 12 LifeMode groups (A through L):

- **A**: Urban Threads
- **B**: Books and Boots
- **C**: Metro Vibes
- **D**: Tech Trailblazers
- **E**: Community Connections
- **F**: Urban Harmony
- **G**: Family Fabric
- **H**: Family Prosperity
- **I**: Countryscapes
- **J**: Mature Reflections
- **K**: Suburban Shine
- **L**: Premier Estates

Try asking: "what is LifeMode A" or "what is tapestry"
"""

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded

    @property
    def version(self) -> str:
        """Get the data version."""
        return self._data.get("version", "unknown")

    @property
    def segment_count(self) -> int:
        """Get the number of segments."""
        return len(self._segments)


# Global singleton instance
_tapestry_service: Optional[TapestryKnowledgeService] = None


def get_tapestry_service() -> TapestryKnowledgeService:
    """Get the global Tapestry service instance."""
    global _tapestry_service
    if _tapestry_service is None:
        _tapestry_service = TapestryKnowledgeService()
    return _tapestry_service


def reset_tapestry_service() -> None:
    """Reset the global service (for testing)."""
    global _tapestry_service
    _tapestry_service = None
