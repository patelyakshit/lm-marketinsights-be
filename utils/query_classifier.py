"""
Query Classification System

Classifies user queries into types for adaptive response formatting.
Different query types get different response styles for optimal UX.

Query Types:
- GREETING: Simple hellos, greetings
- NAVIGATION: Map zoom, pan, location requests
- EXPLORATION: "What", "Show me", information requests
- ANALYSIS: Data analysis, statistics, comparisons
- TASK: Execute actions, export data, generate reports
- CLARIFICATION: Yes/No, follow-up questions
"""

import re
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of user queries for adaptive responses."""
    GREETING = "greeting"           # Simple greetings (hi, hello, hey)
    NAVIGATION = "navigation"       # Map navigation (zoom, pan, go to)
    EXPLORATION = "exploration"     # Information seeking (what, show me, tell me)
    ANALYSIS = "analysis"           # Data analysis (compare, analyze, statistics)
    TASK = "task"                   # Task execution (export, create, generate)
    CLARIFICATION = "clarification" # Yes/no, confirmation, follow-ups
    GENERAL = "general"             # Default for unclassified queries


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    suggested_response_style: str  # "brief", "action", "detailed", "structured"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryClassifier:
    """
    Classifies queries for adaptive response formatting.

    Uses pattern matching for fast, deterministic classification.
    More sophisticated ML-based classification can be added later.
    """

    # Pattern definitions for each query type
    PATTERNS = {
        QueryType.GREETING: [
            r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings|howdy)(\s|!|\.|,|$)",
            r"^(what's\s+up|sup|yo)(\s|!|\.|$)",
        ],
        QueryType.NAVIGATION: [
            r"\b(zoom\s+(to|in|out)|pan\s+to|go\s+to|navigate\s+to|show\s+me\s+on\s+map)\b",
            r"\b(center\s+on|focus\s+on|fly\s+to|move\s+to)\b",
            r"\b(add\s+(pin|marker|layer)|remove\s+(pin|marker|layer)|clear\s+map)\b",
        ],
        QueryType.EXPLORATION: [
            r"^(what|which|where|who|when|why|how)\s+",
            r"\b(show\s+me|tell\s+me|explain|describe|list)\b",
            r"\b(what\s+is|what\s+are|what\s+does)\b",
        ],
        QueryType.ANALYSIS: [
            r"\b(analyze|analysis|compare|comparison|statistics|stats)\b",
            r"\b(trend|pattern|distribution|breakdown|summary)\b",
            r"\b(average|total|count|sum|median|percentage)\b",
            r"\b(demographics|demographic|population|income|age)\b",
        ],
        QueryType.TASK: [
            r"\b(export|download|save|create|generate|make)\b",
            r"\b(run|execute|perform|do)\b",
            r"\b(report|csv|geojson|pdf)\b",
            r"\b(marketing\s+post|social\s+media)\b",
        ],
        QueryType.CLARIFICATION: [
            r"^(yes|no|yeah|nope|sure|okay|ok|correct|right)(\s|!|\.|,|$)",
            r"^(that's\s+(right|correct)|exactly|perfect)(\s|!|\.|$)",
            r"^\d+$",  # Just a number (selection from options)
        ],
    }

    # Response style mappings
    RESPONSE_STYLES = {
        QueryType.GREETING: "brief",          # Short, friendly response
        QueryType.NAVIGATION: "action",       # Confirm action + result
        QueryType.EXPLORATION: "detailed",    # Informative explanation
        QueryType.ANALYSIS: "structured",     # Tables, charts, organized data
        QueryType.TASK: "action",             # Status update + result
        QueryType.CLARIFICATION: "brief",     # Quick acknowledgment
        QueryType.GENERAL: "detailed",        # Default to detailed
    }

    def __init__(self):
        # Compile patterns for performance
        self._compiled_patterns = {}
        for query_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[query_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query into a type with confidence score.

        Args:
            query: User query text

        Returns:
            QueryClassification with type, confidence, and suggested style
        """
        if not query:
            return QueryClassification(
                query_type=QueryType.GENERAL,
                confidence=1.0,
                suggested_response_style="detailed"
            )

        normalized = query.strip().lower()

        # Check patterns in order of specificity
        for query_type in [
            QueryType.GREETING,      # Check first (short patterns)
            QueryType.CLARIFICATION, # Check early (yes/no responses)
            QueryType.NAVIGATION,    # Specific actions
            QueryType.TASK,          # Specific actions
            QueryType.ANALYSIS,      # Data-related keywords
            QueryType.EXPLORATION,   # Question patterns
        ]:
            patterns = self._compiled_patterns.get(query_type, [])
            for pattern in patterns:
                if pattern.search(normalized):
                    return QueryClassification(
                        query_type=query_type,
                        confidence=0.85,  # Pattern match confidence
                        suggested_response_style=self.RESPONSE_STYLES[query_type],
                        metadata={"matched_pattern": pattern.pattern}
                    )

        # Default to GENERAL
        return QueryClassification(
            query_type=QueryType.GENERAL,
            confidence=0.5,  # Lower confidence for unclassified
            suggested_response_style="detailed"
        )

    def get_response_instruction(self, classification: QueryClassification) -> str:
        """
        Get response formatting instruction based on classification.

        Args:
            classification: Query classification result

        Returns:
            Instruction string to append to agent prompt
        """
        style = classification.suggested_response_style

        instructions = {
            "brief": (
                "Respond concisely in 1-2 sentences. "
                "No lists or lengthy explanations needed."
            ),
            "action": (
                "Confirm the action briefly, then show the result. "
                "Format: 'Done. [Result description]' or '[Action] complete.'"
            ),
            "detailed": (
                "Provide a clear, informative response. "
                "Use markdown formatting for readability."
            ),
            "structured": (
                "Present data in organized format using tables or lists. "
                "Include key metrics and insights. Use markdown tables."
            ),
        }

        return instructions.get(style, instructions["detailed"])


# Global singleton instance
query_classifier = QueryClassifier()


def classify_query(query: str) -> QueryClassification:
    """
    Convenience function to classify a query.

    Args:
        query: User query text

    Returns:
        QueryClassification result
    """
    return query_classifier.classify(query)


def get_adaptive_instruction(query: str) -> Optional[str]:
    """
    Get adaptive response instruction for a query.

    Args:
        query: User query text

    Returns:
        Response instruction string or None for general queries
    """
    classification = query_classifier.classify(query)

    # Only return instruction for high-confidence classifications
    if classification.confidence >= 0.7:
        return query_classifier.get_response_instruction(classification)

    return None
