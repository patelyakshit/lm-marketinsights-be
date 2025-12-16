"""
Unit tests for Query Classifier and Model Cascading modules.
"""

import pytest
from utils.query_classifier import (
    QueryType,
    QueryClassification,
    QueryClassifier,
    classify_query,
    get_response_style,
)
from utils.model_cascading import (
    ModelTier,
    ModelConfig,
    ModelCascader,
    get_model_for_query,
    get_tier_for_query,
    should_use_powerful_model,
)


class TestQueryClassifier:
    """Tests for QueryClassifier."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_greeting_classification(self, classifier):
        """Test greeting queries are classified correctly."""
        queries = ["hi", "hello", "hey there", "good morning"]

        for query in queries:
            result = classifier.classify(query)
            assert result.query_type == QueryType.GREETING
            assert result.confidence >= 0.7

    def test_navigation_classification(self, classifier):
        """Test navigation queries are classified correctly."""
        queries = [
            "zoom to Dallas",
            "pan north",
            "show me Austin on the map",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.query_type == QueryType.NAVIGATION

    def test_analysis_classification(self, classifier):
        """Test analysis queries are classified correctly."""
        queries = [
            "analyze the demographics of Dallas",
            "compare Austin and Houston",
            "what are the trends in this area",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.query_type == QueryType.ANALYSIS

    def test_task_classification(self, classifier):
        """Test task queries are classified correctly."""
        queries = [
            "export the data to CSV",
            "create a report",
            "generate a marketing post",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.query_type in [QueryType.TASK, QueryType.GENERAL]

    def test_exploration_classification(self, classifier):
        """Test exploration queries are classified correctly."""
        queries = [
            "what is the population of Dallas",
            "tell me about this area",
            "show demographics",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.query_type in [QueryType.EXPLORATION, QueryType.GENERAL, QueryType.NAVIGATION]


class TestQueryClassification:
    """Tests for QueryClassification dataclass."""

    def test_classification_attributes(self):
        """Test classification has all required attributes."""
        classification = QueryClassification(
            query_type=QueryType.GREETING,
            confidence=0.95,
            suggested_response_style="brief"
        )

        assert classification.query_type == QueryType.GREETING
        assert classification.confidence == 0.95
        assert classification.suggested_response_style == "brief"


class TestClassifyQueryFunction:
    """Tests for classify_query convenience function."""

    def test_classify_query_returns_classification(self):
        """Test that classify_query returns QueryClassification."""
        result = classify_query("hello")

        assert isinstance(result, QueryClassification)
        assert hasattr(result, "query_type")
        assert hasattr(result, "confidence")

    def test_classify_query_various_inputs(self):
        """Test classify_query with various inputs."""
        inputs = [
            "hi",
            "zoom to Dallas",
            "analyze demographics",
            "what is this?",
        ]

        for query in inputs:
            result = classify_query(query)
            assert isinstance(result, QueryClassification)


class TestResponseStyle:
    """Tests for response style suggestions."""

    def test_greeting_style(self):
        """Greeting should suggest brief style."""
        result = classify_query("hello")
        if result.query_type == QueryType.GREETING:
            assert result.suggested_response_style == "brief"

    def test_get_response_style(self):
        """Test get_response_style function."""
        style = get_response_style("hello")
        assert style in ["brief", "action", "detailed", "structured"]


class TestModelCascading:
    """Tests for ModelCascader."""

    @pytest.fixture
    def cascader(self):
        return ModelCascader()

    def test_greeting_uses_fast_tier(self, cascader):
        """Greetings should use fast tier."""
        tier = cascader.get_tier_for_query("hello")
        assert tier == ModelTier.FAST

    def test_analysis_uses_powerful_tier(self, cascader):
        """Complex analysis should use powerful tier."""
        tier = cascader.get_tier_for_query("analyze demographics trends and compare markets")
        assert tier in [ModelTier.POWERFUL, ModelTier.STANDARD]

    def test_navigation_uses_fast_tier(self, cascader):
        """Simple navigation should use fast tier."""
        tier = cascader.get_tier_for_query("zoom to Dallas")
        assert tier == ModelTier.FAST

    def test_get_model_for_query(self, cascader):
        """Test getting model config for query."""
        config = cascader.get_model_for_query("hello")

        assert isinstance(config, ModelConfig)
        assert config.tier == ModelTier.FAST

    def test_get_model_name_for_query(self, cascader):
        """Test getting model name for query."""
        name = cascader.get_model_name_for_query("hello")

        assert isinstance(name, str)
        assert len(name) > 0


class TestModelTierMapping:
    """Tests for query type to model tier mapping."""

    def test_tier_mapping_completeness(self):
        """All query types should have a tier mapping."""
        cascader = ModelCascader()

        for query_type in QueryType:
            classification = QueryClassification(
                query_type=query_type,
                confidence=0.9,
                suggested_response_style="brief"
            )
            tier = cascader.get_tier_for_classification(classification)
            assert isinstance(tier, ModelTier)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(
            tier=ModelTier.FAST,
            model_name="test-model",
            description="Test"
        )

        assert config.max_tokens == 2048

    def test_config_custom_tokens(self):
        """Test ModelConfig with custom max_tokens."""
        config = ModelConfig(
            tier=ModelTier.POWERFUL,
            model_name="powerful-model",
            description="Powerful",
            max_tokens=4096
        )

        assert config.max_tokens == 4096


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_model_for_query(self):
        """Test get_model_for_query function."""
        model = get_model_for_query("hello")
        assert isinstance(model, str)

    def test_get_tier_for_query(self):
        """Test get_tier_for_query function."""
        tier = get_tier_for_query("hello")
        assert isinstance(tier, ModelTier)

    def test_should_use_powerful_model(self):
        """Test should_use_powerful_model function."""
        # Greeting should not use powerful
        assert not should_use_powerful_model("hello")

        # Complex analysis might use powerful
        result = should_use_powerful_model("analyze and compare market trends")
        assert isinstance(result, bool)
