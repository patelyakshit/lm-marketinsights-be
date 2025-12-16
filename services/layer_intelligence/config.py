"""
Configuration for Layer Intelligence System.
"""

from dataclasses import dataclass, field
from typing import Optional
from decouple import config as env_config


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False

    # Collection settings
    layer_collection: str = "layer_catalog"
    field_collection: str = "layer_fields"
    query_collection: str = "query_patterns"

    # Vector settings
    vector_size: int = 1024  # Cohere embed-v4 dimension
    distance: str = "Cosine"


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""
    provider: str = "cohere"  # cohere, openai, local
    model: str = "embed-english-v3.0"  # Cohere model

    # OpenAI alternative
    openai_model: str = "text-embedding-3-small"

    # Batch settings
    batch_size: int = 96
    rate_limit_delay: float = 0.1


@dataclass
class KnowledgeGraphConfig:
    """Knowledge graph configuration."""
    backend: str = "networkx"  # networkx, falkordb, neo4j

    # FalkorDB settings (for production)
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_graph_name: str = "layer_intelligence"

    # Neo4j settings (enterprise alternative)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""


@dataclass
class ArcGISConfig:
    """ArcGIS connection configuration."""
    portal_url: str = "https://locaitionmatters.maps.arcgis.com"
    username: str = ""
    password: str = ""
    api_key: str = ""

    # Default group for curated layers
    curated_layers_group_id: str = "c31bcfb83f2840538314457e2e6f9b8e"


@dataclass
class LLMConfig:
    """LLM configuration for semantic enrichment."""
    provider: str = "gemini"  # gemini, openai
    model: str = "gemini-2.0-flash"

    # Prompts
    field_description_prompt: str = """Generate a clear, concise description for this database field.

Field Name: {field_name}
Field Alias: {field_alias}
Field Type: {field_type}
Layer Name: {layer_name}
Layer Description: {layer_description}

Provide a 1-2 sentence description explaining what this field contains and how it might be used.
Focus on the semantic meaning, not technical details."""

    layer_enrichment_prompt: str = """Analyze this ArcGIS layer and provide semantic enrichment.

Layer Name: {layer_name}
Original Description: {description}
Fields: {fields}
Tags: {tags}

Provide:
1. An enhanced description (2-3 sentences)
2. A category (one of: demographics, residential, commercial, transportation, environmental, boundaries, infrastructure)
3. Semantic tags (5-10 relevant keywords)
4. Common query examples (3-5 natural language questions users might ask)

Format as JSON."""


@dataclass
class LayerIntelligenceConfig:
    """Main configuration for Layer Intelligence System."""

    # Sub-configurations
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    arcgis: ArcGISConfig = field(default_factory=ArcGISConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Search settings
    default_search_limit: int = 5
    min_similarity_score: float = 0.3

    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour

    # Sync settings
    auto_sync_interval_hours: int = 24

    @classmethod
    def from_env(cls) -> "LayerIntelligenceConfig":
        """Load configuration from environment variables."""
        return cls(
            qdrant=QdrantConfig(
                host=env_config("QDRANT_HOST", default="localhost"),
                port=env_config("QDRANT_PORT", default=6333, cast=int),
                api_key=env_config("QDRANT_API_KEY", default=None),
            ),
            embedding=EmbeddingConfig(
                provider=env_config("EMBEDDING_PROVIDER", default="cohere"),
                model=env_config("COHERE_EMBED_MODEL", default="embed-english-v3.0"),
            ),
            knowledge_graph=KnowledgeGraphConfig(
                backend=env_config("KG_BACKEND", default="networkx"),
                falkordb_host=env_config("FALKORDB_HOST", default="localhost"),
                falkordb_port=env_config("FALKORDB_PORT", default=6379, cast=int),
            ),
            arcgis=ArcGISConfig(
                portal_url=env_config("ARCGIS_PORTAL_URL", default="https://locaitionmatters.maps.arcgis.com"),
                username=env_config("ARCGIS_USERNAME", default=""),
                password=env_config("ARCGIS_PASSWORD", default=""),
                api_key=env_config("ARCGIS_API_KEY", default=""),
                curated_layers_group_id=env_config("CURATED_LAYERS_GROUP_ID", default="c31bcfb83f2840538314457e2e6f9b8e"),
            ),
        )


# Global configuration instance
_config: Optional[LayerIntelligenceConfig] = None


def get_config() -> LayerIntelligenceConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = LayerIntelligenceConfig.from_env()
    return _config


def set_config(config: LayerIntelligenceConfig):
    """Set the global configuration instance."""
    global _config
    _config = config
