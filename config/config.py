"""
RabbitMQ Queue Configuration.

This module contains configuration settings for RabbitMQ connections,
queues, exchanges, and routing keys.
"""

from decouple import config
from google.adk.sessions import InMemorySessionService
from services.session_storage_service import SessionStorageService

STREAM_CHUNK_DELAY = config("STREAM_CHUNK_DELAY", default=9, cast=int) / 100

DATABASE_URL = config("DATABASE_URL")

# ArcGIS API Keys
GEOLOCATION_API_KEY = config("ARCGIS_GEOLOCATION_API_KEY", "")
ARCGIS_API_KEY = config("ARCGIS_API_KEY", "")  # For GeoEnrichment API
RABBITMQ_HOST = config("RABBITMQ_HOST", default="localhost")
RABBITMQ_PORT = config("RABBITMQ_PORT", default=5672, cast=int)
RABBITMQ_USER = config("RABBITMQ_USER", default="")
RABBITMQ_PASSWORD = config("RABBITMQ_PASSWORD", default="")
RABBITMQ_VHOST = config("RABBITMQ_VHOST", default="/")

# Connection URL - handle empty credentials for simple RabbitMQ setup
if RABBITMQ_USER and RABBITMQ_PASSWORD:
    RABBITMQ_URL = config(
        "RABBITMQ_URL",
        default=f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}{RABBITMQ_VHOST}",
    )
else:
    RABBITMQ_URL = config(
        "RABBITMQ_URL",
        default=f"amqp://{RABBITMQ_HOST}:{RABBITMQ_PORT}{RABBITMQ_VHOST}",
    )

# Exchange Configuration
DATABASE_OPERATIONS_EXCHANGE = "database_operations"
EXCHANGE_TYPE = "topic"


# Queue Names
CHAT_HISTORY_QUEUE = "chat_history_operations"
CHAT_INPUT_OUTPUT_QUEUE = "chat_input_output_operations"
DEAD_LETTER_QUEUE = "dlq_operations"


# Routing Keys
class RoutingKeys:
    # Chat History Operations
    CHAT_HISTORY_CREATE = "chat.history.create"
    CHAT_HISTORY_UPDATE = "chat.history.update"
    CHAT_HISTORY_DELETE = "chat.history.delete"

    # Chat Message Operations
    CHAT_MESSAGE_CREATE = "chat.message.create"
    CHAT_MESSAGE_UPDATE = "chat.message.update"
    CHAT_MESSAGE_DELETE = "chat.message.delete"


# Queue Configuration
QUEUE_CONFIG = {
    CHAT_HISTORY_QUEUE: {
        "durable": True,
        "auto_delete": False,
        "exclusive": False,
        "routing_keys": [
            RoutingKeys.CHAT_HISTORY_CREATE,
            RoutingKeys.CHAT_HISTORY_UPDATE,
            RoutingKeys.CHAT_HISTORY_DELETE,
        ],
    },
    CHAT_INPUT_OUTPUT_QUEUE: {
        "durable": True,
        "auto_delete": False,
        "exclusive": False,
        "routing_keys": [
            RoutingKeys.CHAT_MESSAGE_CREATE,
            RoutingKeys.CHAT_MESSAGE_UPDATE,
            RoutingKeys.CHAT_MESSAGE_DELETE,
        ],
    },
    DEAD_LETTER_QUEUE: {
        "durable": True,
        "auto_delete": False,
        "exclusive": False,
        "routing_keys": ["dlq.*"],
    },
}

# Connection Parameters
CONNECTION_PARAMS = {
    "host": RABBITMQ_HOST,
    "port": RABBITMQ_PORT,
    "virtual_host": RABBITMQ_VHOST,
    "credentials": {
        "username": RABBITMQ_USER if RABBITMQ_USER else None,
        "password": RABBITMQ_PASSWORD if RABBITMQ_PASSWORD else None,
    },
    "connection_attempts": 3,
    "retry_delay": 2,  # Reduced from 5s to 2s for faster reconnection
    "heartbeat": 60,  # Reduced from 600s to 60s - detect dead connections faster
    "blocked_connection_timeout": 300,
    "socket_timeout": 10,  # Add socket timeout for faster failure detection
}

# Message Properties
MESSAGE_PROPERTIES = {
    "delivery_mode": 2,  # Make messages persistent
    "content_type": "application/json",
    "content_encoding": "utf-8",
}

# Application Settings
APP_SETTINGS = {
    "source": "bumblebee_app",
    "version": "2.0.0",
    "environment": config("ENVIRONMENT", default="development"),
}

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = config("AZURE_STORAGE_CONNECTION_STRING", default="")
AZURE_STORAGE_ACCOUNT_NAME = config("AZURE_STORAGE_ACCOUNT_NAME", default="")
AZURE_STORAGE_ACCOUNT_KEY = config("AZURE_STORAGE_ACCOUNT_KEY", default="")
AZURE_STORAGE_CONTAINER_NAME = config(
    "AZURE_STORAGE_CONTAINER_NAME", default="salesforce-exports"
)
AZURE_STORAGE_SAS_EXPIRY_HOURS = config(
    "AZURE_STORAGE_SAS_EXPIRY_HOURS", default=24, cast=int
)

QDRANT_DB_PORT = config("QDRANT_DB_PORT", 6333, cast=int)
QDRANT_DB_URL = config("QDRANT_DB_URL", "localhost")
QDRANT_COLLECTION_NAME = config("QDRANT_COLLECTION_NAME", "")
VERTEX_AI_RAG_CORPUS = config("VERTEX_AI_RAG_CORPUS", default="")

# OpenAI Configuration
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")

# Vertex AI RAG Configuration
VERTEX_RAG_CORPUS_ID = config(
    "VERTEX_RAG_CORPUS_ID",
    default="projects/957243995509/locations/us-west4/ragCorpora/6917529027641081856",
)

# Custom async session storage service for persistence
database_session_service = SessionStorageService(
    database_url=DATABASE_URL,
    pool_size=10,  # Max persistent connections (for 1-10 concurrent users)
    max_overflow=10,  # Additional connections when pool exhausted
    pool_recycle=600,  # Recycle connections after 10 minutes (prevents stale connections)
    pool_pre_ping=True,  # Test connection health before use (critical!)
    pool_timeout=60,  # Wait 60s for available connection from pool
    connection_timeout=60,  # Wait 60s for establishing new connection (including ping)
    command_timeout=60,  # Wait 60s for individual SQL commands to complete
    echo=False,  # Disable SQL query logging (production)
)

# In-memory session service for fast processing during response generation
in_memory_session_service = InMemorySessionService()

# Keep existing session_service for backward compatibility
session_service = database_session_service


# Map configurations
ARCGIS_ENRICHMENT_URL = config(
    "ARCGIS_ENRICHMENT_URL",
    "https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/GeoEnrichment/enrich",
)
ARCGIS_GEOCODE_URL = config(
    "ARCGIS_GEOCODE_URL",
    "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates",
)
ARCGIS_REVERSE_GEOCODE_URL = config(
    "ARCGIS_REVERSE_GEOCODE_URL",
    "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode",
)
ARCGIS_PLACES_URL = config(
    "ARCGIS_PLACES_URL",
    "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/places/near-point",
)

ARCGIS_USERNAME = config("ARCGIS_USERNAME", None)
ARCGIS_PASSWORD = config("ARCGIS_PASSWORD", None)
