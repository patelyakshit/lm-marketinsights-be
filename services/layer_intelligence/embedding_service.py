"""
Embedding Service for Layer Intelligence System.

Supports multiple embedding providers:
- Cohere (recommended for production)
- OpenAI
- Google (Gemini)
- Local sentence-transformers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional
import hashlib

from decouple import config as env_config

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider - recommended for production."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
    ):
        self.api_key = api_key or env_config("COHERE_API_KEY", default="")
        self.model = model
        self._client = None
        self._dimension = 1024  # Cohere embed-v3 dimension

    @property
    def client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.AsyncClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("cohere package not installed. Run: pip install cohere")
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            response = await self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
            )
            return [list(emb) for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query (different input_type)."""
        try:
            response = await self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
            )
            return list(response.embeddings[0])
        except Exception as e:
            logger.error(f"Cohere query embedding error: {e}")
            raise


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or env_config("OPENAI_API_KEY", default="")
        self.model = model
        self._client = None

        # Dimensions by model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
    ):
        self.api_key = api_key or env_config("GOOGLE_API_KEY", default="")
        self.model = model
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.model}",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])

            return embeddings
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Run in thread pool since sentence-transformers is sync
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts).tolist()
        )
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []


class EmbeddingService:
    """
    Main embedding service with caching and batching.

    Provides a unified interface for generating embeddings
    regardless of the underlying provider.
    """

    def __init__(
        self,
        provider: str = "cohere",
        cache_enabled: bool = True,
        batch_size: int = 96,
    ):
        self.provider_name = provider
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size

        # Initialize provider
        self._provider = self._create_provider(provider)

        # Simple in-memory cache
        self._cache: dict[str, list[float]] = {}

    def _create_provider(self, provider: str) -> BaseEmbeddingProvider:
        """Create the appropriate embedding provider."""
        providers = {
            "cohere": CohereEmbeddingProvider,
            "openai": OpenAIEmbeddingProvider,
            "gemini": GeminiEmbeddingProvider,
            "local": LocalEmbeddingProvider,
        }

        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

        return providers[provider]()

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current provider."""
        return self._provider.dimension

    def _cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with caching and batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache and identify uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if self.cache_enabled:
                cache_key = self._cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    continue

            uncached_indices.append(i)
            uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            all_embeddings = []

            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i + self.batch_size]
                batch_embeddings = await self._provider.embed(batch)
                all_embeddings.extend(batch_embeddings)

                # Small delay between batches to respect rate limits
                if i + self.batch_size < len(uncached_texts):
                    await asyncio.sleep(0.1)

            # Store results and update cache
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                if self.cache_enabled:
                    cache_key = self._cache_key(uncached_texts[uncached_indices.index(idx)])
                    self._cache[cache_key] = embedding

        return results

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a search query.

        Some providers (like Cohere) differentiate between
        document and query embeddings for better retrieval.
        """
        if hasattr(self._provider, "embed_query"):
            return await self._provider.embed_query(text)
        return await self.embed_single(text)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "provider": self.provider_name,
            "dimension": self.dimension,
        }


# =============================================================================
# Factory Functions
# =============================================================================

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(
    provider: Optional[str] = None,
    force_new: bool = False,
) -> EmbeddingService:
    """Get or create the global embedding service."""
    global _embedding_service

    if _embedding_service is None or force_new:
        provider = provider or env_config("EMBEDDING_PROVIDER", default="cohere")
        _embedding_service = EmbeddingService(provider=provider)

    return _embedding_service


def set_embedding_service(service: EmbeddingService):
    """Set the global embedding service."""
    global _embedding_service
    _embedding_service = service
