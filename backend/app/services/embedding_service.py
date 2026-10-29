"""Dedicated embedding provider abstraction and implementations."""

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings

# Suppress HuggingFace symlink warning on Windows machines
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding model providers."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider name (e.g. 'fastembed', 'mock')."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of document chunk texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single search query."""
        ...


class FastEmbedProvider(BaseEmbeddingProvider):
    """Local, lightweight ONNX-based embedding provider using fastembed."""

    _instance: Optional["FastEmbedProvider"] = None
    _model_cache: dict[str, object] = {}

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 32,
    ):
        super().__init__(model_name=model_name, dimension=dimension)
        self.batch_size = batch_size
        self._model = None

    @property
    def provider_name(self) -> str:
        return "fastembed"

    def _get_model(self):
        """Lazily load and cache the FastEmbed TextEmbedding model singleton."""
        if self.model_name not in self._model_cache:
            try:
                from fastembed import TextEmbedding

                self._model_cache[self.model_name] = TextEmbedding(
                    model_name=self.model_name
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize FastEmbed model '{self.model_name}': {e}"
                ) from e
        return self._model_cache[self.model_name]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings in batches using FastEmbed."""
        if not texts:
            return []

        model = self._get_model()
        results: list[list[float]] = []

        try:
            # fastembed's embed() returns a generator of numpy arrays
            generator = model.embed(texts, batch_size=self.batch_size)
            for vec in generator:
                vec_list = [float(x) for x in vec]
                if len(vec_list) != self.dimension:
                    raise ValueError(
                        f"FastEmbed returned vector of dimension {len(vec_list)}, "
                        f"expected {self.dimension}"
                    )
                results.append(vec_list)
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings via FastEmbed: {e}") from e

        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        embeddings = self.embed_texts([query])
        if not embeddings:
            raise RuntimeError("FastEmbed returned no embeddings for query")
        return embeddings[0]


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, offline embedding provider for automated testing.

    Generates reproducible unit-normalized vectors without downloading weights
    or making network requests.
    """

    def __init__(
        self,
        model_name: str = "mock-embedding-v1",
        dimension: int = 384,
    ):
        super().__init__(model_name=model_name, dimension=dimension)

    @property
    def provider_name(self) -> str:
        return "mock"

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized pseudo-vector from text hash."""
        # Use sha256 to create a seed
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed_ints = [int(b) for b in h]

        raw_vector: list[float] = []
        for i in range(self.dimension):
            # Deterministic combination of seed and index
            val = math.sin(
                seed_ints[i % len(seed_ints)] * (i + 1)
                + (seed_ints[(i * 3) % len(seed_ints)] * 0.1)
            )
            raw_vector.append(val)

        # L2-normalize
        norm = math.sqrt(sum(x * x for x in raw_vector))
        if norm > 0:
            return [x / norm for x in raw_vector]
        return [0.0] * self.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._generate_vector(query)


# Module-level provider instance
_active_provider: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider(
    provider_type: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Return the active embedding provider based on configuration or override."""
    global _active_provider
    target_provider = provider_type or settings.embedding_provider

    if _active_provider is not None:
        if _active_provider.provider_name.lower() == target_provider.lower():
            return _active_provider

    if target_provider.lower() == "fastembed":
        _active_provider = FastEmbedProvider(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
        )
    elif target_provider.lower() == "mock":
        _active_provider = MockEmbeddingProvider(
            model_name="mock-embedding-v1",
            dimension=settings.embedding_dimension,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {target_provider}")

    return _active_provider


def set_embedding_provider(provider: Optional[BaseEmbeddingProvider]) -> None:
    """Explicitly set or reset the active embedding provider (useful for tests)."""
    global _active_provider
    _active_provider = provider
