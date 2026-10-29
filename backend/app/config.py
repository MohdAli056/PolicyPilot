"""Application configuration loaded from environment variables."""

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the PolicyPilot backend.

    Values are loaded from environment variables or a .env file.
    """

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "PolicyPilot"
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS: comma-separated list of allowed origins
    cors_origins: str = "http://localhost:3000"

    # Document storage configuration
    documents_dir: str = "data/documents"
    metadata_dir: str = "data/metadata"
    chunks_dir: str = "data/chunks"
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB default

    # Document chunking configuration (in characters)
    target_chunk_size: int = 1000
    chunk_overlap: int = 150

    # Qdrant Vector Database configuration
    qdrant_path: str = "data/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "policypilot_chunks"

    # Embedding provider configuration
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    # Search limits
    default_search_limit: int = 5
    max_search_limit: int = 50

    # BM25 Keyword Search configuration
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid Search and Reciprocal Rank Fusion (RRF) configuration
    rrf_k: int = 60
    hybrid_candidate_limit: int = 20

    # Cross-Encoder Reranking configuration
    reranker_provider: str = "flashrank"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    rerank_candidate_limit: int = 20
    rerank_batch_size: int = 16

    # LLM Provider and RAG Orchestration configuration
    gemini_api_key: str | None = None
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    rag_max_evidence_chunks: int = 5
    rag_context_char_budget: int = 6000
    rag_abstention_score_threshold: float = 0.00001
    rag_request_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate chunking, vector, and search constraints."""
        if self.target_chunk_size <= 0:
            raise ValueError(
                f"target_chunk_size must be positive, got {self.target_chunk_size}"
            )
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap cannot be negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.target_chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less "
                f"than target_chunk_size ({self.target_chunk_size})"
            )
        if self.embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {self.embedding_dimension}"
            )
        if self.embedding_batch_size <= 0:
            raise ValueError(
                f"embedding_batch_size must be positive, "
                f"got {self.embedding_batch_size}"
            )
        if self.default_search_limit <= 0:
            raise ValueError(
                f"default_search_limit must be positive, "
                f"got {self.default_search_limit}"
            )
        if self.max_search_limit < self.default_search_limit:
            raise ValueError(
                f"max_search_limit ({self.max_search_limit}) must be >= "
                f"default_search_limit ({self.default_search_limit})"
            )
        if self.bm25_k1 <= 0:
            raise ValueError(f"bm25_k1 must be positive, got {self.bm25_k1}")
        if not (0.0 <= self.bm25_b <= 1.0):
            raise ValueError(f"bm25_b must be between 0.0 and 1.0, got {self.bm25_b}")
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {self.rrf_k}")
        if self.hybrid_candidate_limit <= 0:
            raise ValueError(
                f"hybrid_candidate_limit must be positive, "
                f"got {self.hybrid_candidate_limit}"
            )
        valid_providers = ("fastembed", "mock")
        if self.embedding_provider.lower() not in valid_providers:
            raise ValueError(
                f"embedding_provider '{self.embedding_provider}' is not supported. "
                f"Choose from: {', '.join(valid_providers)}"
            )
        if self.rerank_candidate_limit <= 0:
            raise ValueError(
                f"rerank_candidate_limit must be positive, "
                f"got {self.rerank_candidate_limit}"
            )
        if self.rerank_batch_size <= 0:
            raise ValueError(
                f"rerank_batch_size must be positive, got {self.rerank_batch_size}"
            )
        valid_rerankers = ("flashrank", "mock")
        if self.reranker_provider.lower() not in valid_rerankers:
            raise ValueError(
                f"reranker_provider '{self.reranker_provider}' is not supported. "
                f"Choose from: {', '.join(valid_rerankers)}"
            )
        valid_llm_providers = ("gemini", "mock")
        if self.llm_provider.lower() not in valid_llm_providers:
            raise ValueError(
                f"llm_provider '{self.llm_provider}' is not supported. "
                f"Choose from: {', '.join(valid_llm_providers)}"
            )
        if self.rag_max_evidence_chunks <= 0:
            raise ValueError(
                f"rag_max_evidence_chunks must be positive, "
                f"got {self.rag_max_evidence_chunks}"
            )
        if self.rag_context_char_budget <= 0:
            raise ValueError(
                f"rag_context_char_budget must be positive, "
                f"got {self.rag_context_char_budget}"
            )
        if self.rag_request_timeout_seconds <= 0:
            raise ValueError(
                f"rag_request_timeout_seconds must be positive, "
                f"got {self.rag_request_timeout_seconds}"
            )
        return self

    def get_documents_path(self):
        """Return resolved pathlib.Path for document storage."""
        from pathlib import Path

        p = Path(self.documents_dir)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()

    def get_metadata_path(self):
        """Return resolved pathlib.Path for metadata storage."""
        from pathlib import Path

        p = Path(self.metadata_dir)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()

    def get_chunks_path(self):
        """Return resolved pathlib.Path for chunk storage."""
        from pathlib import Path

        p = Path(self.chunks_dir)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()

    def get_qdrant_path(self):
        """Return resolved pathlib.Path for local Qdrant embedded storage."""
        from pathlib import Path

        p = Path(self.qdrant_path)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent
            p = backend_root / p
        return p.resolve()


settings = Settings()
