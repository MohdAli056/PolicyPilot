"""Pydantic schemas for Cross-Encoder Reranked Hybrid Retrieval."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RerankSearchRequest(BaseModel):
    """Request payload for hybrid retrieval with cross-encoder reranking."""

    query: str = Field(..., description="User search query string")
    limit: Optional[int] = Field(
        None, description="Maximum number of final reranked results to return"
    )
    candidate_limit: Optional[int] = Field(
        None,
        description="Number of top candidates from hybrid RRF to pass to the reranker",
    )
    rrf_k: Optional[int] = Field(
        None, description="Custom Reciprocal Rank Fusion constant k (default: 60)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        if len(stripped) > 2000:
            raise ValueError("Query exceeds maximum allowed length of 2000 characters")
        return stripped

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"Search limit must be positive, got {v}")
        return v

    @field_validator("candidate_limit")
    @classmethod
    def validate_candidate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"Candidate limit must be positive, got {v}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"RRF constant k must be strictly positive, got {v}")
        return v


class RerankSearchResultItem(BaseModel):
    """A retrieved chunk candidate scored and ranked by a cross-encoder model.

    Preserves all provenance and retrieval metadata (vector rank, BM25 rank,
    RRF score) alongside cross-encoder relevance metrics.
    """

    document_id: str = Field(..., description="UUID of the source document")
    filename: str = Field(..., description="Filename of the source document")
    chunk_id: str = Field(..., description="Deterministic unique identifier of chunk")
    chunk_index: int = Field(..., description="Zero-based index of chunk in document")
    text: str = Field(..., description="Extracted chunk text content")
    section: Optional[str] = Field(None, description="Detected section heading or None")
    page_numbers: list[int] = Field(
        ..., description="One-based source page numbers contributing text"
    )
    character_count: int = Field(..., description="Exact character count of chunk text")
    vector_rank: Optional[int] = Field(
        None,
        description="1-based rank from vector retriever, or null if unranked",
    )
    keyword_rank: Optional[int] = Field(
        None,
        description="1-based rank from BM25 retriever, or null if not in top BM25 hits",
    )
    rrf_score: float = Field(
        ...,
        description="Initial retrieval-stage Reciprocal Rank Fusion score",
    )
    retrievers: list[str] = Field(
        ...,
        description="Retrieval systems that surfaced this candidate ('vector', 'bm25')",
    )
    rerank_score: float = Field(
        ...,
        description="Cross-encoder relevance score (higher is more relevant)",
    )
    rerank_rank: int = Field(
        ...,
        description="1-based final rank after cross-encoder reordering",
    )


class RerankSearchResponse(BaseModel):
    """Response schema for cross-encoder reranked hybrid search."""

    query: str = Field(..., description="The query string that was executed")
    total_results: int = Field(..., description="Number of reranked results returned")
    reranker_model: str = Field(
        ..., description="Identifier of the cross-encoder model used"
    )
    candidate_limit: int = Field(
        ..., description="Number of candidates evaluated by the reranker"
    )
    results: list[RerankSearchResultItem] = Field(
        default_factory=list,
        description="Ordered list of candidates sorted by cross-encoder relevance",
    )
    timing_ms: Optional[dict[str, float]] = Field(
        None,
        description="Execution timing breakdown in milliseconds",
    )
