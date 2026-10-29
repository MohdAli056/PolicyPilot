"""Pydantic schemas for BM25 keyword search and RRF hybrid retrieval."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class HybridSearchRequest(BaseModel):
    """Request payload for hybrid retrieval (semantic + BM25 with RRF)."""

    query: str = Field(..., description="User search query string")
    limit: Optional[int] = Field(
        None, description="Maximum number of fused results to retrieve"
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

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"RRF constant k must be strictly positive, got {v}")
        return v


class HybridSearchResultItem(BaseModel):
    """A single retrieved chunk with vector and keyword ranks and RRF fused score."""

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
        description="1-based rank from vector retriever, or null if not returned",
    )
    keyword_rank: Optional[int] = Field(
        None,
        description="1-based rank from BM25 retriever, or null if not returned",
    )
    rrf_score: float = Field(
        ..., description="Fused reciprocal rank score (higher is more relevant)"
    )
    retrievers: list[str] = Field(
        ..., description="List of retrieval systems that retrieved this chunk"
    )


class HybridSearchResponse(BaseModel):
    """Response schema for hybrid retrieval."""

    query: str = Field(..., description="The query string that was executed")
    total_results: int = Field(..., description="Number of retrieved results returned")
    rrf_k: int = Field(..., description="RRF constant k used in rank fusion")
    results: list[HybridSearchResultItem] = Field(
        default_factory=list,
        description="Ordered list of results sorted by fused relevance",
    )


class KeywordSearchResponse(BaseModel):
    """Response schema for direct BM25 keyword search."""

    query: str = Field(..., description="The query string that was executed")
    total_results: int = Field(..., description="Number of retrieved results returned")
    results: list[HybridSearchResultItem] = Field(
        default_factory=list,
        description="Ordered list of keyword search results",
    )
