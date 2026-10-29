"""Pydantic schemas for semantic vector search."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Request payload for semantic vector search."""

    query: str = Field(..., description="User search query string")
    limit: Optional[int] = Field(
        None, description="Maximum number of search results to retrieve"
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


class SearchResultItem(BaseModel):
    """A single retrieved chunk passage with source metadata and similarity score."""

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
    score: float = Field(
        ..., description="Cosine similarity score (higher is more relevant)"
    )


class SearchResponse(BaseModel):
    """Response schema for semantic vector search."""

    query: str = Field(..., description="The query string that was executed")
    total_results: int = Field(..., description="Number of retrieved results returned")
    results: list[SearchResultItem] = Field(
        default_factory=list, description="Ordered list of search results by relevance"
    )
