"""Pydantic schemas for document chunking and metadata."""

from typing import Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A deterministic text chunk derived from an ingested document."""

    chunk_id: str = Field(
        ..., description="Deterministic unique identifier for the chunk"
    )
    document_id: str = Field(..., description="UUID of the parent document")
    chunk_index: int = Field(
        ..., description="Sequential zero-based index of the chunk"
    )
    text: str = Field(..., description="Extracted chunk text content")
    section: Optional[str] = Field(
        None, description="Detected section or heading title, or None"
    )
    page_numbers: list[int] = Field(
        ..., description="One-based source page numbers contributing text"
    )
    character_count: int = Field(..., description="Exact character count of chunk text")


class DocumentChunkListResponse(BaseModel):
    """Response schema for document chunk retrieval."""

    document_id: str = Field(..., description="UUID of the parent document")
    total_chunks: int = Field(
        ..., description="Total number of chunks produced for the document"
    )
    chunks: list[DocumentChunk] = Field(
        default_factory=list, description="List of document chunks in order"
    )
