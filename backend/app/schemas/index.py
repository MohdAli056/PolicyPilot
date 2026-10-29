"""Pydantic schemas for document vector indexing."""

from pydantic import BaseModel, Field


class IndexDocumentResponse(BaseModel):
    """Response schema for document indexing endpoint."""

    document_id: str = Field(..., description="UUID of the indexed document")
    indexed_chunks: int = Field(
        ..., description="Number of chunks successfully embedded and stored in Qdrant"
    )
    status: str = Field(..., description="Indexing status: 'completed' or 'empty'")
    message: str = Field(..., description="Human-readable indexing summary message")
