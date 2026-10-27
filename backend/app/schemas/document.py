"""Pydantic schemas for document ingestion and retrieval."""

from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    """Extraction details for a single page within a PDF document."""

    page_number: int = Field(
        ..., description="One-based page number matching original PDF"
    )
    text: str = Field(..., description="Extracted raw text content of the page")
    character_count: int = Field(..., description="Character count for this page")


class DocumentMetadata(BaseModel):
    """Metadata representing an ingested document without full page texts."""

    document_id: str = Field(
        ..., description="Unique identifier for the document (UUID)"
    )
    filename: str = Field(..., description="Original filename for display")
    content_type: str = Field(..., description="MIME content type")
    file_size_bytes: int = Field(..., description="Size of the uploaded file in bytes")
    total_pages: int = Field(..., description="Total number of pages in the document")
    total_characters: int = Field(
        ..., description="Total character count across all pages"
    )
    sha256_hash: str = Field(
        ..., description="SHA-256 content hash for duplicate detection"
    )
    ingestion_status: str = Field(
        ..., description="Status of the ingestion and extraction"
    )
    uploaded_at: str = Field(..., description="ISO-8601 UTC upload timestamp")


class DocumentDetail(DocumentMetadata):
    """Full document representation including page-by-page extraction."""

    pages: list[PageExtraction] = Field(
        default_factory=list, description="Extracted pages in sequence"
    )


class DocumentListResponse(BaseModel):
    """Response schema for document listing."""

    documents: list[DocumentMetadata] = Field(
        default_factory=list, description="List of ingested document metadata"
    )
    total: int = Field(..., description="Total number of ingested documents")
