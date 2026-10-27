"""Document ingestion, retrieval, and file download router."""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas.chunk import DocumentChunkListResponse
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentMetadata,
)
from app.services.chunking_service import chunk_document
from app.services.pdf_service import (
    PDFValidationError,
    compute_sha256,
    extract_pdf_pages,
)
from app.services.storage_service import storage_service

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(file: UploadFile = File(...)) -> DocumentDetail:
    """Accept a PDF file, store it locally, extract its text page-by-page,

    and persist its metadata.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    # Sanitize client-supplied filename to prevent directory traversal
    clean_filename = Path(file.filename).name

    # Check extension
    if not clean_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    # Check declared content type (relaxed check,
    # but reject clearly non-pdf declared types)
    if (
        file.content_type
        and "pdf" not in file.content_type.lower()
        and file.content_type != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {file.content_type}. Expected PDF.",
        )

    # Generate internal UUID identifier
    document_id = str(uuid.uuid4())
    temp_target_path = storage_service.documents_dir / f"{document_id}.pdf"

    bytes_written = 0
    max_size = settings.max_upload_size_bytes

    try:
        # Stream file to local storage in chunks to prevent unbounded memory consumption
        with open(temp_target_path, "wb") as out_file:
            while chunk := await file.read(65536):
                bytes_written += len(chunk)
                if bytes_written > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            "File exceeds maximum allowed upload size "
                            f"({max_size} bytes)."
                        ),
                    )
                out_file.write(chunk)

        if bytes_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty.",
            )

        # Run synchronous PyMuPDF extraction in threadpool to avoid blocking event loop
        try:
            pages, total_pages, total_chars, ingestion_status = await asyncio.to_thread(
                extract_pdf_pages, temp_target_path
            )
        except PDFValidationError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from PDF document.",
            ) from exc

        # Calculate content hash for audit / future deduplication
        sha256_hash = await asyncio.to_thread(compute_sha256, temp_target_path)

        doc_detail = DocumentDetail(
            document_id=document_id,
            filename=clean_filename,
            content_type=file.content_type or "application/pdf",
            file_size_bytes=bytes_written,
            total_pages=total_pages,
            total_characters=total_chars,
            sha256_hash=sha256_hash,
            ingestion_status=ingestion_status,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            pages=pages,
        )

        # Persist document metadata and page extraction atomically
        storage_service.save_document(doc_detail)

        # Generate structured text chunks and persist them
        chunks = await asyncio.to_thread(chunk_document, document_id, pages)
        storage_service.save_chunks(document_id, chunks)

        return doc_detail

    except HTTPException:
        # Clean up partial or rejected PDF file and chunks
        storage_service.cleanup_document(document_id)
        raise

    except Exception as exc:
        storage_service.cleanup_document(document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the document.",
        ) from exc

    finally:
        await file.close()


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents metadata",
)
async def list_documents() -> DocumentListResponse:
    """Return all ingested documents metadata (without full page text)."""
    docs: list[DocumentMetadata] = storage_service.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get document details and extracted page text",
)
async def get_document(document_id: str) -> DocumentDetail:
    """Return metadata and page-by-page extraction results for a specific document."""
    if not storage_service.validate_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    doc = storage_service.get_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return doc


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunkListResponse,
    summary="Get structured text chunks for a document",
)
async def get_document_chunks(document_id: str) -> DocumentChunkListResponse:
    """Return sequential text chunks with section and page attribution metadata."""
    if not storage_service.validate_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    doc = storage_service.get_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    chunks = storage_service.get_chunks(document_id)
    if chunks is None:
        # Generate on-demand if not pre-persisted
        chunks = await asyncio.to_thread(chunk_document, document_id, doc.pages)
        storage_service.save_chunks(document_id, chunks)

    return DocumentChunkListResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/{document_id}/file",
    summary="Download or stream original PDF file",
)
async def get_document_file(document_id: str) -> FileResponse:
    """Stream the original PDF document safely."""
    if not storage_service.validate_document_id(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    pdf_path = storage_service.get_document_pdf_path(document_id)
    if pdf_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found.",
        )

    doc = storage_service.get_document(document_id)
    display_filename = doc.filename if doc else f"{document_id}.pdf"

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=display_filename,
    )
