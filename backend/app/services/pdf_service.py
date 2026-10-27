"""PDF validation and page-preserving text extraction service.

Uses PyMuPDF to extract text on a per-page basis while preserving
1-based page indices matching standard PDF viewers.
"""

import hashlib
from pathlib import Path
from typing import Tuple

import pymupdf

from app.schemas.document import PageExtraction


class PDFValidationError(Exception):
    """Raised when an uploaded PDF fails validation (corrupt, encrypted, etc.)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_pdf_pages(file_path: Path) -> Tuple[list[PageExtraction], int, int, str]:
    """Extract text from a PDF file page by page.

    Args:
        file_path: Path to the local PDF file.

    Returns:
        Tuple of:
            - pages: list of PageExtraction objects with 1-based page numbers
            - total_pages: count of pages
            - total_characters: total characters across all pages
            - ingestion_status: descriptive status string

    Raises:
        PDFValidationError: If the PDF cannot be opened, is encrypted, has no pages,
                           or is corrupted.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise PDFValidationError(
            "The uploaded file is empty or missing.", status_code=400
        )

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:
        raise PDFValidationError(
            "The file could not be parsed as a valid PDF document.",
            status_code=400,
        ) from exc

    try:
        # Check if the PDF is encrypted / password-protected
        if doc.is_encrypted or doc.needs_pass:
            raise PDFValidationError(
                "Password-protected or encrypted PDFs are not supported.",
                status_code=400,
            )

        total_pages = doc.page_count
        if total_pages == 0:
            raise PDFValidationError(
                "The PDF document contains no pages.",
                status_code=400,
            )

        pages: list[PageExtraction] = []
        total_characters = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            # Extract plain text from page
            # Note: page_number is 1-based matching human-facing PDF page numbers
            raw_text = page.get_text() or ""
            # Count characters consistently using Python string length
            char_count = len(raw_text)
            total_characters += char_count

            pages.append(
                PageExtraction(
                    page_number=page_idx + 1,
                    text=raw_text,
                    character_count=char_count,
                )
            )

        if total_characters == 0:
            ingestion_status = (
                "completed (no extractable text found; OCR not supported in Phase 2)"
            )
        else:
            ingestion_status = "completed"

        return pages, total_pages, total_characters, ingestion_status

    finally:
        doc.close()
