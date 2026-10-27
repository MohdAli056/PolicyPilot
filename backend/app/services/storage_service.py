"""Local document and metadata storage management.

Handles writing raw PDFs and metadata records to local disk using
server-generated UUIDs, protecting against path traversal and partial writes.
"""

import json
import re
from pathlib import Path
from typing import Optional

from app.config import Settings, settings
from app.schemas.document import DocumentDetail, DocumentMetadata

UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class StorageService:
    """Service for managing PDF and metadata persistence on the local filesystem."""

    def __init__(self, custom_settings: Optional[Settings] = None):
        self.settings = custom_settings or settings
        self.documents_dir = self.settings.get_documents_path()
        self.metadata_dir = self.settings.get_metadata_path()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def validate_document_id(self, document_id: str) -> bool:
        """Check if document_id is a valid UUID format to prevent path traversal."""
        return bool(UUID_REGEX.match(document_id))

    def get_document_pdf_path(self, document_id: str) -> Optional[Path]:
        """Return the path to the stored PDF file if valid and existing."""
        if not self.validate_document_id(document_id):
            return None
        pdf_path = self.documents_dir / f"{document_id}.pdf"
        # Ensure path stays within documents_dir
        try:
            resolved = pdf_path.resolve()
            if not resolved.is_relative_to(self.documents_dir.resolve()):
                return None
            if resolved.exists() and resolved.is_file():
                return resolved
        except (ValueError, RuntimeError):
            return None
        return None

    def get_metadata_path(self, document_id: str) -> Optional[Path]:
        """Return the path to the stored metadata JSON file if valid and existing."""
        if not self.validate_document_id(document_id):
            return None
        meta_path = self.metadata_dir / f"{document_id}.json"
        try:
            resolved = meta_path.resolve()
            if not resolved.is_relative_to(self.metadata_dir.resolve()):
                return None
            if resolved.exists() and resolved.is_file():
                return resolved
        except (ValueError, RuntimeError):
            return None
        return None

    def save_document(self, document: DocumentDetail) -> None:
        """Atomically persist document detail and metadata as JSON."""
        if not self.validate_document_id(document.document_id):
            raise ValueError(f"Invalid document ID format: {document.document_id}")

        target_file = self.metadata_dir / f"{document.document_id}.json"
        temp_file = self.metadata_dir / f"{document.document_id}.tmp"

        payload = document.model_dump()

        # Write to temporary file first, then atomic replace
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        temp_file.replace(target_file)

    def get_document(self, document_id: str) -> Optional[DocumentDetail]:
        """Retrieve full document detail including page extractions."""
        path = self.get_metadata_path(document_id)
        if not path:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DocumentDetail.model_validate(data)
        except Exception:
            return None

    def list_documents(self) -> list[DocumentMetadata]:
        """Retrieve all ingested documents metadata, sorted by uploaded_at."""
        documents: list[DocumentMetadata] = []

        if not self.metadata_dir.exists():
            return documents

        for json_file in self.metadata_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Exclude full page extractions when reading for listing
                data.pop("pages", None)
                documents.append(DocumentMetadata.model_validate(data))
            except Exception:
                # Skip any unparseable / corrupted metadata files gracefully
                continue

        # Sort descending by upload timestamp (newest first)
        documents.sort(key=lambda d: d.uploaded_at, reverse=True)
        return documents

    def cleanup_document(self, document_id: str) -> None:
        """Clean up PDF and metadata files for a given document ID."""
        if not self.validate_document_id(document_id):
            return

        pdf_path = self.documents_dir / f"{document_id}.pdf"
        meta_path = self.metadata_dir / f"{document_id}.json"
        temp_meta = self.metadata_dir / f"{document_id}.tmp"

        for p in (pdf_path, meta_path, temp_meta):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


# Global storage service instance using application settings
storage_service = StorageService()
