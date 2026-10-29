"""Automated tests for document ingestion, page extraction, and retrieval."""

import io
import uuid
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.storage_service import storage_service

client = TestClient(app)


def create_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to create an in-memory PDF with specified text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all tests operate within an isolated temporary storage directory."""
    temp_docs = tmp_path / "documents"
    temp_meta = tmp_path / "metadata"
    temp_chunks = tmp_path / "chunks"
    temp_docs.mkdir(parents=True, exist_ok=True)
    temp_meta.mkdir(parents=True, exist_ok=True)
    temp_chunks.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_service, "documents_dir", temp_docs)
    monkeypatch.setattr(storage_service, "metadata_dir", temp_meta)
    monkeypatch.setattr(storage_service, "chunks_dir", temp_chunks)


class TestDocumentUpload:
    """Tests for POST /api/documents/upload."""

    def test_upload_single_page_pdf_success(self) -> None:
        sample_text = (
            "PolicyPilot Leave Policy: Employees are entitled to 20 annual days."
        )
        pdf_bytes = create_pdf_bytes([sample_text])

        response = client.post(
            "/api/documents/upload",
            files={
                "file": ("leave_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert "document_id" in data
        assert data["filename"] == "leave_policy.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["total_pages"] == 1
        assert data["total_characters"] == len(data["pages"][0]["text"])
        assert data["ingestion_status"] == "completed"

        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_number"] == 1
        assert sample_text in data["pages"][0]["text"]
        assert data["pages"][0]["character_count"] == len(data["pages"][0]["text"])

        # Verify physical files exist in isolated storage
        pdf_file = storage_service.documents_dir / f"{data['document_id']}.pdf"
        meta_file = storage_service.metadata_dir / f"{data['document_id']}.json"
        assert pdf_file.exists()
        assert meta_file.exists()

    def test_upload_multi_page_pdf_preserves_page_numbers(self) -> None:
        pages = [
            "Section 1: Standard Working Hours and Flexibility.",
            "Section 2: Health Insurance, Dental, and Vision Coverage.",
            "Section 3: Travel Expense Reimbursement and Per Diem Rates.",
        ]
        pdf_bytes = create_pdf_bytes(pages)

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "employee_handbook.pdf",
                    io.BytesIO(pdf_bytes),
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert data["total_pages"] == 3
        assert len(data["pages"]) == 3

        for idx, expected_text in enumerate(pages):
            page_obj = data["pages"][idx]
            # Verify 1-based page numbering
            assert page_obj["page_number"] == idx + 1
            assert expected_text in page_obj["text"]
            assert page_obj["character_count"] == len(page_obj["text"])

    def test_page_with_no_extractable_text_represented(self) -> None:
        # Page 1 has text, Page 2 is empty, Page 3 has text
        pages = ["First Page Content", "", "Third Page Content"]
        pdf_bytes = create_pdf_bytes(pages)

        response = client.post(
            "/api/documents/upload",
            files={
                "file": ("mixed_pages.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["total_pages"] == 3
        assert len(data["pages"]) == 3

        # Page 2 must be represented with page_number=2 and empty text
        p2 = data["pages"][1]
        assert p2["page_number"] == 2
        assert p2["text"] == ""
        assert p2["character_count"] == 0

        # Page 3 must have page_number=3
        p3 = data["pages"][2]
        assert p3["page_number"] == 3
        assert "Third Page Content" in p3["text"]

    def test_image_only_or_blank_pdf_handled_per_policy(self) -> None:
        # 2 pages with no text (e.g. image-only or blank scan)
        pdf_bytes = create_pdf_bytes(["", ""])

        response = client.post(
            "/api/documents/upload",
            files={
                "file": ("scanned_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["total_pages"] == 2
        assert data["total_characters"] == 0
        assert "OCR not supported" in data["ingestion_status"]
        assert len(data["pages"]) == 2
        assert data["pages"][0]["page_number"] == 1
        assert data["pages"][0]["text"] == ""
        assert data["pages"][1]["page_number"] == 2
        assert data["pages"][1]["text"] == ""

    def test_reject_non_pdf_file_extension(self) -> None:
        response = client.post(
            "/api/documents/upload",
            files={
                "file": ("policy.txt", io.BytesIO(b"Just plain text"), "text/plain")
            },
        )
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]

    def test_reject_corrupted_pdf_content(self) -> None:
        # A file with .pdf extension but random corrupted bytes
        corrupted_bytes = (
            b"%PDF-1.4\nCorrupted content that cannot be parsed by PyMuPDF"
        )
        response = client.post(
            "/api/documents/upload",
            files={
                "file": ("corrupt.pdf", io.BytesIO(corrupted_bytes), "application/pdf")
            },
        )
        assert response.status_code == 400
        assert "could not be parsed" in response.json()["detail"]

    def test_reject_empty_file(self) -> None:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_reject_oversized_upload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Temporarily set max size to 200 bytes for test
        monkeypatch.setattr(settings, "max_upload_size_bytes", 200)

        large_text = "A" * 1000
        pdf_bytes = create_pdf_bytes([large_text])
        assert len(pdf_bytes) > 200

        response = client.post(
            "/api/documents/upload",
            files={"file": ("oversized.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 413
        assert "maximum allowed upload size" in response.json()["detail"]

        # Ensure no leftover files were kept in storage
        assert list(storage_service.documents_dir.glob("*.pdf")) == []
        assert list(storage_service.metadata_dir.glob("*.json")) == []


class TestDocumentRetrieval:
    """Tests for document listing, detail, and raw PDF file download."""

    def test_list_documents_returns_metadata_only(self) -> None:
        # Upload two documents
        doc1_bytes = create_pdf_bytes(["First document content"])
        doc2_bytes = create_pdf_bytes(["Second document content"])

        r1 = client.post(
            "/api/documents/upload",
            files={"file": ("doc1.pdf", io.BytesIO(doc1_bytes), "application/pdf")},
        )
        r2 = client.post(
            "/api/documents/upload",
            files={"file": ("doc2.pdf", io.BytesIO(doc2_bytes), "application/pdf")},
        )
        assert r1.status_code == 201
        assert r2.status_code == 201

        # Fetch listing
        list_res = client.get("/api/documents")
        assert list_res.status_code == 200
        data = list_res.json()

        assert data["total"] == 2
        assert len(data["documents"]) == 2

        # Check metadata fields present, but NO pages list in DocumentMetadata
        for doc_item in data["documents"]:
            assert "document_id" in doc_item
            assert "filename" in doc_item
            assert "total_pages" in doc_item
            assert "total_characters" in doc_item
            assert "pages" not in doc_item

    def test_get_document_detail_returns_pages(self) -> None:
        content = "Detailed policy text for review."
        pdf_bytes = create_pdf_bytes([content])

        upload_res = client.post(
            "/api/documents/upload",
            files={
                "file": ("detail_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )
        doc_id = upload_res.json()["document_id"]

        detail_res = client.get(f"/api/documents/{doc_id}")
        assert detail_res.status_code == 200
        data = detail_res.json()

        assert data["document_id"] == doc_id
        assert data["filename"] == "detail_test.pdf"
        assert len(data["pages"]) == 1
        assert content in data["pages"][0]["text"]

    def test_get_unknown_document_id_returns_404(self) -> None:
        random_uuid = str(uuid.uuid4())
        response = client.get(f"/api/documents/{random_uuid}")
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found."

    def test_get_document_file_serves_pdf(self) -> None:
        content = "Raw PDF file content test."
        pdf_bytes = create_pdf_bytes([content])

        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("file_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]

        file_res = client.get(f"/api/documents/{doc_id}/file")
        assert file_res.status_code == 200
        assert file_res.headers["content-type"] == "application/pdf"
        assert len(file_res.content) > 0
        assert file_res.content.startswith(b"%PDF")

    def test_path_traversal_attempts_blocked(self) -> None:
        # Invalid UUID / path traversal attempts
        for malicious_id in [
            "../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "invalid-id",
            "../../../file",
        ]:
            detail_res = client.get(f"/api/documents/{malicious_id}")
            assert detail_res.status_code == 404

            file_res = client.get(f"/api/documents/{malicious_id}/file")
            assert file_res.status_code == 404


class TestDocumentDelete:
    """Tests for DELETE /api/documents/{document_id}."""

    def test_delete_document_success(self) -> None:
        content = "Temporary policy document to be deleted."
        pdf_bytes = create_pdf_bytes([content])

        upload_res = client.post(
            "/api/documents/upload",
            files={
                "file": ("delete_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document_id"]

        # Confirm document exists before deletion
        assert client.get(f"/api/documents/{doc_id}").status_code == 200

        # Execute DELETE
        del_res = client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 200
        assert del_res.json()["document_id"] == doc_id
        assert "deleted" in del_res.json()["detail"].lower()

        # Confirm 404 on subsequent lookups
        assert client.get(f"/api/documents/{doc_id}").status_code == 404
        assert client.get(f"/api/documents/{doc_id}/file").status_code == 404
        assert client.get(f"/api/documents/{doc_id}/chunks").status_code == 404

    def test_delete_nonexistent_document_returns_404(self) -> None:
        random_uuid = str(uuid.uuid4())
        del_res = client.delete(f"/api/documents/{random_uuid}")
        assert del_res.status_code == 404
        assert del_res.json()["detail"] == "Document not found."

    def test_delete_invalid_uuid_returns_404(self) -> None:
        del_res = client.delete("/api/documents/invalid-uuid-123")
        assert del_res.status_code == 404
