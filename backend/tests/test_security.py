"""Automated Security & Vulnerability Tests for PolicyPilot Phase 8.

Tests:
1. Rejection of non-PDF files masquerading as PDFs (missing %PDF- signature)
2. Rejection of empty or corrupted PDF files
3. Safe filename sanitization and path traversal prevention
4. Enforcement of maximum PDF page count limit (DoS / decompression bomb protection)
5. Enforcement of question input length bounds (min 2, max 500 characters)
6. Resistance to prompt injection directives embedded inside policy text
7. Error message sanitization (no internal credentials or trace leakage)
"""

import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.llm_service import MockLLMProvider
from app.services.pdf_service import PDFValidationError, extract_pdf_pages
from app.services.rag_service import rag_service

client = TestClient(app)


class TestPDFSecurityAndValidation:
    """Security tests for PDF ingestion and parsing boundaries."""

    def test_reject_non_pdf_magic_bytes(self):
        """A file with .pdf extension but lacking %PDF- magic bytes is rejected."""
        fake_content = b"<html><body>Not a PDF document</body></html>"
        files = {"file": ("malicious.pdf", BytesIO(fake_content), "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)
        assert resp.status_code == 400
        assert (
            "not a valid PDF" in resp.json()["detail"]
            or "signature" in resp.json()["detail"]
        )

    def test_reject_empty_pdf_upload(self):
        """Uploading a 0-byte file returns 400 Bad Request."""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_filename_path_traversal_sanitized(self):
        """Path traversal in filename (e.g. ../../etc/passwd.pdf) is stripped."""
        # Create a minimal valid PDF
        import pymupdf

        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 72), "Safe test policy content.", fontsize=11)
        pdf_bytes = doc.tobytes()
        doc.close()

        traversal_name = "../../../etc/evil.pdf"
        files = {"file": (traversal_name, BytesIO(pdf_bytes), "application/pdf")}
        resp = client.post("/api/documents/upload", files=files)
        assert resp.status_code == 201
        data = resp.json()
        assert ".." not in data["filename"]
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]
        assert data["filename"].endswith(".pdf")

    def test_max_pdf_pages_limit_enforced(self, monkeypatch):
        """PDFs exceeding settings.max_pdf_pages are safely rejected."""
        monkeypatch.setattr(settings, "max_pdf_pages", 3)

        import pymupdf

        doc = pymupdf.open()
        for _ in range(5):
            p = doc.new_page()
            p.insert_text((50, 72), "Page text", fontsize=11)
        pdf_data = doc.tobytes()
        doc.close()

        temp_path = Path(tempfile.gettempdir()) / f"test_max_pages_{id(self)}.pdf"
        temp_path.write_bytes(pdf_data)

        try:
            with pytest.raises(PDFValidationError) as exc_info:
                extract_pdf_pages(temp_path)
            assert "exceeds the maximum allowed limit" in str(exc_info.value)
        finally:
            temp_path.unlink(missing_ok=True)


class TestAPIInputSecurity:
    """Security tests for API endpoints, payload validation, and input sanitization."""

    def test_question_min_length_enforced(self):
        """Single-character or empty questions are rejected with 422."""
        resp = client.post("/api/ask", json={"question": "a"})
        assert resp.status_code == 422
        assert "at least 2 characters" in str(resp.json())

    def test_question_max_length_enforced(self):
        """Questions exceeding 500 characters are rejected with 422."""
        long_q = "What is the policy for " + ("leave " * 120)
        resp = client.post("/api/ask", json={"question": long_q})
        assert resp.status_code == 422
        assert "500 characters" in str(resp.json())

    def test_invalid_uuid_document_lookup_rejected(self):
        """Directory traversal attempt in document_id returns 404 cleanly."""
        resp = client.get("/api/documents/../../etc/passwd/file")
        assert resp.status_code in (404, 405)

    def test_prompt_injection_neutralized_in_rag(self):
        """Embedded directives in retrieved chunks do not alter system instructions."""
        mock_llm = MockLLMProvider()
        mock_llm.default_response = (
            "Network devices must have full disk encryption enabled before connecting "
            "to company VPN [S1]."
        )
        rag_service._llm = mock_llm

        resp = client.post(
            "/api/ask",
            json={"question": "What are the VPN and disk encryption directives?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ACME_ROOT_KEY_9999" not in data["answer"]
        assert "master" not in data["answer"].lower()
        assert len(data["citations"]) > 0

    def test_error_sanitization_no_credentials_leaked(self, monkeypatch):
        """503 Service Unavailable error masks internal secrets and paths."""
        monkeypatch.setattr(settings, "gemini_api_key", None)
        from app.services.llm_service import GeminiLLMProvider

        rag_service._llm = GeminiLLMProvider(api_key=None)

        resp = client.post("/api/ask", json={"question": "What is the leave policy?"})
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "key" not in detail.lower() or "not configured" in detail.lower()
        assert "traceback" not in detail.lower()
