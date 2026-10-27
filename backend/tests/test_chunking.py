"""Automated tests for document chunking, section detection, and chunk retrieval API."""

import io
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.document import PageExtraction
from app.services.chunking_service import chunk_document, detect_heading
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


class TestHeadingDetection:
    """Tests for heuristic heading and section detection."""

    def test_detect_markdown_headings(self) -> None:
        assert detect_heading("# Annual Leave Policy") == "Annual Leave Policy"
        assert detect_heading("## Medical Benefits") == "Medical Benefits"
        assert detect_heading("### Travel Guidelines") == "Travel Guidelines"

    def test_detect_numbered_headings(self) -> None:
        assert detect_heading("1. Purpose") == "1. Purpose"
        assert detect_heading("2.1 Eligibility Criteria") == "2.1 Eligibility Criteria"
        assert detect_heading("3.4.1 Travel Expenses") == "3.4.1 Travel Expenses"

    def test_detect_keyword_headings(self) -> None:
        assert detect_heading("Section 1: Working Hours") == "Section 1: Working Hours"
        assert (
            detect_heading("Chapter 4: Disciplinary Action")
            == "Chapter 4: Disciplinary Action"
        )
        assert (
            detect_heading("Article 2 - Non-Disclosure") == "Article 2 - Non-Disclosure"
        )

    def test_detect_all_caps_headings(self) -> None:
        assert detect_heading("CODE OF CONDUCT") == "CODE OF CONDUCT"
        assert (
            detect_heading("EXPENSE REIMBURSEMENT POLICY")
            == "EXPENSE REIMBURSEMENT POLICY"
        )

    def test_non_headings_return_none(self) -> None:
        assert detect_heading("This is just regular sentence body text.") is None
        assert detect_heading("123") is None
        assert detect_heading("") is None
        assert detect_heading("   ") is None


class TestChunkingAlgorithm:
    """Unit tests for the chunk_document function."""

    def test_short_text_produces_single_chunk(self) -> None:
        text = "This is a short policy statement for employees."
        pages = [PageExtraction(page_number=1, text=text, character_count=len(text))]

        chunks = chunk_document(
            "doc-1", pages, target_chunk_size=1000, chunk_overlap=100
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.text == text
        assert chunk.page_numbers == [1]
        assert chunk.character_count == len(text)
        assert chunk.section is None

    def test_long_text_splits_into_multiple_chunks_with_overlap(self) -> None:
        # Create a document of 2500 characters
        paragraphs = [
            f"Paragraph {i}: "
            + ("Content describing company rules and compliance. " * 5)
            for i in range(1, 10)
        ]
        full_text = "\n\n".join(paragraphs)
        pages = [
            PageExtraction(
                page_number=1, text=full_text, character_count=len(full_text)
            )
        ]

        target_size = 500
        overlap = 100
        chunks = chunk_document(
            "doc-2", pages, target_chunk_size=target_size, chunk_overlap=overlap
        )

        assert len(chunks) > 1

        # Check sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert len(chunk.text) > 0
            assert chunk.character_count == len(chunk.text)
            assert chunk.document_id == "doc-2"

        # Check overlap: successive chunks share text
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].text
            next_text = chunks[i + 1].text
            # Some trailing words/tokens of curr_text should appear in next_text
            # Extract last word from curr_text
            last_words = curr_text.split()[-3:]
            assert any(word in next_text for word in last_words)

    def test_paragraph_boundaries_preferred(self) -> None:
        para1 = "Short paragraph one."
        para2 = "Short paragraph two."
        para3 = "Short paragraph three."
        full_text = f"{para1}\n\n{para2}\n\n{para3}"

        pages = [
            PageExtraction(
                page_number=1, text=full_text, character_count=len(full_text)
            )
        ]
        # Set target size to fit exactly two paragraphs
        target_size = len(para1) + len(para2) + 5
        chunks = chunk_document(
            "doc-3", pages, target_chunk_size=target_size, chunk_overlap=0
        )

        # Chunks should not split inside sentences when paragraphs fit
        assert para1 in chunks[0].text
        assert para2 in chunks[0].text

    def test_oversized_paragraph_split_without_loss(self) -> None:
        # Paragraph with 1200 characters and no double-newlines
        sentences = [f"Sentence number {i} with vital policy terms." for i in range(40)]
        oversized_para = " ".join(sentences)
        assert len(oversized_para) > 1000

        pages = [
            PageExtraction(
                page_number=1, text=oversized_para, character_count=len(oversized_para)
            )
        ]

        chunks = chunk_document("doc-4", pages, target_chunk_size=400, chunk_overlap=50)

        assert len(chunks) > 1
        # Verify no words are lost
        for sent in sentences[:5]:
            assert any(sent in c.text for c in chunks)

    def test_section_metadata_assigned_from_headings(self) -> None:
        text = (
            "Section 1: Working Hours\n"
            "Employees are required to work 40 hours per week.\n\n"
            "Section 2: Overtime Pay\n"
            "Overtime must be pre-approved by the direct manager."
        )
        pages = [PageExtraction(page_number=1, text=text, character_count=len(text))]

        chunks = chunk_document("doc-5", pages, target_chunk_size=120, chunk_overlap=0)

        assert len(chunks) >= 2
        assert chunks[0].section == "Section 1: Working Hours"
        assert any("Section 2: Overtime Pay" in (c.section or "") for c in chunks)

    def test_chunks_preserve_page_numbers_including_spanning_pages(self) -> None:
        page1_text = (
            "Page one content: Annual Leave guidelines.\n\nContinued to next page:"
        )
        page2_text = "Employees must submit leave requests 2 weeks in advance."

        pages = [
            PageExtraction(
                page_number=1, text=page1_text, character_count=len(page1_text)
            ),
            PageExtraction(
                page_number=2, text=page2_text, character_count=len(page2_text)
            ),
        ]

        # Target size large enough to hold both pages in one chunk
        chunks = chunk_document(
            "doc-6", pages, target_chunk_size=2000, chunk_overlap=50
        )

        assert len(chunks) == 1
        assert chunks[0].page_numbers == [1, 2]

    def test_no_empty_chunks_produced(self) -> None:
        pages = [
            PageExtraction(page_number=1, text="   \n\n   \n\n", character_count=0),
            PageExtraction(
                page_number=2, text="Actual text on page two.", character_count=24
            ),
            PageExtraction(page_number=3, text="", character_count=0),
        ]

        chunks = chunk_document("doc-7", pages, target_chunk_size=500, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].text == "Actual text on page two."
        assert chunks[0].page_numbers == [2]

    def test_empty_document_produces_empty_chunk_list(self) -> None:
        pages = [
            PageExtraction(page_number=1, text="", character_count=0),
            PageExtraction(page_number=2, text="  ", character_count=0),
        ]
        chunks = chunk_document("doc-empty", pages)
        assert chunks == []

    def test_invalid_chunk_configuration_rejected(self) -> None:
        pages = [PageExtraction(page_number=1, text="Sample text", character_count=11)]

        with pytest.raises(ValueError, match="target_chunk_size must be positive"):
            chunk_document("doc", pages, target_chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            chunk_document("doc", pages, target_chunk_size=500, chunk_overlap=-10)

        with pytest.raises(
            ValueError, match="chunk_overlap .* must be strictly less than"
        ):
            chunk_document("doc", pages, target_chunk_size=500, chunk_overlap=500)

        with pytest.raises(
            ValueError, match="chunk_overlap .* must be strictly less than"
        ):
            chunk_document("doc", pages, target_chunk_size=500, chunk_overlap=600)

    def test_deterministic_chunk_identity(self) -> None:
        pages = [
            PageExtraction(
                page_number=1,
                text="Deterministic chunk identity verification text.",
                character_count=46,
            )
        ]

        run1 = chunk_document("doc-det", pages, target_chunk_size=500, chunk_overlap=50)
        run2 = chunk_document("doc-det", pages, target_chunk_size=500, chunk_overlap=50)

        assert len(run1) == len(run2) == 1
        assert run1[0].chunk_id == run2[0].chunk_id

        # Different content must produce a different chunk_id
        diff_pages = [
            PageExtraction(
                page_number=1, text="Completely different text.", character_count=26
            )
        ]
        run3 = chunk_document(
            "doc-det", diff_pages, target_chunk_size=500, chunk_overlap=50
        )
        assert run3[0].chunk_id != run1[0].chunk_id


class TestChunkAPIIntegration:
    """Integration tests for chunk persistence and GET /api/documents/{id}/chunks."""

    def test_upload_generates_and_persists_chunks(self) -> None:
        pdf_bytes = create_pdf_bytes(
            [
                (
                    "Section 1: General Provisions\n"
                    "Employees are expected to adhere to standard conduct."
                ),
                (
                    "Section 2: Health Insurance\n"
                    "Comprehensive coverage is provided starting day 30."
                ),
            ]
        )

        upload_res = client.post(
            "/api/documents/upload",
            files={
                "file": ("chunk_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document_id"]

        # Fetch chunks through API
        chunks_res = client.get(f"/api/documents/{doc_id}/chunks")
        assert chunks_res.status_code == 200
        data = chunks_res.json()

        assert data["document_id"] == doc_id
        assert data["total_chunks"] >= 1
        assert len(data["chunks"]) == data["total_chunks"]

        # Verify chunk structure
        first_chunk = data["chunks"][0]
        assert "chunk_id" in first_chunk
        assert first_chunk["document_id"] == doc_id
        assert first_chunk["chunk_index"] == 0
        assert "General Provisions" in first_chunk["text"]
        assert first_chunk["page_numbers"] == [1, 2]
        assert first_chunk["character_count"] == len(first_chunk["text"])

        # Verify disk persistence in isolated temp storage
        chunk_file = storage_service.chunks_dir / f"{doc_id}.json"
        assert chunk_file.exists()

    def test_get_chunks_unknown_document_returns_404(self) -> None:
        res = client.get("/api/documents/non-existent-uuid/chunks")
        assert res.status_code == 404
        assert res.json()["detail"] == "Document not found."

    def test_get_chunks_for_empty_document(self) -> None:
        # Upload image-only / blank PDF
        blank_pdf = create_pdf_bytes(["", ""])
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": ("blank.pdf", io.BytesIO(blank_pdf), "application/pdf")},
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document_id"]

        chunks_res = client.get(f"/api/documents/{doc_id}/chunks")
        assert chunks_res.status_code == 200
        data = chunks_res.json()
        assert data["document_id"] == doc_id
        assert data["total_chunks"] == 0
        assert data["chunks"] == []
