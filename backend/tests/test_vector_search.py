"""Automated tests for embeddings, Qdrant vector storage, and search."""

import io
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.document import DocumentDetail, PageExtraction
from app.services.embedding_service import (
    FastEmbedProvider,
    MockEmbeddingProvider,
    set_embedding_provider,
)
from app.services.indexing_service import IndexingService, indexing_service
from app.services.search_service import search_service
from app.services.storage_service import storage_service
from app.services.vector_service import (
    VectorService,
    reset_vector_service,
)

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


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def memory_vector_service() -> VectorService:
    """Return a VectorService backed by an isolated in-memory Qdrant instance."""
    q_client = QdrantClient(location=":memory:")
    service = VectorService(
        client=q_client,
        default_collection="test_collection",
        default_dimension=384,
    )
    service.ensure_collection()
    return service


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate filesystem storage and set up in-memory Qdrant and Mock embeddings."""
    temp_docs = tmp_path / "documents"
    temp_meta = tmp_path / "metadata"
    temp_chunks = tmp_path / "chunks"
    temp_docs.mkdir(parents=True, exist_ok=True)
    temp_meta.mkdir(parents=True, exist_ok=True)
    temp_chunks.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_service, "documents_dir", temp_docs)
    monkeypatch.setattr(storage_service, "metadata_dir", temp_meta)
    monkeypatch.setattr(storage_service, "chunks_dir", temp_chunks)

    # Use in-memory Qdrant for all default operations during testing
    test_qdrant_client = QdrantClient(location=":memory:")
    test_vector_service = VectorService(
        client=test_qdrant_client,
        default_collection="test_chunks",
        default_dimension=384,
    )
    test_vector_service.ensure_collection()

    test_embedding_provider = MockEmbeddingProvider(dimension=384)
    set_embedding_provider(test_embedding_provider)

    # Monkeypatch the global services so router endpoints use isolated in-memory store
    monkeypatch.setattr(indexing_service, "_custom_vector_service", test_vector_service)
    monkeypatch.setattr(
        indexing_service, "_custom_embedding_provider", test_embedding_provider
    )
    monkeypatch.setattr(search_service, "_custom_vector_service", test_vector_service)
    monkeypatch.setattr(
        search_service, "_custom_embedding_provider", test_embedding_provider
    )

    yield

    reset_vector_service()
    set_embedding_provider(None)


# ============================================================================
# Unit Tests: Embedding Provider Abstraction
# ============================================================================


class TestEmbeddingProvider:
    """Tests for embedding generation and abstraction."""

    def test_mock_provider_generates_normalized_vectors_with_correct_dimension(
        self, mock_provider: MockEmbeddingProvider
    ) -> None:
        texts = ["First policy chunk", "Second reimbursement chunk"]
        vectors = mock_provider.embed_texts(texts)

        assert len(vectors) == 2
        for vec in vectors:
            assert len(vec) == 384
            # Verify unit normalization: sum of squares ≈ 1.0
            norm = sum(x * x for x in vec)
            assert pytest.approx(norm, 0.001) == 1.0

    def test_mock_provider_deterministic_for_identical_text(
        self, mock_provider: MockEmbeddingProvider
    ) -> None:
        text = "Employees are entitled to 20 days paid leave."
        vec1 = mock_provider.embed_query(text)
        vec2 = mock_provider.embed_query(text)
        assert vec1 == vec2

    def test_mock_provider_different_for_different_text(
        self, mock_provider: MockEmbeddingProvider
    ) -> None:
        vec1 = mock_provider.embed_query("Annual leave policy guidelines")
        vec2 = mock_provider.embed_query("Travel reimbursement rules")
        assert vec1 != vec2

    def test_fastembed_provider_integration(self) -> None:
        """Integration test with the real FastEmbed local ONNX model."""
        provider = FastEmbedProvider(
            model_name="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        texts = ["Health insurance coverage policy."]
        vectors = provider.embed_texts(texts)

        assert len(vectors) == 1
        assert len(vectors[0]) == 384
        # Verify query embedding matches
        query_vec = provider.embed_query("Health insurance")
        assert len(query_vec) == 384


# ============================================================================
# Unit Tests: Vector Service & Qdrant Operations
# ============================================================================


class TestVectorService:
    """Tests for VectorService Qdrant collection management and operations."""

    def test_ensure_collection_creates_collection(
        self, memory_vector_service: VectorService
    ) -> None:
        assert memory_vector_service.client.collection_exists("test_collection")

    def test_ensure_collection_raises_on_incompatible_dimension(
        self, memory_vector_service: VectorService
    ) -> None:
        # Attempt to ensure existing 384-d collection with 512 dimensions
        with pytest.raises(ValueError, match="Incompatible vector dimension"):
            memory_vector_service.ensure_collection(
                collection_name="test_collection",
                dimension=512,
            )

    def test_upsert_chunks_and_count(
        self,
        memory_vector_service: VectorService,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        chunk = DocumentChunk(
            chunk_id="a1b2c3d4-e5f6-5a7b-8c9d-0e1f2a3b4c5d",
            document_id="doc-123",
            chunk_index=0,
            text="Employees get 20 days vacation.",
            section="Vacation Policy",
            page_numbers=[1],
            character_count=31,
        )
        embeddings = mock_provider.embed_texts([chunk.text])

        count = memory_vector_service.upsert_chunks(
            chunks=[chunk],
            embeddings=embeddings,
            filename="vacation.pdf",
        )
        assert count == 1
        assert memory_vector_service.count() == 1
        assert memory_vector_service.count(document_id="doc-123") == 1
        assert memory_vector_service.count(document_id="non-existent") == 0

    def test_vector_payload_preserves_chunk_metadata(
        self,
        memory_vector_service: VectorService,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        chunk = DocumentChunk(
            chunk_id="b2c3d4e5-f6a7-5b8c-9d0e-1f2a3b4c5d6e",
            document_id="doc-456",
            chunk_index=2,
            text="Spanning chunk across pages.",
            section="Section 3: Benefits",
            page_numbers=[1, 2],
            character_count=28,
        )
        embeddings = mock_provider.embed_texts([chunk.text])
        memory_vector_service.upsert_chunks(
            chunks=[chunk],
            embeddings=embeddings,
            filename="benefits.pdf",
        )

        results = memory_vector_service.search(
            query_vector=embeddings[0],
            limit=1,
        )
        assert len(results) == 1
        payload = results[0]["payload"]
        assert payload["document_id"] == "doc-456"
        assert payload["filename"] == "benefits.pdf"
        assert payload["chunk_id"] == "b2c3d4e5-f6a7-5b8c-9d0e-1f2a3b4c5d6e"
        assert payload["chunk_index"] == 2
        assert payload["text"] == "Spanning chunk across pages."
        assert payload["section"] == "Section 3: Benefits"
        assert payload["page_numbers"] == [1, 2]
        assert payload["character_count"] == 28

    def test_reindexing_idempotency_no_duplicate_points(
        self,
        memory_vector_service: VectorService,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        chunk = DocumentChunk(
            chunk_id="c3d4e5f6-a7b8-5c9d-0e1f-2a3b4c5d6e7f",
            document_id="doc-789",
            chunk_index=0,
            text="Reindexing should not duplicate vectors.",
            section=None,
            page_numbers=[1],
            character_count=40,
        )
        embeddings = mock_provider.embed_texts([chunk.text])

        # First upsert
        memory_vector_service.upsert_chunks([chunk], embeddings, "reindex.pdf")
        assert memory_vector_service.count() == 1

        # Second upsert of identical chunk
        memory_vector_service.upsert_chunks([chunk], embeddings, "reindex.pdf")
        assert memory_vector_service.count() == 1

    def test_delete_document_vectors(
        self,
        memory_vector_service: VectorService,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        chunks = [
            DocumentChunk(
                chunk_id="d1d1d1d1-d1d1-5d1d-8d1d-d1d1d1d1d1d1",
                document_id="doc-A",
                chunk_index=0,
                text="Doc A chunk.",
                section=None,
                page_numbers=[1],
                character_count=12,
            ),
            DocumentChunk(
                chunk_id="d2d2d2d2-d2d2-5d2d-8d2d-d2d2d2d2d2d2",
                document_id="doc-B",
                chunk_index=0,
                text="Doc B chunk.",
                section=None,
                page_numbers=[1],
                character_count=12,
            ),
        ]
        embeddings = mock_provider.embed_texts([c.text for c in chunks])
        memory_vector_service.upsert_chunks(chunks, embeddings, "multi.pdf")
        assert memory_vector_service.count() == 2

        memory_vector_service.delete_document_vectors("doc-A")
        assert memory_vector_service.count() == 1
        assert memory_vector_service.count(document_id="doc-B") == 1
        assert memory_vector_service.count(document_id="doc-A") == 0


# ============================================================================
# Unit Tests: Indexing Service
# ============================================================================


class TestIndexingService:
    """Tests for IndexingService business logic."""

    def test_index_document_unknown_id_raises_keyerror(
        self, memory_vector_service: VectorService, mock_provider: MockEmbeddingProvider
    ) -> None:
        svc = IndexingService(
            embedding_provider=mock_provider,
            vector_service=memory_vector_service,
        )
        with pytest.raises(KeyError, match="not found"):
            svc.index_document("non-existent-doc-id")

    def test_index_document_with_no_chunks_returns_empty_status(
        self, memory_vector_service: VectorService, mock_provider: MockEmbeddingProvider
    ) -> None:
        doc_id = "11111111-1111-4111-8111-111111111111"
        doc = DocumentDetail(
            document_id=doc_id,
            filename="blank.pdf",
            content_type="application/pdf",
            file_size_bytes=100,
            total_pages=1,
            total_characters=0,
            sha256_hash="hash123",
            ingestion_status="completed",
            uploaded_at="2026-10-29T00:00:00Z",
            pages=[],
        )
        storage_service.save_document(doc)
        storage_service.save_chunks(doc_id, [])

        svc = IndexingService(
            embedding_provider=mock_provider,
            vector_service=memory_vector_service,
        )
        resp = svc.index_document(doc_id)
        assert resp.indexed_chunks == 0
        assert resp.status == "empty"
        assert "no extractable chunks" in resp.message

    def test_index_document_success(
        self, memory_vector_service: VectorService, mock_provider: MockEmbeddingProvider
    ) -> None:
        doc_id = "22222222-2222-4222-8222-222222222222"
        doc = DocumentDetail(
            document_id=doc_id,
            filename="valid.pdf",
            content_type="application/pdf",
            file_size_bytes=1000,
            total_pages=1,
            total_characters=80,
            sha256_hash="hash456",
            ingestion_status="completed",
            uploaded_at="2026-10-29T00:00:00Z",
            pages=[
                PageExtraction(
                    page_number=1, text="Text content here", character_count=17
                )
            ],
        )
        chunks = [
            DocumentChunk(
                chunk_id="e1e1e1e1-e1e1-5e1e-8e1e-e1e1e1e1e1e1",
                document_id=doc_id,
                chunk_index=0,
                text="Text content here",
                section="Introduction",
                page_numbers=[1],
                character_count=17,
            )
        ]
        storage_service.save_document(doc)
        storage_service.save_chunks(doc_id, chunks)

        svc = IndexingService(
            embedding_provider=mock_provider,
            vector_service=memory_vector_service,
        )
        resp = svc.index_document(doc_id)
        assert resp.indexed_chunks == 1
        assert resp.status == "completed"
        assert memory_vector_service.count(document_id=doc_id) == 1


# ============================================================================
# API Integration Tests: Search & Index Endpoints
# ============================================================================


class TestSearchAndIndexAPI:
    """Integration tests for POST /api/documents/{id}/index and POST /api/search."""

    def test_search_validates_empty_query(self) -> None:
        resp = client.post("/api/search", json={"query": "   "})
        assert resp.status_code == 422

    def test_search_validates_invalid_limits(self) -> None:
        resp = client.post("/api/search", json={"query": "valid query", "limit": 0})
        assert resp.status_code == 422

        resp2 = client.post("/api/search", json={"query": "valid query", "limit": -5})
        assert resp2.status_code == 422

    def test_search_over_empty_collection(self) -> None:
        resp = client.post("/api/search", json={"query": "Annual leave policy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Annual leave policy"
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_end_to_end_upload_index_and_search_flow(self) -> None:
        """Full pipeline: upload real PDF -> extract -> chunk -> index -> search."""
        pdf_text_1 = (
            "1. Annual Leave Policy\n"
            "Full-time employees receive 20 days paid leave per calendar year."
        )
        pdf_text_2 = (
            "2. Travel Reimbursement\n"
            "Business travel expenses must be submitted within 30 days."
        )
        pdf_bytes = create_pdf_bytes([pdf_text_1, pdf_text_2])

        # 1. Upload PDF
        upload_resp = client.post(
            "/api/documents/upload",
            files={
                "file": ("company_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )
        assert upload_resp.status_code == 201
        doc_data = upload_resp.json()
        doc_id = doc_data["document_id"]

        # 2. Verify chunks were created
        chunks_resp = client.get(f"/api/documents/{doc_id}/chunks")
        assert chunks_resp.status_code == 200
        chunks_data = chunks_resp.json()
        assert chunks_data["total_chunks"] >= 1

        # 3. Index the document
        index_resp = client.post(f"/api/documents/{doc_id}/index")
        assert index_resp.status_code == 200
        index_data = index_resp.json()
        assert index_data["document_id"] == doc_id
        assert index_data["status"] == "completed"
        assert index_data["indexed_chunks"] == chunks_data["total_chunks"]

        # 4. Search for content from chunk 1
        search_resp = client.post(
            "/api/search",
            json={"query": "annual leave days", "limit": 5},
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["total_results"] >= 1

        top_hit = search_data["results"][0]
        assert top_hit["document_id"] == doc_id
        assert top_hit["filename"] == "company_policy.pdf"
        assert "chunk_id" in top_hit
        assert "chunk_index" in top_hit
        assert -1.0 <= top_hit["score"] <= 1.0

    def test_index_unknown_document_returns_404(self) -> None:
        unknown_id = "00000000-0000-0000-0000-000000000000"
        resp = client.post(f"/api/documents/{unknown_id}/index")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_reindexing_replaces_vectors_cleanly(self) -> None:
        """Reindexing the same document should not create duplicate vector points."""
        pdf_bytes = create_pdf_bytes(["Test policy for reindexing."])
        upload_resp = client.post(
            "/api/documents/upload",
            files={
                "file": ("reindex_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")
            },
        )
        doc_id = upload_resp.json()["document_id"]

        # Index first time
        resp1 = client.post(f"/api/documents/{doc_id}/index")
        assert resp1.status_code == 200
        chunks_count = resp1.json()["indexed_chunks"]

        # Index second time
        resp2 = client.post(f"/api/documents/{doc_id}/index")
        assert resp2.status_code == 200
        assert resp2.json()["indexed_chunks"] == chunks_count

        # Search should return only 1 result, not duplicates
        search_resp = client.post(
            "/api/search",
            json={"query": "Test policy for reindexing", "limit": 10},
        )
        assert search_resp.status_code == 200
        results = [
            r for r in search_resp.json()["results"] if r["document_id"] == doc_id
        ]
        assert len(results) == chunks_count
