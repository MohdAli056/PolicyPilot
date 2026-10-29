"""Automated tests for BM25 keyword search and Hybrid Retrieval with RRF."""

import io
import uuid
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.schemas.chunk import DocumentChunk
from app.services.bm25_service import BM25Service, tokenize
from app.services.embedding_service import MockEmbeddingProvider, set_embedding_provider
from app.services.hybrid_search_service import (
    HybridSearchService,
    compute_rrf_score,
    hybrid_search_service,
)
from app.services.indexing_service import indexing_service
from app.services.search_service import search_service
from app.services.storage_service import storage_service
from app.services.vector_service import VectorService

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


def make_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    section: str = "Test Section",
    page_numbers: list[int] = None,
) -> DocumentChunk:
    """Helper to construct a valid DocumentChunk."""
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}"))
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        section=section,
        page_numbers=page_numbers or [1],
        character_count=len(text),
    )


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate filesystem storage, in-memory Qdrant, mock embeddings, and BM25 state."""
    temp_docs = tmp_path / "documents"
    temp_meta = tmp_path / "metadata"
    temp_chunks = tmp_path / "chunks"
    temp_docs.mkdir(parents=True, exist_ok=True)
    temp_meta.mkdir(parents=True, exist_ok=True)
    temp_chunks.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_service, "documents_dir", temp_docs)
    monkeypatch.setattr(storage_service, "metadata_dir", temp_meta)
    monkeypatch.setattr(storage_service, "chunks_dir", temp_chunks)

    # In-memory Qdrant
    test_qdrant_client = QdrantClient(location=":memory:")
    test_vector_service = VectorService(
        client=test_qdrant_client,
        default_collection="test_chunks",
        default_dimension=384,
    )
    test_vector_service.ensure_collection()

    test_embedding_provider = MockEmbeddingProvider(dimension=384)
    set_embedding_provider(test_embedding_provider)

    # Clear and isolate BM25 global service
    test_bm25 = BM25Service()

    # Monkeypatch services used in routers
    monkeypatch.setattr(indexing_service, "_custom_vector_service", test_vector_service)
    monkeypatch.setattr(
        indexing_service, "_custom_embedding_provider", test_embedding_provider
    )
    monkeypatch.setattr(search_service, "_custom_vector_service", test_vector_service)
    monkeypatch.setattr(
        search_service, "_custom_embedding_provider", test_embedding_provider
    )
    monkeypatch.setattr(
        hybrid_search_service, "_custom_vector_service", test_vector_service
    )
    monkeypatch.setattr(
        hybrid_search_service, "_custom_embedding_provider", test_embedding_provider
    )
    monkeypatch.setattr(hybrid_search_service, "_custom_bm25_service", test_bm25)
    monkeypatch.setattr("app.routers.search.bm25_service", test_bm25)
    monkeypatch.setattr(
        "app.routers.search.hybrid_search_service", hybrid_search_service
    )
    monkeypatch.setattr(indexing_service, "_custom_bm25_service", test_bm25)
    monkeypatch.setattr("app.routers.documents.bm25_service", test_bm25)


# ============================================================================
# BM25 Unit Tests
# ============================================================================


class TestBM25Unit:
    """Unit tests for BM25 tokenization, retrieval, and lifecycle."""

    def test_tokenization_case_punctuation_and_numbers(self):
        """Tokenization handles case-folding, policy identifiers, and numbers."""
        text = "Policy HR-2024 (Section 1.1): Reimbursement of $500 for travel!"
        tokens = tokenize(text)

        # Compound identifiers preserved and constituent parts emitted
        assert "hr-2024" in tokens
        assert "2024" in tokens
        assert "hr" in tokens
        assert "1.1" in tokens
        assert "500" in tokens
        assert "travel" in tokens
        assert "reimbursement" in tokens
        # Punctuation stripped
        assert "$" not in tokens
        assert "!" not in tokens
        assert "(" not in tokens

    def test_exact_keyword_retrieval(self):
        """Exact query retrieves the targeted chunk with preserved metadata."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(doc_id, 0, "Employees receive 25 days annual leave per year."),
            make_chunk(
                doc_id, 1, "Travel reimbursement requests must include receipts."
            ),
        ]
        bm25.index_document_chunks(doc_id, "policy.pdf", chunks)

        results = bm25.search("annual leave")
        assert len(results) == 1
        assert results[0].chunk_index == 0
        assert "annual leave" in results[0].text
        assert results[0].document_id == doc_id
        assert results[0].filename == "policy.pdf"
        assert results[0].keyword_rank == 1
        assert results[0].vector_rank is None
        assert results[0].retrievers == ["bm25"]

    def test_case_insensitive_matching(self):
        """Upper, lower, or mixed case queries produce identical keyword matches."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(
                doc_id, 0, "Health Insurance coverage applies to full-time staff."
            )
        ]
        bm25.index_document_chunks(doc_id, "benefits.pdf", chunks)

        lower_res = bm25.search("health insurance")
        upper_res = bm25.search("HEALTH INSURANCE")
        mixed_res = bm25.search("hEaLtH InSuRaNcE")

        assert len(lower_res) == 1
        assert len(upper_res) == 1
        assert len(mixed_res) == 1
        assert lower_res[0].chunk_id == upper_res[0].chunk_id == mixed_res[0].chunk_id

    def test_numeric_and_code_identifiers(self):
        """Exact policy codes like 'SEC-101' and amounts like '500' can be retrieved."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(
                doc_id, 0, "Under rule SEC-101, employees may claim up to 500 dollars."
            ),
            make_chunk(
                doc_id,
                1,
                "Standard working hours are 9am to 5pm Monday through Friday.",
            ),
        ]
        bm25.index_document_chunks(doc_id, "handbook.pdf", chunks)

        res_code = bm25.search("SEC-101")
        assert len(res_code) == 1
        assert res_code[0].chunk_index == 0

        res_amount = bm25.search("500")
        assert len(res_amount) == 1
        assert res_amount[0].chunk_index == 0

    def test_empty_corpus_returns_empty_list(self):
        """Searching an empty corpus never crashes and returns empty list."""
        bm25 = BM25Service()
        assert bm25.total_chunks == 0
        results = bm25.search("vacation")
        assert results == []

    def test_no_match_returns_no_fabricated_results(self):
        """Queries that match no chunk return empty list, never fabricated results."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(
                doc_id, 0, "Workplace safety guidelines and fire exit locations."
            )
        ]
        bm25.index_document_chunks(doc_id, "safety.pdf", chunks)

        results = bm25.search("cryptocurrency investment")
        assert results == []

    def test_empty_query_returns_empty_list(self):
        """Whitespace or empty query returns empty list without error."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [make_chunk(doc_id, 0, "Parental leave policy details.")]
        bm25.index_document_chunks(doc_id, "leave.pdf", chunks)

        assert bm25.search("") == []
        assert bm25.search("   ") == []
        assert bm25.search("!@#$%^&*") == []

    def test_result_limits_respected(self):
        """Limit parameter constrains the number of returned chunks."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(doc_id, i, f"Policy statement {i} regarding company travel.")
            for i in range(10)
        ]
        bm25.index_document_chunks(doc_id, "travel.pdf", chunks)

        res_3 = bm25.search("travel", limit=3)
        assert len(res_3) == 3

        res_5 = bm25.search("travel", limit=5)
        assert len(res_5) == 5

    def test_duplicate_chunks_not_returned(self):
        """A chunk is returned at most once per query."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(doc_id, 0, "Remote work policy remote remote remote work."),
            make_chunk(doc_id, 1, "Office attendance expectations."),
        ]
        bm25.index_document_chunks(doc_id, "remote.pdf", chunks)

        results = bm25.search("remote work")
        assert len(results) == 1
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_reindexing_replaces_chunks_cleanly(self):
        """Reindexing replaces chunks for a document without leaving stale orphans."""
        bm25 = BM25Service()
        doc_id = str(uuid.uuid4())

        # Initial indexing with 2 chunks
        initial_chunks = [
            make_chunk(doc_id, 0, "Old version mentions parking permits in lot A."),
            make_chunk(doc_id, 1, "Old version mentions cafeteria hours."),
        ]
        bm25.index_document_chunks(doc_id, "amenities.pdf", initial_chunks)
        assert len(bm25.search("parking permits")) == 1

        # Reindex with replaced chunks (parking permits removed)
        updated_chunks = [
            make_chunk(doc_id, 0, "New version: free transit passes provided."),
        ]
        bm25.index_document_chunks(doc_id, "amenities.pdf", updated_chunks)

        # Old term is no longer searchable
        assert len(bm25.search("parking permits")) == 0
        # New term is searchable
        assert len(bm25.search("transit passes")) == 1
        assert bm25.total_chunks == 1


# ============================================================================
# Hybrid Search and RRF Unit Tests
# ============================================================================


class TestHybridRRFUnit:
    """Unit tests for Reciprocal Rank Fusion formula and hybrid fusion logic."""

    def test_rrf_formula_hand_calculated_verification(self):
        """RRF score matches exact hand-calculated formula: sum(1 / (k + rank))."""
        # Let k = 60
        k = 60
        # Chunk rank 1 in vector, rank 2 in BM25:
        # 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252247
        score = compute_rrf_score([1, 2], k=k)
        expected = (1.0 / 61.0) + (1.0 / 62.0)
        assert pytest.approx(score, rel=1e-5) == expected
        assert pytest.approx(round(score, 6), abs=1e-6) == 0.032522

    def test_single_retriever_chunk_retained_with_null_other_rank(self):
        """A chunk returned by only one retriever is retained with correct null rank."""
        doc_id = str(uuid.uuid4())
        chunk = make_chunk(doc_id, 0, "Bereavement leave entitlement.")

        # Vector service returns chunk at rank 1
        mock_vec = [
            {
                "id": chunk.chunk_id,
                "score": 0.85,
                "payload": {
                    "document_id": doc_id,
                    "filename": "policy.pdf",
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": 0,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_numbers": chunk.page_numbers,
                    "character_count": chunk.character_count,
                },
            }
        ]

        class FakeVectorService:
            def search(self, query_vector, limit):
                return mock_vec

        class FakeBM25Service:
            def search(self, query, limit):
                return []  # BM25 returns nothing

        hybrid = HybridSearchService(
            vector_service=FakeVectorService(),
            bm25_retriever=FakeBM25Service(),
            embedding_provider=MockEmbeddingProvider(384),
        )

        resp = hybrid.search(query="bereavement leave", limit=5)
        assert resp.total_results == 1
        item = resp.results[0]
        assert item.chunk_id == chunk.chunk_id
        assert item.vector_rank == 1
        assert item.keyword_rank is None  # Must be None, not 0
        assert item.retrievers == ["vector"]
        assert pytest.approx(item.rrf_score, abs=1e-5) == 1.0 / 61.0

    def test_both_retrievers_contribution_and_deduplication(self):
        """Chunk returned by both retrievers is deduplicated and accumulates score."""
        doc_id = str(uuid.uuid4())
        chunk_both = make_chunk(doc_id, 0, "Both retrievers found this passage.")
        chunk_vec_only = make_chunk(doc_id, 1, "Only vector found this.")

        mock_vec = [
            {
                "id": chunk_both.chunk_id,
                "score": 0.90,
                "payload": {
                    "document_id": doc_id,
                    "filename": "policy.pdf",
                    "chunk_id": chunk_both.chunk_id,
                    "chunk_index": 0,
                    "text": chunk_both.text,
                    "section": chunk_both.section,
                    "page_numbers": chunk_both.page_numbers,
                    "character_count": chunk_both.character_count,
                },
            },
            {
                "id": chunk_vec_only.chunk_id,
                "score": 0.80,
                "payload": {
                    "document_id": doc_id,
                    "filename": "policy.pdf",
                    "chunk_id": chunk_vec_only.chunk_id,
                    "chunk_index": 1,
                    "text": chunk_vec_only.text,
                    "section": chunk_vec_only.section,
                    "page_numbers": chunk_vec_only.page_numbers,
                    "character_count": chunk_vec_only.character_count,
                },
            },
        ]

        # BM25 returns chunk_both at rank 1
        bm25_unit = BM25Service()
        bm25_unit.index_document_chunks(doc_id, "policy.pdf", [chunk_both])

        class FakeVectorService:
            def search(self, query_vector, limit):
                return mock_vec

        hybrid = HybridSearchService(
            vector_service=FakeVectorService(),
            bm25_retriever=bm25_unit,
            embedding_provider=MockEmbeddingProvider(384),
        )

        resp = hybrid.search(query="passage", limit=5)
        # Deduplicated: total 2 unique chunks
        assert resp.total_results == 2

        # chunk_both received ranks from both: vector #1, bm25 #1 -> higher score
        top = resp.results[0]
        assert top.chunk_id == chunk_both.chunk_id
        assert top.vector_rank == 1
        assert top.keyword_rank == 1
        assert "vector" in top.retrievers and "bm25" in top.retrievers
        assert pytest.approx(top.rrf_score, abs=1e-5) == (1.0 / 61.0) + (1.0 / 61.0)

        # chunk_vec_only is second
        second = resp.results[1]
        assert second.chunk_id == chunk_vec_only.chunk_id
        assert second.vector_rank == 2
        assert second.keyword_rank is None
        assert second.retrievers == ["vector"]
        assert pytest.approx(second.rrf_score, abs=1e-5) == 1.0 / 62.0

    def test_deterministic_tie_breaking(self):
        """Tied RRF scores are sorted deterministically by chunk_id ascending."""
        doc_id = str(uuid.uuid4())
        # Two chunks with identical vector score and rank (or single retriever rank)
        chunk_b = make_chunk(doc_id, 0, "Passage B")
        chunk_a = make_chunk(doc_id, 1, "Passage A")
        # Ensure distinct IDs
        chunk_a.chunk_id = "chunk-1111"
        chunk_b.chunk_id = "chunk-2222"

        # Both have rank 1 from different retrievers (same single score 1/61)
        # Let fake vector return chunk_b at rank 1, and BM25 return chunk_a at rank 1
        class FakeVectorService:
            def search(self, query_vector, limit):
                return [
                    {
                        "id": chunk_b.chunk_id,
                        "score": 0.8,
                        "payload": {
                            "document_id": doc_id,
                            "filename": "policy.pdf",
                            "chunk_id": chunk_b.chunk_id,
                            "chunk_index": 0,
                            "text": chunk_b.text,
                            "section": chunk_b.section,
                            "page_numbers": chunk_b.page_numbers,
                            "character_count": chunk_b.character_count,
                        },
                    }
                ]

        class FakeBM25Service:
            def search(self, query, limit):
                from app.schemas.hybrid_search import HybridSearchResultItem

                return [
                    HybridSearchResultItem(
                        document_id=doc_id,
                        filename="policy.pdf",
                        chunk_id=chunk_a.chunk_id,
                        chunk_index=1,
                        text=chunk_a.text,
                        section=chunk_a.section,
                        page_numbers=chunk_a.page_numbers,
                        character_count=chunk_a.character_count,
                        vector_rank=None,
                        keyword_rank=1,
                        rrf_score=1.0,
                        retrievers=["bm25"],
                    )
                ]

        hybrid = HybridSearchService(
            vector_service=FakeVectorService(),
            bm25_retriever=FakeBM25Service(),
            embedding_provider=MockEmbeddingProvider(384),
        )

        resp = hybrid.search("test", limit=5)
        assert resp.total_results == 2
        # Both have RRF score = 1/61 = 0.016393
        assert resp.results[0].rrf_score == resp.results[1].rrf_score
        # Tied tie-breaking must put chunk-1111 before chunk-2222
        assert resp.results[0].chunk_id == "chunk-1111"
        assert resp.results[1].chunk_id == "chunk-2222"

    def test_invalid_rrf_constant_rejected(self):
        """RRF constant k <= 0 raises ValueError to prevent division by zero."""
        with pytest.raises(
            ValueError, match="RRF constant k must be strictly positive"
        ):
            compute_rrf_score([1], k=0)

        with pytest.raises(
            ValueError, match="RRF constant k must be strictly positive"
        ):
            compute_rrf_score([1], k=-10)


# ============================================================================
# API and Integration Tests
# ============================================================================


class TestSearchAPIIntegration:
    """Integration tests for search, keyword, and hybrid endpoints."""

    def test_existing_semantic_endpoint_backward_compatible(self):
        """POST /api/search retains pure semantic results without hybrid fields."""
        pdf_bytes = create_pdf_bytes(
            ["Expense claim procedure for meals and incidentals."]
        )

        # Upload document
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("meals.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload_resp.status_code == 201

        # Index document into Qdrant
        index_resp = client.post(
            f"/api/documents/{upload_resp.json()['document_id']}/index"
        )
        assert index_resp.status_code == 200

        # Query semantic search endpoint
        search_resp = client.post(
            "/api/search",
            json={"query": "meal expenses", "limit": 5},
        )
        assert search_resp.status_code == 200
        data = search_resp.json()
        assert "results" in data
        assert data["total_results"] >= 1
        item = data["results"][0]
        # Semantic search contract has 'score', but NOT 'rrf_score' or 'vector_rank'
        assert "score" in item
        assert "rrf_score" not in item
        assert "vector_rank" not in item

    def test_bm25_keyword_endpoint(self):
        """POST /api/search/keyword returns exact keyword matching chunks."""
        pdf_bytes = create_pdf_bytes(
            ["Policy REF-7788: Company vehicle maintenance and insurance requirements."]
        )
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("fleet.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload_resp.status_code == 201

        # Search for unique identifier
        kw_resp = client.post(
            "/api/search/keyword",
            json={"query": "REF-7788", "limit": 5},
        )
        assert kw_resp.status_code == 200
        data = kw_resp.json()
        assert data["total_results"] == 1
        assert "REF-7788" in data["results"][0]["text"]
        assert data["results"][0]["keyword_rank"] == 1
        assert data["results"][0]["vector_rank"] is None

    def test_hybrid_search_endpoint_with_custom_k_and_limit(self):
        """POST /api/search/hybrid combines signals and respects limits and rrf_k."""
        pdf_bytes = create_pdf_bytes(
            [
                "Section 4.1: Maternity Leave provisions for eligible staff.",
                "Section 4.2: Health and Dental benefits coverage during leave.",
            ]
        )
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("leave.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_resp.json()["document_id"]
        client.post(f"/api/documents/{doc_id}/index")

        # Hybrid search with custom rrf_k=50 and limit=1
        hybrid_resp = client.post(
            "/api/search/hybrid",
            json={"query": "Section 4.1 maternity leave", "limit": 1, "rrf_k": 50},
        )
        assert hybrid_resp.status_code == 200
        data = hybrid_resp.json()
        assert data["rrf_k"] == 50
        assert data["total_results"] == 1
        assert len(data["results"]) == 1
        item = data["results"][0]
        assert item["rrf_score"] > 0
        assert "retrievers" in item
        assert "chunk_id" in item

    def test_hybrid_search_validates_inputs(self):
        """Invalid query, limit, or rrf_k return HTTP 422 validation errors."""
        # Empty query
        res1 = client.post("/api/search/hybrid", json={"query": "   "})
        assert res1.status_code == 422

        # Invalid limit
        res2 = client.post("/api/search/hybrid", json={"query": "leave", "limit": 0})
        assert res2.status_code == 422

        # Invalid rrf_k
        res3 = client.post("/api/search/hybrid", json={"query": "leave", "rrf_k": -5})
        assert res3.status_code == 422

    def test_sync_from_storage_restores_index_across_restarts(self, tmp_path):
        """sync_from_storage restores BM25 index from disk without loss."""
        # Create a document and persist chunks to storage_service
        doc_id = str(uuid.uuid4())
        chunks = [
            make_chunk(doc_id, 0, "Persistent policy chunk surviving restart."),
        ]
        storage_service.save_chunks(doc_id, chunks)

        # Fresh BM25Service instance simulating backend restart
        restarted_bm25 = BM25Service()
        assert restarted_bm25.total_chunks == 0

        # Run startup sync
        loaded_count = restarted_bm25.sync_from_storage(storage_service)
        assert loaded_count == 1
        assert restarted_bm25.total_chunks == 1

        results = restarted_bm25.search("surviving restart")
        assert len(results) == 1
        assert results[0].document_id == doc_id
