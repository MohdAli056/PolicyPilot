"""Automated unit and integration tests for Cross-Encoder Reranking and Search."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.hybrid_search import HybridSearchResultItem
from app.services.bm25_service import BM25Service
from app.services.embedding_service import MockEmbeddingProvider, set_embedding_provider
from app.services.hybrid_search_service import (
    hybrid_search_service,
)
from app.services.indexing_service import indexing_service
from app.services.reranker_service import (
    FlashRankRerankerProvider,
    MockRerankerProvider,
    RerankerService,
    reranker_service,
    set_reranker_provider,
)
from app.services.search_service import search_service
from app.services.storage_service import storage_service
from app.services.vector_service import VectorService
from tests.evaluate_retrieval import run_retrieval_evaluation

client = TestClient(app)


def make_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    section: str = "Test Policy",
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
    """Isolate filesystem, in-memory Qdrant, mock embeddings, and mock reranker."""
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
        default_collection="test_rerank_chunks",
        default_dimension=384,
    )
    test_vector_service.ensure_collection()

    test_embedding_provider = MockEmbeddingProvider(dimension=384)
    set_embedding_provider(test_embedding_provider)

    test_bm25 = BM25Service()
    test_mock_reranker = MockRerankerProvider()
    set_reranker_provider(test_mock_reranker)

    # Monkeypatch services used in routers
    monkeypatch.setattr(indexing_service, "_custom_vector_service", test_vector_service)
    monkeypatch.setattr(
        indexing_service, "_custom_embedding_provider", test_embedding_provider
    )
    monkeypatch.setattr(indexing_service, "_custom_bm25_service", test_bm25)

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

    monkeypatch.setattr(
        reranker_service, "_custom_reranker_provider", test_mock_reranker
    )
    monkeypatch.setattr(
        reranker_service, "_custom_hybrid_service", hybrid_search_service
    )

    monkeypatch.setattr("app.routers.search.search_service", search_service)
    monkeypatch.setattr("app.routers.search.bm25_service", test_bm25)
    monkeypatch.setattr(
        "app.routers.search.hybrid_search_service", hybrid_search_service
    )
    monkeypatch.setattr("app.routers.search.reranker_service", reranker_service)


# ============================================================================
# Reranker Unit Tests
# ============================================================================


class TestRerankerUnit:
    """Unit tests for reranker providers and RerankerService logic."""

    def test_mock_provider_deterministic_scores(self):
        """Mock provider assigns deterministic scores based on query overlap."""
        mock = MockRerankerProvider()
        passages = [
            "Travel reimbursement policy for business flights.",
            "Annual vacation leave accrual rate.",
            "Travel meals and business per diem.",
        ]
        scores = mock.rerank("business travel", passages)
        assert len(scores) == 3
        # Passage 0 and 2 have 'business' and 'travel', passage 1 does not
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_mock_provider_score_overrides(self):
        """Explicit score overrides are respected."""
        mock = MockRerankerProvider()
        mock.set_score_override("Custom text", 0.999)
        scores = mock.rerank("query", ["Custom text", "Other text"])
        assert scores[0] == 0.999

    def test_mock_provider_simulates_honest_failure(self):
        """Mock provider raises RuntimeError when force_failure is enabled."""
        mock = MockRerankerProvider(force_failure=True)
        with pytest.raises(
            RuntimeError, match="Simulated cross-encoder inference failure"
        ):
            mock.rerank("query", ["Some text"])

    def test_mock_provider_empty_candidates(self):
        """Empty candidate lists produce empty score lists safely."""
        mock = MockRerankerProvider()
        assert mock.rerank("query", []) == []

    def test_flashrank_provider_1_to_1_mapping_preservation(self):
        """FlashRank provider maps scores back to the original candidate order."""
        provider = FlashRankRerankerProvider(model_name="test-model")
        mock_ranker = MagicMock()
        # Simulate FlashRank returning reordered passages
        mock_ranker.rerank.return_value = [
            {"id": 1, "text": "P1", "score": 0.85},
            {"id": 0, "text": "P0", "score": 0.25},
        ]
        provider._ranker = mock_ranker

        scores = provider.rerank("query", ["P0", "P1"])
        assert scores == [0.25, 0.85]

    def test_flashrank_provider_initialization_failure(self):
        """Model initialization failure raises clean logged RuntimeError."""
        provider = FlashRankRerankerProvider(model_name="nonexistent-model")
        with patch("flashrank.Ranker", side_effect=Exception("Disk full")):
            with pytest.raises(
                RuntimeError, match="FlashRank init failed for 'nonexistent-model'"
            ):
                provider._get_ranker()

    def test_descending_score_ordering(self):
        """RerankerService sorts candidates strictly by descending rerank_score."""
        mock_hybrid = MagicMock()
        candidates = [
            HybridSearchResultItem(
                document_id="doc1",
                filename="f1.pdf",
                chunk_id="c1",
                chunk_index=0,
                text="text 1",
                section=None,
                page_numbers=[1],
                character_count=6,
                vector_rank=1,
                keyword_rank=2,
                rrf_score=0.03,
                retrievers=["vector", "bm25"],
            ),
            HybridSearchResultItem(
                document_id="doc2",
                filename="f2.pdf",
                chunk_id="c2",
                chunk_index=0,
                text="text 2",
                section=None,
                page_numbers=[1],
                character_count=6,
                vector_rank=2,
                keyword_rank=1,
                rrf_score=0.02,
                retrievers=["vector", "bm25"],
            ),
        ]
        mock_hybrid.search.return_value = MagicMock(results=candidates)

        mock_reranker = MockRerankerProvider()
        mock_reranker.set_score_override("text 1", 0.10)
        mock_reranker.set_score_override("text 2", 0.95)

        service = RerankerService(
            reranker_provider=mock_reranker, hybrid_retriever=mock_hybrid
        )
        response = service.search_and_rerank(query="test", limit=5)

        assert len(response.results) == 2
        # text 2 had higher rerank score, so it must be ranked #1
        assert response.results[0].chunk_id == "c2"
        assert response.results[0].rerank_rank == 1
        assert response.results[0].rerank_score == 0.95
        assert response.results[1].chunk_id == "c1"
        assert response.results[1].rerank_rank == 2
        assert response.results[1].rerank_score == 0.10

    def test_deterministic_tie_breaking_by_chunk_id(self):
        """When rerank scores tie, ordering falls back deterministically to chunk_id."""
        mock_hybrid = MagicMock()
        candidates = [
            HybridSearchResultItem(
                document_id="doc1",
                filename="f1.pdf",
                chunk_id="chunk-z",
                chunk_index=0,
                text="same score text",
                section=None,
                page_numbers=[1],
                character_count=15,
                vector_rank=1,
                keyword_rank=1,
                rrf_score=0.03,
                retrievers=["vector"],
            ),
            HybridSearchResultItem(
                document_id="doc2",
                filename="f2.pdf",
                chunk_id="chunk-a",
                chunk_index=0,
                text="same score text",
                section=None,
                page_numbers=[1],
                character_count=15,
                vector_rank=2,
                keyword_rank=2,
                rrf_score=0.03,
                retrievers=["vector"],
            ),
        ]
        mock_hybrid.search.return_value = MagicMock(results=candidates)

        mock_reranker = MockRerankerProvider()
        mock_reranker.set_score_override("same score text", 0.50)

        service = RerankerService(
            reranker_provider=mock_reranker, hybrid_retriever=mock_hybrid
        )
        response = service.search_and_rerank(query="test", limit=5)

        # 'chunk-a' precedes 'chunk-z' alphabetically on equal scores
        assert response.results[0].chunk_id == "chunk-a"
        assert response.results[1].chunk_id == "chunk-z"

    def test_metadata_preservation_across_reranking(self):
        """All document and retrieval metadata are preserved through reranking."""
        mock_hybrid = MagicMock()
        candidates = [
            HybridSearchResultItem(
                document_id="doc-preserve-123",
                filename="special_policy.pdf",
                chunk_id="chunk-preserve-456",
                chunk_index=3,
                text="Preserved content",
                section="Section 4.1",
                page_numbers=[2, 3],
                character_count=17,
                vector_rank=1,
                keyword_rank=5,
                rrf_score=0.0315,
                retrievers=["vector", "bm25"],
            )
        ]
        mock_hybrid.search.return_value = MagicMock(results=candidates)
        mock_reranker = MockRerankerProvider()
        mock_reranker.set_score_override("Preserved content", 0.88)

        service = RerankerService(
            reranker_provider=mock_reranker, hybrid_retriever=mock_hybrid
        )
        res = service.search_and_rerank(query="test", limit=5)

        item = res.results[0]
        assert item.document_id == "doc-preserve-123"
        assert item.filename == "special_policy.pdf"
        assert item.chunk_id == "chunk-preserve-456"
        assert item.chunk_index == 3
        assert item.section == "Section 4.1"
        assert item.page_numbers == [2, 3]
        assert item.character_count == 17
        assert item.vector_rank == 1
        assert item.keyword_rank == 5
        assert item.rrf_score == 0.0315
        assert item.retrievers == ["vector", "bm25"]
        assert item.rerank_score == 0.88
        assert item.rerank_rank == 1

    def test_empty_candidate_pool_returns_empty_response(self):
        """Empty candidate pools return valid empty response with timing."""
        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = MagicMock(results=[])

        service = RerankerService(
            reranker_provider=MockRerankerProvider(), hybrid_retriever=mock_hybrid
        )
        response = service.search_and_rerank(query="test", limit=5)

        assert response.total_results == 0
        assert response.results == []
        assert response.timing_ms is not None
        assert response.timing_ms["total_ms"] >= 0.0

    def test_candidate_pool_size_and_final_result_slicing(self):
        """Requesting limit=2 from 5 candidates evaluates 5 but returns top 2."""
        mock_hybrid = MagicMock()
        candidates = [
            HybridSearchResultItem(
                document_id=f"doc{i}",
                filename=f"f{i}.pdf",
                chunk_id=f"c{i}",
                chunk_index=i,
                text=f"text {i}",
                section=None,
                page_numbers=[1],
                character_count=6,
                vector_rank=i + 1,
                keyword_rank=i + 1,
                rrf_score=0.01 * (5 - i),
                retrievers=["vector"],
            )
            for i in range(5)
        ]
        mock_hybrid.search.return_value = MagicMock(results=candidates)

        mock_reranker = MockRerankerProvider()
        # Invert scores so c4 is best
        for i in range(5):
            mock_reranker.set_score_override(f"text {i}", float(i * 0.2))

        service = RerankerService(
            reranker_provider=mock_reranker, hybrid_retriever=mock_hybrid
        )
        response = service.search_and_rerank(query="test", limit=2, candidate_limit=5)

        assert response.total_results == 2
        assert len(response.results) == 2
        assert response.results[0].chunk_id == "c4"
        assert response.results[1].chunk_id == "c3"


# ============================================================================
# Integration & API Tests
# ============================================================================


def index_test_chunks(document_id: str, filename: str, chunks: list[DocumentChunk]):
    """Index test chunks directly into the isolated vector and BM25 stores."""
    vec_svc = hybrid_search_service.vector_service
    bm25 = hybrid_search_service.bm25_retriever
    emb = hybrid_search_service.embedding_provider
    embeddings = emb.embed_texts([c.text for c in chunks])
    vec_svc.upsert_chunks(chunks=chunks, embeddings=embeddings, filename=filename)
    bm25.index_document_chunks(
        document_id=document_id, filename=filename, chunks=chunks
    )


class TestRerankSearchAPI:
    """Integration tests for POST /api/search/rerank and backward compatibility."""

    def test_rerank_api_successful_reordering(self):
        """POST /api/search/rerank retrieves and reranks candidates."""
        # Index two chunks
        c1 = make_chunk("doc-1", 0, "Annual vacation leave allowance")
        c2 = make_chunk("doc-1", 1, "Travel lodging expense limits")
        index_test_chunks("doc-1", "policy.pdf", [c1, c2])

        payload = {"query": "travel reimbursement", "limit": 5}
        resp = client.post("/api/search/rerank", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "travel reimbursement"
        assert data["total_results"] == 2
        assert "reranker_model" in data
        assert "timing_ms" in data
        assert len(data["results"]) == 2

        # Check fields of top result
        top = data["results"][0]
        assert "rerank_score" in top
        assert "rerank_rank" in top
        assert "rrf_score" in top
        assert "retrievers" in top
        assert top["rerank_rank"] == 1

    def test_rerank_api_empty_corpus(self):
        """POST /api/search/rerank on empty index returns 200 with empty list."""
        resp = client.post("/api/search/rerank", json={"query": "nonexistent policy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_rerank_api_validation_errors(self):
        """Empty queries and invalid limits produce 422 Unprocessable Entity."""
        # Empty query
        r1 = client.post("/api/search/rerank", json={"query": "   "})
        assert r1.status_code == 422

        # Negative limit
        r2 = client.post("/api/search/rerank", json={"query": "leave", "limit": -1})
        assert r2.status_code == 422

        # Negative candidate limit
        r3 = client.post(
            "/api/search/rerank", json={"query": "leave", "candidate_limit": 0}
        )
        assert r3.status_code == 422

    def test_rerank_api_surfaces_honest_503_error(self, monkeypatch):
        """Inference failure returns 503 without silently returning fake scores."""
        failing_reranker = MockRerankerProvider(force_failure=True)
        monkeypatch.setattr(
            reranker_service, "_custom_reranker_provider", failing_reranker
        )

        c = make_chunk("doc-err", 0, "Some policy text")
        index_test_chunks("doc-err", "p.pdf", [c])

        resp = client.post("/api/search/rerank", json={"query": "test query"})
        assert resp.status_code == 503
        assert "Reranking search service error" in resp.json()["detail"]

    def test_existing_endpoints_backward_compatibility(self):
        """Phases 4 & 5 endpoints (/api/search, /api/search/keyword,
        /api/search/hybrid) remain fully functional.
        """
        c = make_chunk("doc-compat", 0, "Health insurance coverage details")
        index_test_chunks("doc-compat", "compat.pdf", [c])

        # Phase 4 semantic endpoint
        r_sem = client.post("/api/search", json={"query": "health insurance"})
        assert r_sem.status_code == 200
        assert "score" in r_sem.json()["results"][0]

        # Phase 5 keyword endpoint
        r_kw = client.post("/api/search/keyword", json={"query": "health insurance"})
        assert r_kw.status_code == 200
        assert r_kw.json()["results"][0]["chunk_id"] == c.chunk_id

        # Phase 5 hybrid endpoint
        r_hyb = client.post("/api/search/hybrid", json={"query": "health insurance"})
        assert r_hyb.status_code == 200
        assert "rrf_score" in r_hyb.json()["results"][0]
        # Confirm hybrid response does not contain rerank fields
        assert "rerank_score" not in r_hyb.json()["results"][0]


# ============================================================================
# Retrieval Quality Benchmark Test
# ============================================================================


class TestRetrievalBenchmark:
    """Validates that retrieval evaluation fixture runs and produces valid metrics."""

    def test_evaluation_fixture_deterministic(self):
        """Evaluation fixture executes across all pipelines and returns
        bounded metrics.
        """
        emb = MockEmbeddingProvider(dimension=384)
        rerank = MockRerankerProvider()

        report = run_retrieval_evaluation(emb, rerank)
        assert "summary" in report
        assert "details" in report
        assert len(report["details"]) == 5

        for pipe in ("vector_only", "bm25_only", "hybrid_rrf", "reranked"):
            metrics = report["summary"][pipe]
            assert 0.0 <= metrics["mean_mrr"] <= 1.0
            assert 0.0 <= metrics["mean_recall_1"] <= 1.0
            assert 0.0 <= metrics["mean_recall_3"] <= 1.0
            assert 0.0 <= metrics["mean_ndcg_3"] <= 1.0
