"""Performance Benchmarking Script for PolicyPilot Phase 8.

Empirically measures real operational latencies across the pipeline:
1. PDF text extraction & validation (PyMuPDF)
2. Paragraph/heading chunking & metadata generation
3. Dense vector embedding generation (FastEmbed BGE-small-en-v1.5)
4. BM25 keyword index retrieval
5. Semantic vector search (in-memory Qdrant)
6. Hybrid Reciprocal Rank Fusion (RRF)
7. Cross-Encoder reranking (FlashRank ms-marco-TinyBERT-L-2-v2, 20 candidates)
8. End-to-end RAG question answering pipeline

Reports cold-start vs warm-start latencies with mean, min, and max over
repeated samples.
"""

import sys
import time
from pathlib import Path

# Ensure backend directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from qdrant_client import QdrantClient

from app.schemas.qa import AskRequest
from app.services.bm25_service import BM25Service
from app.services.chunking_service import chunk_document
from app.services.embedding_service import FastEmbedProvider, set_embedding_provider
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import MockLLMProvider
from app.services.pdf_service import extract_pdf_pages
from app.services.rag_service import RAGService
from app.services.reranker_service import FlashRankRerankerProvider, RerankerService
from app.services.vector_service import VectorService


def create_sample_pdf(pdf_path: Path) -> None:
    """Generate a realistic 5-page sample policy PDF for performance testing."""
    doc = pymupdf.open()
    for page_idx in range(5):
        page = doc.new_page()
        sec_num = page_idx + 1
        page_text = (
            f"POLICY SECTION {sec_num}.0: GENERAL STANDARD OPERATING PROCEDURES\n\n"
            f"Paragraph 1: All personnel must adhere to compliance standards set "
            f"forth in revision 2026. This section governs corporate travel, leave "
            f"allowances, and computational infrastructure across all regions.\n\n"
            f"Paragraph 2: Maximum lodging expense per night is $250.00 for tier-1. "
            f"Expense claims must be submitted within 30 days of trip conclusion.\n\n"
            f"Paragraph 3: Hardware multi-factor authentication is required on mobile "
            f"workstations. Password entropy must meet 16 alphanumeric characters."
        )
        page.insert_text((50, 72), page_text, fontsize=11)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()


def benchmark_pipeline(num_iterations: int = 10) -> dict[str, dict[str, float]]:
    """Run performance benchmarks and compute latency metrics."""
    results: dict[str, dict[str, float]] = {}
    temp_dir = Path("data/benchmark_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / "bench_policy.pdf"
    create_sample_pdf(pdf_path)

    # 1. PDF Extraction (PyMuPDF)
    cold_start = time.perf_counter()
    pages, total_pages, _, _ = extract_pdf_pages(pdf_path)
    extract_cold_ms = (time.perf_counter() - cold_start) * 1000.0

    extract_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        extract_pdf_pages(pdf_path)
        extract_times.append((time.perf_counter() - t0) * 1000.0)

    results["pdf_extraction_5_pages"] = {
        "cold_ms": round(extract_cold_ms, 2),
        "mean_ms": round(sum(extract_times) / len(extract_times), 2),
        "min_ms": round(min(extract_times), 2),
        "max_ms": round(max(extract_times), 2),
    }

    # 2. Document Chunking
    chunk_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        chunks = chunk_document(document_id="bench-doc-id", pages=pages)
        chunk_times.append((time.perf_counter() - t0) * 1000.0)

    results["chunking_and_metadata"] = {
        "cold_ms": round(chunk_times[0], 2),
        "mean_ms": round(sum(chunk_times) / len(chunk_times), 2),
        "min_ms": round(min(chunk_times), 2),
        "max_ms": round(max(chunk_times), 2),
    }

    # 3. Dense Vector Embedding (FastEmbed BGE-small-en-v1.5)
    embed_provider = FastEmbedProvider(
        model_name="BAAI/bge-small-en-v1.5", dimension=384
    )
    set_embedding_provider(embed_provider)

    sample_texts = [c.text for c in chunks]
    cold_start = time.perf_counter()
    embed_provider.embed_texts(sample_texts)
    embed_cold_ms = (time.perf_counter() - cold_start) * 1000.0

    embed_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        embed_provider.embed_texts(sample_texts)
        embed_times.append((time.perf_counter() - t0) * 1000.0)

    results["dense_embedding_batch"] = {
        "cold_ms": round(embed_cold_ms, 2),
        "mean_ms": round(sum(embed_times) / len(embed_times), 2),
        "min_ms": round(min(embed_times), 2),
        "max_ms": round(max(embed_times), 2),
    }

    # 4. In-Memory Qdrant Search
    test_qdrant = QdrantClient(location=":memory:")
    vec_svc = VectorService(
        client=test_qdrant,
        default_collection="bench_collection",
        default_dimension=384,
    )
    vec_svc.ensure_collection()
    embeddings = embed_provider.embed_texts([c.text for c in chunks])
    vec_svc.upsert_chunks(
        chunks=chunks, embeddings=embeddings, filename="bench_policy.pdf"
    )

    query_vec = embed_provider.embed_query("What is the hotel lodging reimbursement?")
    vec_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        vec_svc.search(query_vector=query_vec, limit=5)
        vec_times.append((time.perf_counter() - t0) * 1000.0)

    results["vector_search_qdrant"] = {
        "cold_ms": round(vec_times[0], 2),
        "mean_ms": round(sum(vec_times) / len(vec_times), 2),
        "min_ms": round(min(vec_times), 2),
        "max_ms": round(max(vec_times), 2),
    }

    # 5. BM25 Search
    bm25 = BM25Service()
    bm25.index_document_chunks(
        document_id="bench-doc-id", filename="bench_policy.pdf", chunks=chunks
    )
    bm25_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        bm25.search(query="lodging reimbursement limit hotel", limit=5)
        bm25_times.append((time.perf_counter() - t0) * 1000.0)

    results["bm25_search"] = {
        "cold_ms": round(bm25_times[0], 2),
        "mean_ms": round(sum(bm25_times) / len(bm25_times), 2),
        "min_ms": round(min(bm25_times), 2),
        "max_ms": round(max(bm25_times), 2),
    }

    # 6. Hybrid RRF Search
    hybrid = HybridSearchService(
        vector_service=vec_svc,
        embedding_provider=embed_provider,
        bm25_retriever=bm25,
    )
    hyb_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        hybrid.search(query="lodging reimbursement limit hotel", limit=5)
        hyb_times.append((time.perf_counter() - t0) * 1000.0)

    results["hybrid_rrf_search"] = {
        "cold_ms": round(hyb_times[0], 2),
        "mean_ms": round(sum(hyb_times) / len(hyb_times), 2),
        "min_ms": round(min(hyb_times), 2),
        "max_ms": round(max(hyb_times), 2),
    }

    # 7. FlashRank Cross-Encoder Reranker (20 candidates)
    reranker_provider = FlashRankRerankerProvider(model_name="ms-marco-TinyBERT-L-2-v2")
    reranker = RerankerService(
        reranker_provider=reranker_provider,
        hybrid_retriever=hybrid,
    )

    cold_start = time.perf_counter()
    reranker.search_and_rerank(
        query="lodging reimbursement limit hotel",
        limit=5,
        candidate_limit=20,
    )
    rerank_cold_ms = (time.perf_counter() - cold_start) * 1000.0

    rerank_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        reranker.search_and_rerank(
            query="lodging reimbursement limit hotel",
            limit=5,
            candidate_limit=20,
        )
        rerank_times.append((time.perf_counter() - t0) * 1000.0)

    results["cross_encoder_rerank_20_cands"] = {
        "cold_ms": round(rerank_cold_ms, 2),
        "mean_ms": round(sum(rerank_times) / len(rerank_times), 2),
        "min_ms": round(min(rerank_times), 2),
        "max_ms": round(max(rerank_times), 2),
    }

    # 8. End-to-End RAG Pipeline (using MockLLM for deterministic harness timing)
    mock_llm = MockLLMProvider()
    rag_svc = RAGService(rerank_service=reranker, llm_provider=mock_llm)

    cold_start = time.perf_counter()
    rag_svc.answer_question(
        AskRequest(question="What is the hotel lodging reimbursement in tier-1 areas?")
    )
    rag_cold_ms = (time.perf_counter() - cold_start) * 1000.0

    rag_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        rag_svc.answer_question(
            AskRequest(
                question="What is the hotel lodging reimbursement in tier-1 areas?"
            )
        )
        rag_times.append((time.perf_counter() - t0) * 1000.0)

    results["end_to_end_rag_orchestration"] = {
        "cold_ms": round(rag_cold_ms, 2),
        "mean_ms": round(sum(rag_times) / len(rag_times), 2),
        "min_ms": round(min(rag_times), 2),
        "max_ms": round(max(rag_times), 2),
    }

    # Clean up temp files
    try:
        pdf_path.unlink(missing_ok=True)
        temp_dir.rmdir()
    except Exception:
        pass

    return results


def print_benchmark_report(metrics: dict[str, dict[str, float]]) -> None:
    """Print formatted performance table."""
    print("=" * 76)
    print("PolicyPilot Phase 8 — Empirical Latency Benchmarks (CPU / Local Runtime)")
    print("=" * 76)
    print(
        f"{'Operation':<32} | {'Cold Start (ms)':<16} | "
        f"{'Warm Mean (ms)':<14} | {'Min (ms)':<8} | {'Max (ms)':<8}"
    )
    print("-" * 76)
    for op, data in metrics.items():
        row = (
            f"{op:<30} | {data['cold_ms']:<14.2f} | "
            f"{data['mean_ms']:<10.2f} | {data['min_ms']:<6.2f} | "
            f"{data['max_ms']:<6.2f}"
        )
        print(row)
    print("=" * 76)


if __name__ == "__main__":
    benchmarks = benchmark_pipeline(num_iterations=10)
    print_benchmark_report(benchmarks)
