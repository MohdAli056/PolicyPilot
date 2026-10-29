"""Retrieval Quality Evaluation Fixture for PolicyPilot Phase 6.

Evaluates retrieval quality across candidate ranking stages:
  1. Semantic Vector Retrieval (dense cosine similarity)
  2. Okapi BM25 Retrieval (sparse keyword matching)
  3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
  4. Cross-Encoder Reranked Hybrid Retrieval (FlashRank ms-marco-TinyBERT-L-2-v2)

Metrics measured:
  - Mean Reciprocal Rank (MRR)
  - Recall@1, Recall@3, Recall@5
  - Normalized Discounted Cumulative Gain (nDCG@3, nDCG@5)
"""

import math
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure backend root is in sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from app.schemas.chunk import DocumentChunk
from app.services.bm25_service import BM25Service
from app.services.embedding_service import BaseEmbeddingProvider, FastEmbedProvider
from app.services.hybrid_search_service import HybridSearchService
from app.services.reranker_service import (
    BaseRerankerProvider,
    FlashRankRerankerProvider,
    RerankerService,
)
from app.services.vector_service import VectorService


# Deterministic UUID mapping for evaluation corpus
def make_eval_uuid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"eval:{name}"))


CID_TRV_01 = make_eval_uuid("chunk-pol-trv-01")
CID_TRV_02 = make_eval_uuid("chunk-pol-trv-02")
CID_LV_01 = make_eval_uuid("chunk-pol-lv-01")
CID_PAR_01 = make_eval_uuid("chunk-pol-par-01")
CID_RMT_01 = make_eval_uuid("chunk-pol-rmt-01")
CID_INS_01 = make_eval_uuid("chunk-pol-ins-01")
CID_SEC_01 = make_eval_uuid("chunk-pol-sec-01")
CID_DIS_01 = make_eval_uuid("chunk-pol-dis-01")

EVAL_DOCUMENTS = [
    {
        "chunk_id": CID_TRV_01,
        "document_id": "doc-travel-policy",
        "filename": "travel_and_expense_policy_2024.pdf",
        "chunk_index": 0,
        "section": "Section 3.1: Lodging and Hotel Limits",
        "page_numbers": [3],
        "text": (
            "Under Policy POL-TRV-2024, hotel lodging for domestic business travel "
            "is reimbursed up to a maximum rate of $250 per night excluding taxes. "
            "All expense reports must be submitted within 30 days of trip completion."
        ),
    },
    {
        "chunk_id": CID_TRV_02,
        "document_id": "doc-travel-policy",
        "filename": "travel_and_expense_policy_2024.pdf",
        "chunk_index": 1,
        "section": "Section 4.2: Daily Meals and Incidental Expenses",
        "page_numbers": [4],
        "text": (
            "The company provides a per diem meal allowance capped at $75 per day "
            "for overnight travel. Alcoholic beverages and entertainment expenses "
            "are strictly non-reimbursable under standard business expense rules."
        ),
    },
    {
        "chunk_id": CID_LV_01,
        "document_id": "doc-leave-policy",
        "filename": "employee_leave_handbook_2024.pdf",
        "chunk_index": 0,
        "section": "Section 2.0: Annual Vacation Leave",
        "page_numbers": [1],
        "text": (
            "Full-time regular employees accrue 20 days of paid annual vacation leave "
            "and 10 days of paid sick leave per calendar year under POL-LV-2024. "
            "A maximum of 5 unused vacation days may carry over to the subsequent year."
        ),
    },
    {
        "chunk_id": CID_PAR_01,
        "document_id": "doc-leave-policy",
        "filename": "employee_leave_handbook_2024.pdf",
        "chunk_index": 1,
        "section": "Section 5.0: Parental and Caregiver Leave",
        "page_numbers": [3],
        "text": (
            "Under Policy POL-PAR-2024, eligible primary caregivers receive 16 "
            "consecutive weeks of 100% fully paid parental leave following birth. "
            "Secondary caregivers are entitled to 4 weeks of fully paid leave."
        ),
    },
    {
        "chunk_id": CID_RMT_01,
        "document_id": "doc-remote-policy",
        "filename": "remote_work_guidelines_2024.pdf",
        "chunk_index": 0,
        "section": "Section 2.4: Home Office Equipment Stipend",
        "page_numbers": [2],
        "text": (
            "Employees approved for permanent remote work are eligible for an "
            "ergonomic workstation subsidy of up to $500 once every two years. "
            "Reimbursable items include ergonomic chairs and monitor converters."
        ),
    },
    {
        "chunk_id": CID_INS_01,
        "document_id": "doc-benefits-policy",
        "filename": "health_and_insurance_benefits_2024.pdf",
        "chunk_index": 0,
        "section": "Section 1.2: Health and Dental Enrollment",
        "page_numbers": [1],
        "text": (
            "Comprehensive health, vision, and dental insurance benefits under Policy "
            "POL-INS-2024 take effect on the first day of the calendar month following "
            "30 consecutive days of full-time active employment."
        ),
    },
    {
        "chunk_id": CID_SEC_01,
        "document_id": "doc-security-policy",
        "filename": "information_security_policy_2024.pdf",
        "chunk_index": 0,
        "section": "Section 7.1: Password Complexity and Multi-Factor Auth",
        "page_numbers": [5],
        "text": (
            "Corporate passwords must contain a minimum of 14 characters including "
            "uppercase, lowercase, numbers, and symbols. Multi-factor auth (MFA) "
            "is mandatory for all remote and VPN access to internal systems."
        ),
    },
    {
        "chunk_id": CID_DIS_01,
        "document_id": "doc-travel-policy",
        "filename": "travel_and_expense_policy_2024.pdf",
        "chunk_index": 2,
        "section": "Section 1.1: General Travel Guidelines",
        "page_numbers": [1],
        "text": (
            "All employee business travel proposals that fail to satisfy annual leave "
            "scheduling deadlines or exceed expenditure ceilings will be rejected."
        ),
    },
]

# ============================================================================
# Representative Evaluation Queries
# ============================================================================

EVAL_QUERIES = [
    {
        "query_id": "Q1",
        "type": "Exact policy identifier and terminology",
        "query": "POL-TRV-2024 hotel reimbursement limit",
        "relevant_chunks": {CID_TRV_01: 3, CID_DIS_01: 1},
    },
    {
        "query_id": "Q2",
        "type": "Natural-language question",
        "query": "How many weeks of fully paid leave can a new mother take off?",
        "relevant_chunks": {CID_PAR_01: 3},
    },
    {
        "query_id": "Q3",
        "type": "Numeric limits and monetary amounts",
        "query": "What is the maximum daily per diem food allowance on trips?",
        "relevant_chunks": {CID_TRV_02: 3, CID_DIS_01: 1},
    },
    {
        "query_id": "Q4",
        "type": "Misleading lexical overlap",
        "query": "Ergonomic chair and home desk setup subsidy",
        "relevant_chunks": {CID_RMT_01: 3},
    },
    {
        "query_id": "Q5",
        "type": "Semantic intent differs from policy verbatim phrasing",
        "query": "When does medical insurance coverage start after joining?",
        "relevant_chunks": {CID_INS_01: 3},
    },
]

# ============================================================================
# Metric Calculators
# ============================================================================


def reciprocal_rank(
    ranked_chunk_ids: list[str], relevant_chunks: dict[str, int]
) -> float:
    """Calculate reciprocal rank (1 / rank) of the first relevant document."""
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant_chunks and relevant_chunks[chunk_id] > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked_chunk_ids: list[str], relevant_chunks: dict[str, int], k: int
) -> float:
    """Calculate Recall@k: 1.0 if any primary relevant chunk is retrieved, else 0.0."""
    primary_relevant = {cid for cid, rel in relevant_chunks.items() if rel >= 2}
    if not primary_relevant:
        primary_relevant = set(relevant_chunks.keys())
    top_k_chunks = set(ranked_chunk_ids[:k])
    return 1.0 if len(primary_relevant.intersection(top_k_chunks)) > 0 else 0.0


def dcg_at_k(
    ranked_chunk_ids: list[str], relevant_chunks: dict[str, int], k: int
) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        rel = relevant_chunks.get(chunk_id, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(
    ranked_chunk_ids: list[str], relevant_chunks: dict[str, int], k: int
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    actual_dcg = dcg_at_k(ranked_chunk_ids, relevant_chunks, k)

    # Ideal DCG: sorted by graded relevance descending
    sorted_rels = sorted(relevant_chunks.values(), reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(sorted_rels[:k], start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    if idcg <= 0.0:
        return 0.0
    return actual_dcg / idcg


# ============================================================================
# Evaluation Runner
# ============================================================================


def build_evaluation_index(
    embedding_provider: BaseEmbeddingProvider,
) -> tuple[VectorService, BM25Service]:
    """Index evaluation corpus into in-memory Qdrant and BM25 index."""
    qdrant = QdrantClient(location=":memory:")
    dim = embedding_provider.dimension
    vector_svc = VectorService(
        client=qdrant, default_collection="eval_chunks", default_dimension=dim
    )
    vector_svc.ensure_collection()

    bm25_svc = BM25Service()

    doc_chunks = [
        DocumentChunk(
            chunk_id=doc["chunk_id"],
            document_id=doc["document_id"],
            chunk_index=doc["chunk_index"],
            text=doc["text"],
            section=doc["section"],
            page_numbers=doc["page_numbers"],
            character_count=len(doc["text"]),
        )
        for doc in EVAL_DOCUMENTS
    ]

    texts = [c.text for c in doc_chunks]
    embeddings = embedding_provider.embed_texts(texts)

    vector_svc.upsert_chunks(
        chunks=doc_chunks, embeddings=embeddings, filename="eval_policy_corpus.pdf"
    )

    bm25_svc.index_document_chunks(
        document_id="eval-doc",
        filename="eval_policy_corpus.pdf",
        chunks=doc_chunks,
    )

    return vector_svc, bm25_svc


def run_retrieval_evaluation(
    embedding_provider: BaseEmbeddingProvider,
    reranker_provider: BaseRerankerProvider,
) -> dict[str, Any]:
    """Execute evaluation queries across all retrieval pipelines and compute metrics."""
    vector_svc, bm25_svc = build_evaluation_index(embedding_provider)
    hybrid_svc = HybridSearchService(
        embedding_provider=embedding_provider,
        vector_service=vector_svc,
        bm25_retriever=bm25_svc,
    )
    rerank_svc = RerankerService(
        reranker_provider=reranker_provider,
        hybrid_retriever=hybrid_svc,
    )

    results = {
        "vector_only": {"mrr": [], "recall_1": [], "recall_3": [], "ndcg_3": []},
        "bm25_only": {"mrr": [], "recall_1": [], "recall_3": [], "ndcg_3": []},
        "hybrid_rrf": {"mrr": [], "recall_1": [], "recall_3": [], "ndcg_3": []},
        "reranked": {"mrr": [], "recall_1": [], "recall_3": [], "ndcg_3": []},
    }

    per_query_details = []

    for q in EVAL_QUERIES:
        query_text = q["query"]
        relevant = q["relevant_chunks"]

        # 1. Vector only
        q_vec = embedding_provider.embed_query(query_text)
        vec_hits = vector_svc.search(query_vector=q_vec, limit=5)
        vec_ids = [h["payload"]["chunk_id"] for h in vec_hits]

        # 2. BM25 only
        bm25_hits = bm25_svc.search(query=query_text, limit=5)
        bm25_ids = [h.chunk_id for h in bm25_hits]

        # 3. Hybrid RRF
        hybrid_resp = hybrid_svc.search(query=query_text, limit=5)
        hybrid_ids = [h.chunk_id for h in hybrid_resp.results]

        # 4. Cross-Encoder Reranked
        rerank_resp = rerank_svc.search_and_rerank(
            query=query_text, limit=5, candidate_limit=5
        )
        rerank_ids = [r.chunk_id for r in rerank_resp.results]

        pipelines = {
            "vector_only": vec_ids,
            "bm25_only": bm25_ids,
            "hybrid_rrf": hybrid_ids,
            "reranked": rerank_ids,
        }

        query_detail = {
            "query_id": q["query_id"],
            "type": q["type"],
            "query": query_text,
            "pipelines": {},
        }

        for pipe_name, ranked_ids in pipelines.items():
            mrr_val = reciprocal_rank(ranked_ids, relevant)
            r1_val = recall_at_k(ranked_ids, relevant, 1)
            r3_val = recall_at_k(ranked_ids, relevant, 3)
            ndcg_val = ndcg_at_k(ranked_ids, relevant, 3)

            results[pipe_name]["mrr"].append(mrr_val)
            results[pipe_name]["recall_1"].append(r1_val)
            results[pipe_name]["recall_3"].append(r3_val)
            results[pipe_name]["ndcg_3"].append(ndcg_val)

            query_detail["pipelines"][pipe_name] = {
                "top_1": ranked_ids[0] if ranked_ids else None,
                "mrr": round(mrr_val, 4),
                "recall_1": round(r1_val, 4),
                "ndcg_3": round(ndcg_val, 4),
            }

        per_query_details.append(query_detail)

    # Compute aggregate means across all queries
    summary = {}
    for pipe_name, metric_dict in results.items():
        summary[pipe_name] = {
            "mean_mrr": round(
                sum(metric_dict["mrr"]) / max(len(metric_dict["mrr"]), 1), 4
            ),
            "mean_recall_1": round(
                sum(metric_dict["recall_1"]) / max(len(metric_dict["recall_1"]), 1), 4
            ),
            "mean_recall_3": round(
                sum(metric_dict["recall_3"]) / max(len(metric_dict["recall_3"]), 1), 4
            ),
            "mean_ndcg_3": round(
                sum(metric_dict["ndcg_3"]) / max(len(metric_dict["ndcg_3"]), 1), 4
            ),
        }

    return {
        "summary": summary,
        "details": per_query_details,
        "reranker_model": reranker_provider.model_name,
        "embedding_model": embedding_provider.model_name,
    }


if __name__ == "__main__":
    print("Running Retrieval Quality Evaluation with real models...")
    emb_provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    rerank_provider = FlashRankRerankerProvider(model_name="ms-marco-TinyBERT-L-2-v2")

    report = run_retrieval_evaluation(emb_provider, rerank_provider)
    print("\n=== Aggregate Retrieval Quality Benchmark ===")
    print(f"Embedding Model: {report['embedding_model']}")
    print(f"Reranker Model:  {report['reranker_model']}\n")
    header = (
        f"{'Pipeline':<15} | {'MRR':<8} | {'Recall@1':<10} | "
        f"{'Recall@3':<10} | {'nDCG@3':<8}"
    )
    print(header)

    print("-" * len(header))
    for pipe, metrics in report["summary"].items():
        line = (
            f"{pipe:<15} | {metrics['mean_mrr']:<8.4f} | "
            f"{metrics['mean_recall_1']:<10.4f} | "
            f"{metrics['mean_recall_3']:<10.4f} | "
            f"{metrics['mean_ndcg_3']:<8.4f}"
        )
        print(line)

    print("\n=== Per-Query Results ===")
    for d in report["details"]:
        print(f"\n[{d['query_id']}] {d['query']} ({d['type']})")
        for pipe, res in d["pipelines"].items():
            print(
                f"  {pipe:<12}: top-1={res['top_1']} | "
                f"MRR={res['mrr']} | nDCG@3={res['ndcg_3']}"
            )
