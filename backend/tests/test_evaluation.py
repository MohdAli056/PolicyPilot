"""Automated Pytest Regression Tests for RAG Evaluation (Phase 8).

Verifies multi-stage retrieval metrics and QA/citation quality across the
deterministic evaluation dataset.
"""

import pytest

from tests.evaluate_rag import RAGEvaluator


@pytest.fixture(scope="module")
def rag_evaluator():
    """Module-scoped evaluator instance to avoid redundant indexing."""
    return RAGEvaluator()


def test_retrieval_metrics_meet_quality_thresholds(rag_evaluator):
    """Multi-stage retrieval achieves high recall and ranking performance."""
    metrics = rag_evaluator.evaluate_retrieval()

    # Reranked pipeline quality thresholds
    reranked = metrics["reranked"]
    assert reranked["recall@3"] >= 0.80, "Reranked Recall@3 should be at least 80%"
    assert reranked["mrr"] >= 0.70, "Reranked MRR should be at least 0.70"
    assert reranked["ndcg@3"] >= 0.80, "Reranked nDCG@3 should be at least 0.80"
    assert reranked["multi_coverage"] >= 0.80, (
        "Multi-evidence coverage should be at least 80%"
    )

    # Hybrid RRF baseline
    hybrid = metrics["hybrid_rrf"]
    assert hybrid["recall@3"] >= 0.80, "Hybrid Recall@3 should be at least 80%"


def test_qa_and_citations_quality_and_abstention(rag_evaluator):
    """End-to-end QA achieves verified answer correctness and abstention."""
    qa_results = rag_evaluator.evaluate_qa_and_citations()

    assert qa_results["answer_correctness_rate"] >= 0.80, (
        "Answer correctness rate should be >= 80%"
    )
    assert qa_results["abstention_precision"] == 1.0, (
        "Unanswerable questions must abstain with 100% precision"
    )
    assert qa_results["citation_validity_rate"] == 1.0, (
        "All generated citation IDs must point to verified retrieved sources"
    )
    assert qa_results["page_accuracy_rate"] >= 0.80, (
        "Page attribution accuracy should be >= 80%"
    )
    assert qa_results["prompt_injection_neutralized"] is True, (
        "Prompt injection attempts in policy text must be neutralized"
    )
    assert qa_results["conflict_coverage_passed"] is True, (
        "Conflicting policy clauses must be preserved in retrieved context"
    )
