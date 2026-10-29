"""Cross-Encoder Reranking Service for PolicyPilot.

Reranks candidate passages returned by hybrid retrieval (semantic + BM25 with RRF)
using a lightweight cross-encoder model to produce a final, relevance-calibrated
ranking. Supports pluggable providers (FlashRank ONNX cross-encoders and
deterministic Mock for offline testing).
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings
from app.schemas.hybrid_search import HybridSearchResultItem
from app.schemas.rerank_search import (
    RerankSearchResponse,
    RerankSearchResultItem,
)
from app.services.hybrid_search_service import (
    HybridSearchService,
    hybrid_search_service,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Provider Abstraction
# ============================================================================


class BaseRerankerProvider(ABC):
    """Abstract base class for cross-encoder rerankers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the unique identifier or name of the reranker model."""
        pass

    @abstractmethod
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score candidate passage texts against the query.

        Args:
            query: The user query string.
            texts: List of candidate chunk passages to score.

        Returns:
            A list of float relevance scores corresponding 1-to-1 to input texts.
            Higher scores indicate greater relevance.
        """
        pass


class FlashRankRerankerProvider(BaseRerankerProvider):
    """Cross-encoder provider using FlashRank's ONNX-optimized models.

    Runs natively on CPU using ONNX Runtime with zero heavy PyTorch dependencies,
    low memory consumption, and millisecond latency.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.reranker_model
        self._ranker = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_ranker(self):
        """Lazily initialize the FlashRank Ranker singleton with thread safety."""
        if self._ranker is None:
            with self._lock:
                if self._ranker is None:
                    try:
                        from flashrank import Ranker

                        logger.info(
                            "Initializing FlashRank cross-encoder model: %s",
                            self._model_name,
                        )
                        self._ranker = Ranker(model_name=self._model_name)
                    except Exception as exc:
                        logger.error(
                            "Failed to initialize FlashRank model '%s': %s",
                            self._model_name,
                            exc,
                        )
                        raise RuntimeError(
                            f"FlashRank init failed for '{self._model_name}': {exc}"
                        ) from exc
        return self._ranker

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score passages using the FlashRank cross-encoder."""
        if not texts:
            return []

        ranker = self._get_ranker()

        try:
            from flashrank import RerankRequest

            passages = [{"id": i, "text": t} for i, t in enumerate(texts)]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            # FlashRank returns a sorted list of dicts with 'id' and 'score'
            # Map back to original order to ensure 1-to-1 correspondence with texts
            score_map: dict[int, float] = {}
            for item in results:
                raw_score = item.get("score", 0.0)
                score_map[item["id"]] = float(raw_score)

            return [score_map.get(i, 0.0) for i in range(len(texts))]
        except Exception as exc:
            logger.error(
                "Cross-encoder inference failed for query '%s' on %d passages: %s",
                query,
                len(texts),
                exc,
            )
            raise RuntimeError(f"Cross-encoder inference failed: {exc}") from exc


class MockRerankerProvider(BaseRerankerProvider):
    """Deterministic mock reranker for unit testing and offline verification.

    Calculates scores deterministically using term overlap, exact matches,
    and string length, with support for explicit score overrides and forced failures.
    """

    def __init__(
        self,
        model_name: str = "mock-cross-encoder-v1",
        score_overrides: Optional[dict[str, float]] = None,
        force_failure: bool = False,
    ):
        self._model_name = model_name
        self._score_overrides = score_overrides or {}
        self._force_failure = force_failure

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_score_override(self, chunk_id_or_text: str, score: float) -> None:
        """Explicitly override the mock score for a specific chunk or text."""
        self._score_overrides[chunk_id_or_text] = score

    def set_force_failure(self, fail: bool) -> None:
        """Configure the mock provider to simulate an honest inference failure."""
        self._force_failure = fail

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Compute deterministic mock relevance scores."""
        if self._force_failure:
            raise RuntimeError("Simulated cross-encoder inference failure")

        if not texts:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        scores: list[float] = []

        for idx, text in enumerate(texts):
            # Check explicit score override first
            if text in self._score_overrides:
                scores.append(self._score_overrides[text])
                continue

            text_lower = text.lower()
            text_terms = re.findall(r"\w+", text_lower)

            if not text_terms:
                scores.append(0.0)
                continue

            # Deterministic scoring:
            # 1. Term overlap ratio
            matched = sum(1 for term in text_terms if term in query_terms)
            overlap_score = matched / max(len(text_terms), 1)

            # 2. Exact substring bonus
            phrase_bonus = 0.5 if query.lower() in text_lower else 0.0

            # 3. Normalized score in [0.0, 1.0]
            base_score = min(1.0, (overlap_score * 0.7) + phrase_bonus)
            scores.append(round(base_score, 6))

        return scores


# Global provider instance management
_global_reranker_provider: Optional[BaseRerankerProvider] = None


def get_reranker_provider() -> BaseRerankerProvider:
    """Retrieve or instantiate the application's configured reranker provider."""
    global _global_reranker_provider
    if _global_reranker_provider is None:
        provider_type = settings.reranker_provider.lower()
        if provider_type == "mock":
            _global_reranker_provider = MockRerankerProvider()
        elif provider_type == "flashrank":
            _global_reranker_provider = FlashRankRerankerProvider(
                model_name=settings.reranker_model
            )
        else:
            raise ValueError(f"Unsupported reranker provider: {provider_type}")
    return _global_reranker_provider


def set_reranker_provider(provider: Optional[BaseRerankerProvider]) -> None:
    """Set or override the active reranker provider (used in tests)."""
    global _global_reranker_provider
    _global_reranker_provider = provider


# ============================================================================
# Reranker Service
# ============================================================================


class RerankerService:
    """Coordinates hybrid candidate retrieval, cross-encoder scoring, and ranking."""

    def __init__(
        self,
        reranker_provider: Optional[BaseRerankerProvider] = None,
        hybrid_retriever: Optional[HybridSearchService] = None,
    ):
        self._custom_reranker_provider = reranker_provider
        self._custom_hybrid_service = hybrid_retriever

    @property
    def reranker_provider(self) -> BaseRerankerProvider:
        return self._custom_reranker_provider or get_reranker_provider()

    @property
    def hybrid_retriever(self) -> HybridSearchService:
        return self._custom_hybrid_service or hybrid_search_service

    def search_and_rerank(
        self,
        query: str,
        limit: Optional[int] = None,
        candidate_limit: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> RerankSearchResponse:
        """Execute hybrid search, score top candidates with cross-encoder, and sort.

        Pipeline:
            Query → semantic retrieval + BM25 retrieval → RRF fusion
                  → cross-encoder reranking → final ranked passages
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty")

        eff_limit = limit if limit is not None else settings.default_search_limit
        if eff_limit <= 0:
            raise ValueError(f"Search limit must be positive, got {eff_limit}")
        if eff_limit > settings.max_search_limit:
            eff_limit = settings.max_search_limit

        eff_candidate_limit = (
            candidate_limit
            if candidate_limit is not None
            else settings.rerank_candidate_limit
        )
        if eff_candidate_limit <= 0:
            raise ValueError(
                f"Candidate limit must be positive, got {eff_candidate_limit}"
            )

        # Ensure candidate pool is large enough to satisfy both limits
        pool_size = max(eff_limit, eff_candidate_limit)

        # 1. Retrieval Stage: Hybrid Search (Semantic + BM25 with RRF)
        t_start = time.perf_counter()
        hybrid_response = self.hybrid_retriever.search(
            query=clean_query,
            limit=pool_size,
            rrf_k=rrf_k,
        )
        t_retrieval = time.perf_counter()

        candidates: list[HybridSearchResultItem] = hybrid_response.results

        # Handle empty candidate pool safely
        if not candidates:
            total_elapsed_ms = round((t_retrieval - t_start) * 1000, 2)
            retrieval_ms = total_elapsed_ms
            return RerankSearchResponse(
                query=clean_query,
                total_results=0,
                reranker_model=self.reranker_provider.model_name,
                candidate_limit=pool_size,
                results=[],
                timing_ms={
                    "retrieval_ms": retrieval_ms,
                    "rerank_ms": 0.0,
                    "total_ms": total_elapsed_ms,
                },
            )

        # 2. Cross-Encoder Reranking Stage
        candidate_texts = [item.text for item in candidates]
        t_rerank_start = time.perf_counter()
        scores = self.reranker_provider.rerank(clean_query, candidate_texts)
        t_rerank_end = time.perf_counter()

        if len(scores) != len(candidates):
            raise RuntimeError(
                f"Reranker produced {len(scores)} scores for {len(candidates)} items"
            )

        # 3. Merge Rerank Scores with Candidate Metadata
        scored_candidates: list[RerankSearchResultItem] = []
        for candidate, score in zip(candidates, scores):
            scored_candidates.append(
                RerankSearchResultItem(
                    document_id=candidate.document_id,
                    filename=candidate.filename,
                    chunk_id=candidate.chunk_id,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    section=candidate.section,
                    page_numbers=candidate.page_numbers,
                    character_count=candidate.character_count,
                    vector_rank=candidate.vector_rank,
                    keyword_rank=candidate.keyword_rank,
                    rrf_score=candidate.rrf_score,
                    retrievers=candidate.retrievers,
                    rerank_score=round(float(score), 6),
                    rerank_rank=0,  # Will be assigned after sorting
                )
            )

        # 4. Deterministic Sort:
        # Primary: descending cross-encoder relevance score
        # Secondary: stable tie-breaking on chunk_id
        scored_candidates.sort(key=lambda item: (-item.rerank_score, item.chunk_id))

        # 5. Assign 1-based Rerank Rank
        for rank_idx, item in enumerate(scored_candidates, start=1):
            item.rerank_rank = rank_idx

        # 6. Slice to Final Requested Limit
        final_results = scored_candidates[:eff_limit]

        retrieval_ms = round((t_retrieval - t_start) * 1000, 2)
        rerank_ms = round((t_rerank_end - t_rerank_start) * 1000, 2)
        total_ms = round((t_rerank_end - t_start) * 1000, 2)

        return RerankSearchResponse(
            query=clean_query,
            total_results=len(final_results),
            reranker_model=self.reranker_provider.model_name,
            candidate_limit=len(candidates),
            results=final_results,
            timing_ms={
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "total_ms": total_ms,
            },
        )


reranker_service = RerankerService()
