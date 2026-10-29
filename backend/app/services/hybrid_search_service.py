"""Hybrid Search Service combining Dense Semantic Vector and Sparse BM25 Retrieval.

Merges candidate rankings using Reciprocal Rank Fusion (RRF) with configurable
constant k and stable secondary tie-breaking.
"""

from typing import Optional

from app.config import settings
from app.schemas.hybrid_search import (
    HybridSearchResponse,
    HybridSearchResultItem,
)
from app.services.bm25_service import BM25Service, bm25_service
from app.services.embedding_service import BaseEmbeddingProvider, get_embedding_provider
from app.services.vector_service import VectorService, get_vector_service


def compute_rrf_score(ranks: list[int], k: int) -> float:
    """Calculate Reciprocal Rank Fusion score for 1-based ranks.

    Formula:
        RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be strictly positive, got {k}")
    score = 0.0
    for r in ranks:
        if r <= 0:
            raise ValueError(
                f"Retriever rank must be 1-based positive integer, got {r}"
            )
        score += 1.0 / (k + r)
    return score


class HybridSearchService:
    """Coordinates semantic vector retrieval, BM25 retrieval, and RRF rank fusion."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_service: Optional[VectorService] = None,
        bm25_retriever: Optional[BM25Service] = None,
    ):
        self._custom_embedding_provider = embedding_provider
        self._custom_vector_service = vector_service
        self._custom_bm25_service = bm25_retriever

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        return self._custom_embedding_provider or get_embedding_provider()

    @property
    def vector_service(self) -> VectorService:
        return self._custom_vector_service or get_vector_service()

    @property
    def bm25_retriever(self) -> BM25Service:
        return self._custom_bm25_service or bm25_service

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> HybridSearchResponse:
        """Execute hybrid search combining vector and BM25 retrievers via RRF."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty")

        eff_limit = limit if limit is not None else settings.default_search_limit
        if eff_limit <= 0:
            raise ValueError(f"Search limit must be positive, got {eff_limit}")
        if eff_limit > settings.max_search_limit:
            eff_limit = settings.max_search_limit

        eff_k = rrf_k if rrf_k is not None else settings.rrf_k
        if eff_k <= 0:
            raise ValueError(f"RRF constant k must be strictly positive, got {eff_k}")

        # Retrieve up to candidate_limit from each system
        candidate_limit = max(eff_limit, settings.hybrid_candidate_limit)

        # 1. Run Semantic Vector Retrieval
        # Failures are not swallowed; an honest error is raised if vector
        # retriever fails
        query_vector = self.embedding_provider.embed_query(clean_query)
        vector_hits = self.vector_service.search(
            query_vector=query_vector,
            limit=candidate_limit,
        )

        # 2. Run BM25 Keyword Retrieval
        bm25_hits = self.bm25_retriever.search(
            query=clean_query,
            limit=candidate_limit,
        )

        # Map candidates by chunk_id to unify metadata and assign 1-based ranks
        candidates_meta: dict[str, dict] = {}
        vector_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        for rank_idx, hit in enumerate(vector_hits, start=1):
            payload = hit.get("payload", {})
            chunk_id = payload.get("chunk_id", hit.get("id", ""))
            vector_ranks[chunk_id] = rank_idx
            if chunk_id not in candidates_meta:
                candidates_meta[chunk_id] = {
                    "document_id": payload.get("document_id", ""),
                    "filename": payload.get("filename", "unknown"),
                    "chunk_id": chunk_id,
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "section": payload.get("section"),
                    "page_numbers": payload.get("page_numbers", []),
                    "character_count": payload.get(
                        "character_count", len(payload.get("text", ""))
                    ),
                }

        for rank_idx, hit in enumerate(bm25_hits, start=1):
            chunk_id = hit.chunk_id
            bm25_ranks[chunk_id] = rank_idx
            if chunk_id not in candidates_meta:
                candidates_meta[chunk_id] = {
                    "document_id": hit.document_id,
                    "filename": hit.filename,
                    "chunk_id": chunk_id,
                    "chunk_index": hit.chunk_index,
                    "text": hit.text,
                    "section": hit.section,
                    "page_numbers": hit.page_numbers,
                    "character_count": hit.character_count,
                }

        # 3. Reciprocal Rank Fusion
        fused_items: list[HybridSearchResultItem] = []
        for chunk_id, meta in candidates_meta.items():
            vr = vector_ranks.get(chunk_id)
            kr = bm25_ranks.get(chunk_id)

            ranks_list: list[int] = []
            retrievers: list[str] = []
            if vr is not None:
                ranks_list.append(vr)
                retrievers.append("vector")
            if kr is not None:
                ranks_list.append(kr)
                retrievers.append("bm25")

            rrf_score = compute_rrf_score(ranks_list, eff_k)

            fused_items.append(
                HybridSearchResultItem(
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    chunk_id=meta["chunk_id"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    section=meta["section"],
                    page_numbers=meta["page_numbers"],
                    character_count=meta["character_count"],
                    vector_rank=vr,
                    keyword_rank=kr,
                    rrf_score=round(rrf_score, 6),
                    retrievers=retrievers,
                )
            )

        # 4. Deterministic Ordering
        # Primary: fused RRF score descending
        # Secondary: chunk_id ascending (stable deterministic tie-breaking)
        fused_items.sort(key=lambda item: (-item.rrf_score, item.chunk_id))

        final_results = fused_items[:eff_limit]

        return HybridSearchResponse(
            query=clean_query,
            total_results=len(final_results),
            rrf_k=eff_k,
            results=final_results,
        )


hybrid_search_service = HybridSearchService()
