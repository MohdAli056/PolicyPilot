"""Service for semantic vector search over indexed document chunks."""

from typing import Optional

from app.config import settings
from app.schemas.search import SearchResponse, SearchResultItem
from app.services.embedding_service import BaseEmbeddingProvider, get_embedding_provider
from app.services.vector_service import VectorService, get_vector_service


class SearchService:
    """Coordinates query embedding and semantic retrieval from Qdrant."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_service: Optional[VectorService] = None,
    ):
        self._custom_embedding_provider = embedding_provider
        self._custom_vector_service = vector_service

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        return self._custom_embedding_provider or get_embedding_provider()

    @property
    def vector_service(self) -> VectorService:
        return self._custom_vector_service or get_vector_service()

    def search(self, query: str, limit: Optional[int] = None) -> SearchResponse:
        """Embed user query and retrieve top matching chunks from Qdrant."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty")

        eff_limit = limit if limit is not None else settings.default_search_limit
        if eff_limit <= 0:
            raise ValueError(f"Search limit must be positive, got {eff_limit}")
        if eff_limit > settings.max_search_limit:
            eff_limit = settings.max_search_limit

        # 1. Generate query vector using active embedding provider
        query_vector = self.embedding_provider.embed_query(clean_query)

        # 2. Search Qdrant vector database
        hits = self.vector_service.search(query_vector=query_vector, limit=eff_limit)

        # 3. Format results into structured response
        results: list[SearchResultItem] = []
        for hit in hits:
            payload = hit.get("payload", {})
            results.append(
                SearchResultItem(
                    document_id=payload.get("document_id", ""),
                    filename=payload.get("filename", "unknown"),
                    chunk_id=payload.get("chunk_id", hit.get("id", "")),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    section=payload.get("section"),
                    page_numbers=payload.get("page_numbers", []),
                    character_count=payload.get(
                        "character_count", len(payload.get("text", ""))
                    ),
                    score=round(hit.get("score", 0.0), 4),
                )
            )

        return SearchResponse(
            query=clean_query,
            total_results=len(results),
            results=results,
        )


search_service = SearchService()
