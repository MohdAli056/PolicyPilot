"""Semantic vector search API router."""

import asyncio

from fastapi import APIRouter, HTTPException, status

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import search_service

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search over policy document chunks",
)
async def semantic_search(request: SearchRequest) -> SearchResponse:
    """Retrieve semantically relevant document chunks matching the query."""
    try:
        # Run synchronous embedding generation and Qdrant search in threadpool
        response = await asyncio.to_thread(
            search_service.search,
            query=request.query,
            limit=request.limit,
        )
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Semantic search service error: {exc}",
        ) from exc
