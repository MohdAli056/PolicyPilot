"""Search API router supporting Semantic Vector, BM25 Keyword, and RRF Hybrid search."""

import asyncio

from fastapi import APIRouter, HTTPException, status

from app.schemas.hybrid_search import (
    HybridSearchRequest,
    HybridSearchResponse,
    KeywordSearchResponse,
)
from app.schemas.search import SearchRequest, SearchResponse
from app.services.bm25_service import bm25_service
from app.services.hybrid_search_service import hybrid_search_service
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


@router.post(
    "/keyword",
    response_model=KeywordSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="BM25 keyword search over policy document chunks",
)
async def keyword_search(request: SearchRequest) -> KeywordSearchResponse:
    """Retrieve document chunks matching query keywords using Okapi BM25."""
    try:
        results = await asyncio.to_thread(
            bm25_service.search,
            query=request.query,
            limit=request.limit,
        )
        return KeywordSearchResponse(
            query=request.query.strip(),
            total_results=len(results),
            results=results,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Keyword search service error: {exc}",
        ) from exc


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid search combining semantic vectors and BM25 keywords via RRF",
)
async def hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    """Retrieve and fuse candidate chunks from vector and BM25 search using RRF."""
    try:
        response = await asyncio.to_thread(
            hybrid_search_service.search,
            query=request.query,
            limit=request.limit,
            rrf_k=request.rrf_k,
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
            detail=f"Hybrid search service error: {exc}",
        ) from exc
