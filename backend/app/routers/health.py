"""Health and readiness check router.

Provides:
- GET /health: A lightweight liveness probe verifying the FastAPI event loop
  is responsive (independent of downstream dependencies).
- GET /health/ready: A readiness probe evaluating storage filesystem access,
  vector database connectivity, and retrieval model provider readiness.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.config import settings
from app.services.embedding_service import get_embedding_provider
from app.services.reranker_service import reranker_service
from app.services.storage_service import storage_service
from app.services.vector_service import get_vector_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Schema for the lightweight liveness probe response."""

    status: str
    service: str
    version: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Schema for the operational readiness probe response."""

    status: str
    service: str
    version: str
    timestamp: str
    checks: dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return current process liveness status."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(response: Response) -> ReadinessResponse:
    """Evaluate readiness of local storage, vector store, and AI models."""
    checks: dict[str, Any] = {}
    is_ready = True

    # 1. Storage filesystem checks
    try:
        storage_ok = (
            storage_service.documents_dir.exists()
            and storage_service.metadata_dir.exists()
            and storage_service.chunks_dir.exists()
        )
        checks["storage"] = {
            "status": "ready" if storage_ok else "unhealthy",
            "documents_dir": str(storage_service.documents_dir),
            "chunks_dir": str(storage_service.chunks_dir),
        }
        if not storage_ok:
            is_ready = False
    except Exception as exc:
        logger.error("Readiness check storage error: %s", exc)
        checks["storage"] = {"status": "error", "error": str(exc)}
        is_ready = False

    # 2. Vector database (Qdrant) check
    try:
        vec_svc = get_vector_service()
        vec_svc.ensure_collection()
        checks["vector_database"] = {
            "status": "ready",
            "provider": "qdrant",
            "collection": vec_svc.default_collection,
            "dimension": vec_svc.default_dimension,
        }
    except Exception as exc:
        logger.error("Readiness check vector database error: %s", exc)
        checks["vector_database"] = {"status": "error", "error": str(exc)}
        is_ready = False

    # 3. Embedding model check
    try:
        embed_provider = get_embedding_provider()
        checks["embedding_provider"] = {
            "status": "ready",
            "provider": settings.embedding_provider,
            "model": embed_provider.model_name,
            "dimension": embed_provider.dimension,
        }
    except Exception as exc:
        logger.error("Readiness check embedding provider error: %s", exc)
        checks["embedding_provider"] = {"status": "error", "error": str(exc)}
        is_ready = False

    # 4. Cross-encoder reranker check
    try:
        checks["reranker_provider"] = {
            "status": "ready",
            "provider": settings.reranker_provider,
            "model": reranker_service.reranker_provider.model_name,
        }
    except Exception as exc:
        logger.error("Readiness check reranker provider error: %s", exc)
        checks["reranker_provider"] = {"status": "error", "error": str(exc)}
        is_ready = False

    # 5. LLM provider configuration status (informational, not blocking)
    llm_configured = bool(settings.gemini_api_key or settings.llm_provider == "mock")
    checks["llm_provider"] = {
        "status": "configured" if llm_configured else "unconfigured",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "note": (
            "Ready for inference"
            if llm_configured
            else "GEMINI_API_KEY is not set; retrieval active, QA requires key"
        ),
    }

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
    )
