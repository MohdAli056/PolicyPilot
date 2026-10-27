"""Health check router.

Provides a lightweight endpoint for verifying that the backend service
is running and responsive. Intentionally independent of external
services (databases, AI providers, etc.) so it always reflects the
server's own liveness.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Schema for the health-check response."""

    status: str
    service: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return current service health."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
