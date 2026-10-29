"""PolicyPilot FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import documents, health, search
from app.services.vector_service import reset_vector_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for clean startup and resource release."""
    yield
    reset_vector_service()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered company policy assistant",
    lifespan=lifespan,
)

# Configure CORS so the frontend dev server can reach the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
