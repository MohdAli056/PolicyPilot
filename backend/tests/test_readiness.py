"""Tests for Liveness and Operational Readiness Probes (Phase 8)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.services.vector_service import VectorService

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_qdrant_for_readiness(monkeypatch: pytest.MonkeyPatch):
    """Ensure readiness tests use an isolated in-memory Qdrant instance."""
    mem_client = QdrantClient(location=":memory:")
    mem_vec = VectorService(
        client=mem_client,
        default_collection="policypilot_chunks",
        default_dimension=384,
    )
    mem_vec.ensure_collection()
    monkeypatch.setattr("app.routers.health.get_vector_service", lambda: mem_vec)


def test_liveness_probe_returns_ok():
    """GET /health returns 200 OK with process metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "PolicyPilot" in data["service"]
    assert "timestamp" in data


def test_readiness_probe_returns_ready_under_normal_operation():
    """GET /health/ready returns 200 OK and inspects all subsystems."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    checks = data["checks"]
    assert "storage" in checks
    assert checks["storage"]["status"] == "ready"
    assert "vector_database" in checks
    assert checks["vector_database"]["status"] == "ready"
    assert "embedding_provider" in checks
    assert checks["embedding_provider"]["status"] == "ready"
    assert "reranker_provider" in checks
    assert checks["reranker_provider"]["status"] == "ready"
    assert "llm_provider" in checks


def test_readiness_probe_returns_503_on_storage_failure():
    """GET /health/ready returns 503 if storage paths are inaccessible."""
    mock_storage = MagicMock()
    mock_storage.documents_dir.exists.return_value = False
    mock_storage.metadata_dir.exists.return_value = True
    mock_storage.chunks_dir.exists.return_value = True

    with patch("app.routers.health.storage_service", mock_storage):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["storage"]["status"] == "unhealthy"


def test_readiness_probe_returns_503_on_vector_failure():
    """GET /health/ready returns 503 if vector store collection fails."""
    mock_vec = MagicMock()
    mock_vec.ensure_collection.side_effect = RuntimeError("Qdrant offline")

    with patch("app.routers.health.get_vector_service", return_value=mock_vec):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["vector_database"]["status"] == "error"
