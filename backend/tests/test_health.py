"""Tests for the health-check endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Verify GET /health behaviour and response schema."""

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self) -> None:
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_status_is_healthy(self) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_service_name(self) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "PolicyPilot"

    def test_health_version_present(self) -> None:
        response = client.get("/health")
        data = response.json()
        # Version should be a non-empty string
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_timestamp_is_iso_format(self) -> None:
        from datetime import datetime

        response = client.get("/health")
        data = response.json()
        # Should parse without error
        parsed = datetime.fromisoformat(data["timestamp"])
        assert parsed is not None

    def test_health_content_type_json(self) -> None:
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
