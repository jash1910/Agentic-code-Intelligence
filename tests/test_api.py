"""API integration tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.app.main import app
    return TestClient(app)


class TestHealthAPI:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agents" in data
        assert len(data["agents"]) == 7


class TestAuthAPI:
    def test_login_success(self, client):
        response = client.post("/api/v1/auth/login", json={
            "email": "admin@archon.ai",
            "password": "admin123",
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_failure(self, client):
        response = client.post("/api/v1/auth/login", json={
            "email": "admin@archon.ai",
            "password": "wrong",
        })
        assert response.status_code == 401


class TestRepositoriesAPI:
    def test_list_repositories(self, client):
        response = client.get("/api/v1/chat/repositories")
        assert response.status_code == 200
        repos = response.json()["repositories"]
        assert "langchain" in repos
