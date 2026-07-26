from fastapi.testclient import TestClient


def test_health_check_endpoint(client: TestClient) -> None:
    """Test /api/v1/health returns healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
