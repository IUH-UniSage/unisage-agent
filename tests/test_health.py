from fastapi.testclient import TestClient


def test_health_check_endpoint(client: TestClient) -> None:
    """Test /api/v1/health returns healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 1000
    data = body["data"]
    assert data["status"] == "healthy"
    assert "service" in data
