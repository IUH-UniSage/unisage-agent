from fastapi.testclient import TestClient

from app.api.v1 import health as health_module


def _all_dependencies_up(monkeypatch) -> None:
    async def _fake_check_database() -> dict[str, object]:
        return {"status": "up", "response_time_ms": 1}

    def _fake_sync_check() -> dict[str, object]:
        return {"status": "up", "response_time_ms": 1}

    monkeypatch.setattr(health_module, "_check_database", _fake_check_database)
    monkeypatch.setattr(health_module, "_check_redis_sync", _fake_sync_check)
    monkeypatch.setattr(health_module, "_check_qdrant_sync", _fake_sync_check)


def test_health_check_endpoint(client: TestClient, monkeypatch) -> None:
    """Test /api/v1/health returns healthy status when every dependency is up."""
    _all_dependencies_up(monkeypatch)
    monkeypatch.setattr(health_module, "outbox_health", lambda: {"pending": 0, "dead": 0})

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 1000
    data = body["data"]
    assert data["status"] == "healthy"
    assert "service" in data
    assert data["usageOutbox"] == {"pending": 0, "dead": 0}


def test_health_reports_degraded_when_dead_letter_queue_is_non_empty(
    client: TestClient, monkeypatch
) -> None:
    _all_dependencies_up(monkeypatch)
    monkeypatch.setattr(health_module, "outbox_health", lambda: {"pending": 2, "dead": 3})

    response = client.get("/api/v1/health")

    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["usageOutbox"] == {"pending": 2, "dead": 3}
