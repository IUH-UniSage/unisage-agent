from fastapi.testclient import TestClient


def test_chat_endpoint(client: TestClient) -> None:
    """Test chat returns a grounded response, citations, and trace ID."""

    payload = {
        "query": "Hạn đăng ký môn học học kỳ này là khi nào?",
        "user_faculty": "GLOBAL",
        "user_level": 1,
    }
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 1000
    data = body["data"]
    assert "trace_id" in data
    assert data["query"] == payload["query"]
    assert data["intent"] == "SINGLE_INTENT"
    assert data["citations"]
    assert data["suggestions"]
    assert "UniSage" in data["response"]
