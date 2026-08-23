from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.worker.celery_app import celery_app

celery_app.conf.update(
    broker_url="memory://",
    result_backend="cache+memory://",
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,
)

_EMBEDDING_PAYLOAD = {
    "document_id": "doc-1",
    "object_key": "docs/handbook.pdf",
    "chunks": [{"chunk_index": 0, "content": "chunk content", "region_type": "text"}],
}


def test_embedding_returns_400_when_trusted_headers_missing(client: TestClient) -> None:
    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD)

    assert response.status_code == 400


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_embedding_returns_202_and_task_id_for_valid_request(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    mock_embedder_cls.return_value.embed.return_value = [[0.1], [0.2], [0.3]]
    mock_enricher_cls.return_value.enrich.return_value = MagicMock(
        summary="a summary", questions=["Q1?", "Q2?"]
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    response = client.post(
        "/api/v1/ingestion/embedding",
        json=_EMBEDDING_PAYLOAD,
        headers={"X-User-Department": "CNTT", "X-User-Access-Level": "STUDENT"},
    )

    assert response.status_code == 202
    assert response.json()["task_id"]


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_websocket_receives_progress_frames_and_terminal_state(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    mock_embedder_cls.return_value.embed.return_value = [[0.1], [0.2], [0.3]]
    mock_enricher_cls.return_value.enrich.return_value = MagicMock(
        summary="a summary", questions=["Q1?", "Q2?"]
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    response = client.post(
        "/api/v1/ingestion/embedding",
        json=_EMBEDDING_PAYLOAD,
        headers={"X-User-Department": "CNTT", "X-User-Access-Level": "STUDENT"},
    )
    task_id = response.json()["task_id"]

    with client.websocket_connect(f"/api/v1/ingestion/embedding/{task_id}/progress") as ws:
        frame = ws.receive_json()
        assert frame["state"] in {"PROGRESS", "SUCCESS"}
        while frame["state"] not in {"SUCCESS", "FAILURE"}:
            frame = ws.receive_json()

    assert frame["state"] == "SUCCESS"
    assert frame["percent"] == 100
