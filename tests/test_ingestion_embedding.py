import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.worker.celery_app import celery_app
from tests.fixtures.documents import make_pdf_bytes

celery_app.conf.update(
    broker_url="memory://",
    result_backend="cache+memory://",
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,
)

_EMBEDDING_PAYLOAD = {
    "document_id": "doc-1",
    "department_id": "CNTT",
    "access_level": 2,
    "object_key": "docs/handbook.pdf",
    "chunks": [{"chunk_index": 0, "content": "chunk content", "region_type": "text"}],
}

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


def test_embedding_returns_400_when_trusted_headers_missing(client: TestClient) -> None:
    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD)

    assert response.status_code == 400


def test_embedding_returns_403_when_missing_document_permission(client: TestClient) -> None:
    headers = {**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


def test_embedding_returns_403_when_department_not_granted(client: TestClient) -> None:
    headers = {
        **_TRUSTED_HEADERS,
        "X-User-Department-Access": json.dumps(
            [{"department_id": "KHOA_KINH_TE", "access_level": 5}]
        ),
    }

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


def test_embedding_returns_403_when_access_level_exceeds_grant(client: TestClient) -> None:
    headers = {
        **_TRUSTED_HEADERS,
        "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 1}]),
    }

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


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
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202
    assert response.json()["task_id"]


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_valid_request_dispatches_task_with_request_department_and_level(
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

    client.post(
        "/api/v1/ingestion/embedding",
        json=_EMBEDDING_PAYLOAD,
        headers=_TRUSTED_HEADERS,
    )

    point_kwargs = mock_qdrant_store.ChunkPoint.call_args.kwargs
    assert point_kwargs["department"] == "CNTT"
    assert point_kwargs["access_level"] == 2


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_successful_embedding_dispatch_marks_draft_as_embedding(
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

    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as mock_get_object_bytes:
        mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")
        client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": "doc-embed-1",
                "department_id": "CNTT",
                "object_key": "docs/handbook.pdf",
                "strategy": "recursive",
            },
            headers=_TRUSTED_HEADERS,
        )
    job_before = client.get("/api/v1/ingestion/jobs/doc-embed-1", headers=_TRUSTED_HEADERS)
    assert job_before.status_code == 200

    payload = {**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-1"}
    response = client.post(
        "/api/v1/ingestion/embedding",
        json=payload,
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    job = client.get("/api/v1/ingestion/jobs/doc-embed-1", headers=_TRUSTED_HEADERS)
    assert job.status_code == 200
    body = job.json()
    assert body["current_step"] == "embedding"
    assert body["task_id"] == task_id
    # The job endpoint carries the live task state inline, so the client's
    # reconciliation sweep needs only this one authorized call.
    assert body["task_state"] == "SUCCESS"
    assert body["task_percent"] == 100


def _canned_stream(
    *frames: dict[str, Any],
) -> Callable[[], AsyncIterator[dict[str, Any]]]:
    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for frame in frames:
            yield frame

    return _stream


def test_events_ws_only_forwards_frames_for_the_callers_departments(client: TestClient) -> None:
    own = {"type": "completed", "document_id": "doc-a", "department_id": "CNTT", "state": "SUCCESS"}
    other = {
        "type": "completed",
        "document_id": "doc-b",
        "department_id": "KHOA_KT",
        "state": "SUCCESS",
    }

    with patch("app.api.v1.ingestion.ingestion_event_stream", _canned_stream(other, own, other)):
        with client.websocket_connect("/api/v1/ingestion/events", headers=_TRUSTED_HEADERS) as ws:
            received = ws.receive_json()

    assert received == own


def test_events_ws_rejects_a_connection_without_the_internal_secret() -> None:
    bare_client = TestClient(app)

    with pytest.raises(WebSocketDisconnect):
        with bare_client.websocket_connect(
            "/api/v1/ingestion/events", headers=_TRUSTED_HEADERS
        ) as ws:
            ws.receive_json()
