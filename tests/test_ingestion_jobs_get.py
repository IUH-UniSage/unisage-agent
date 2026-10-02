import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.documents import make_pdf_bytes

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


def _chunk_a_document(client: TestClient, document_id: str, *, department_id: str = "CNTT") -> None:
    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as mock_get_object_bytes:
        mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")
        client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": document_id,
                "department_id": department_id,
                "object_key": "docs/handbook.pdf",
                "strategy": "recursive",
                "params": {"chunk_size": 800, "overlap": 120},
            },
            headers={
                "X-User-Department-Access": json.dumps(
                    [{"department_id": department_id, "access_level": 3}]
                ),
                "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
            },
        )


def test_get_job_returns_the_draft_after_chunking(client: TestClient) -> None:
    _chunk_a_document(client, "doc-jobs-1")

    response = client.get("/api/v1/ingestion/jobs/doc-jobs-1", headers=_TRUSTED_HEADERS)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["object_key"] == "docs/handbook.pdf"
    assert body["current_step"] == "chunked"
    assert body["chunking_strategy"] == "recursive"
    assert body["chunking_params"] == {"chunk_size": 800, "overlap": 120}
    assert body["chunks"]
    assert body["task_state"] is None


def test_get_job_returns_404_when_no_draft_exists(client: TestClient) -> None:
    response = client.get("/api/v1/ingestion/jobs/no-such-document", headers=_TRUSTED_HEADERS)

    assert response.status_code == 404


def test_get_job_returns_403_without_document_permission(client: TestClient) -> None:
    _chunk_a_document(client, "doc-jobs-perm")

    response = client.get(
        "/api/v1/ingestion/jobs/doc-jobs-perm",
        headers={**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])},
    )

    assert response.status_code == 403


def test_get_job_returns_403_for_a_caller_not_in_the_drafts_department(client: TestClient) -> None:
    _chunk_a_document(client, "doc-jobs-dept", department_id="CNTT")

    response = client.get(
        "/api/v1/ingestion/jobs/doc-jobs-dept",
        headers={
            "X-User-Department-Access": json.dumps(
                [{"department_id": "KHOA_KINH_TE", "access_level": 5}]
            ),
            "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
        },
    )

    assert response.status_code == 403


def test_rechunking_replaces_rather_than_duplicates_the_draft(client: TestClient) -> None:
    _chunk_a_document(client, "doc-jobs-2")
    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as mock_get_object_bytes:
        mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")
        client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": "doc-jobs-2",
                "department_id": "CNTT",
                "object_key": "docs/handbook.pdf",
                "strategy": "token_based",
            },
            headers=_TRUSTED_HEADERS,
        )

    response = client.get("/api/v1/ingestion/jobs/doc-jobs-2", headers=_TRUSTED_HEADERS)

    assert response.status_code == 200
    assert response.json()["data"]["chunking_strategy"] == "token_based"


def test_failed_task_progress_shows_the_jobs_specific_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`embed_chunks` ends with `IngestionJobFailedError(message, code)` - polling must show
    that reason, not the generic EMBEDDING_JOB_FAILED sentence."""

    from app.api.v1 import ingestion
    from app.worker.embedding_job_errors import IngestionJobFailedError

    class _FakeResult:
        state = "FAILURE"
        info = IngestionJobFailedError("Mô hình Extraction: API key không hợp lệ (HTTP 401).", 5008)

    monkeypatch.setattr(ingestion, "AsyncResult", lambda *_a, **_k: _FakeResult())

    progress = ingestion._read_task_progress("task-1")

    assert progress.state == "FAILURE"
    assert progress.error_code == 5008
    assert progress.message == "Mô hình Extraction: API key không hợp lệ (HTTP 401)."
