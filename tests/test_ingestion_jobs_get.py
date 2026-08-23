from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from tests.fixtures.documents import make_pdf_bytes


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_get_job_returns_the_draft_after_chunking(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-jobs-1",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
            "params": {"chunk_size": 800, "overlap": 120},
        },
    )

    response = client.get("/api/v1/ingestion/jobs/doc-jobs-1")

    assert response.status_code == 200
    body = response.json()
    assert body["object_key"] == "docs/handbook.pdf"
    assert body["current_step"] == "chunked"
    assert body["chunking_strategy"] == "recursive"
    assert body["chunking_params"] == {"chunk_size": 800, "overlap": 120}
    assert body["chunks"]


def test_get_job_returns_404_when_no_draft_exists(client: TestClient) -> None:
    response = client.get("/api/v1/ingestion/jobs/no-such-document")

    assert response.status_code == 404


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_rechunking_replaces_rather_than_duplicates_the_draft(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-jobs-2",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
        },
    )
    client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-jobs-2",
            "object_key": "docs/handbook.pdf",
            "strategy": "token_based",
        },
    )

    response = client.get("/api/v1/ingestion/jobs/doc-jobs-2")

    assert response.status_code == 200
    assert response.json()["chunking_strategy"] == "token_based"
