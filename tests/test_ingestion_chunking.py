from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.documents import make_pdf_bytes, make_xlsx_bytes


@pytest.mark.parametrize(
    "strategy_name,object_key,content",
    [
        ("recursive", "docs/handbook.pdf", make_pdf_bytes("A paragraph for chunking.")),
        ("token_based", "docs/handbook.pdf", make_pdf_bytes("A paragraph for chunking.")),
        ("markdown_aware", "docs/handbook.pdf", make_pdf_bytes("A paragraph for chunking.")),
        (
            "excel_row",
            "docs/roster.xlsx",
            make_xlsx_bytes(header=["Name"], rows=[["Alice"], ["Bob"]]),
        ),
    ],
)
@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_returns_non_empty_chunk_list_for_each_strategy(
    mock_get_object_bytes: MagicMock,
    client: TestClient,
    strategy_name: str,
    object_key: str,
    content: bytes,
) -> None:
    mock_get_object_bytes.return_value = content

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={"document_id": "doc-1", "object_key": object_key, "strategy": strategy_name},
    )

    assert response.status_code == 200
    assert response.json()["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_semantic_strategy_without_live_openai_calls(
    mock_get_object_bytes: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    def _fake_embed(self: OpenAIEmbedder, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(OpenAIEmbedder, "embed", _fake_embed)
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={"document_id": "doc-1", "object_key": "docs/handbook.pdf", "strategy": "semantic"},
    )

    assert response.status_code == 200
    assert response.json()["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_excel_row_on_non_xlsx_returns_422(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Not a spreadsheet.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={"document_id": "doc-1", "object_key": "docs/handbook.pdf", "strategy": "excel_row"},
    )

    assert response.status_code == 422


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_successful_chunking_persists_a_resumable_draft(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-draft-1",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
        },
    )
    assert response.status_code == 200

    job = client.get("/api/v1/ingestion/jobs/doc-draft-1")
    assert job.status_code == 200
    assert job.json()["chunking_strategy"] == "recursive"
    assert job.json()["chunks"] == response.json()["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_failed_chunking_call_does_not_write_a_draft_row(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Not a spreadsheet.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-draft-failed",
            "object_key": "docs/handbook.pdf",
            "strategy": "excel_row",
        },
    )
    assert response.status_code == 422

    job = client.get("/api/v1/ingestion/jobs/doc-draft-failed")
    assert job.status_code == 404
