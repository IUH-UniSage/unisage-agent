import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.documents import make_pdf_bytes, make_xlsx_bytes

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


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
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": object_key,
            "strategy": strategy_name,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["data"]["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_semantic_strategy_without_live_openai_calls(
    mock_get_object_bytes: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    async def _fake_embed_tracked(
        self: OpenAIEmbedder, texts: list[str], usage_recorder: object, budget_tracker: object
    ) -> list[list[float]]:
        del usage_recorder, budget_tracker
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(OpenAIEmbedder, "embed_tracked", _fake_embed_tracked)
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "semantic",
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["data"]["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_semantic_strategy_reports_embedding_identity_mismatch_clearly(
    mock_get_object_bytes: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `EmbeddingIdentityMismatchError` raised deep inside the semantic strategy's embedding
    step must reach the client as a specific, actionable error (`app.main`'s dedicated handler),
    not `unhandled_exception_handler`'s generic "Có lỗi xảy ra, bạn thử lại sau nhé." 500."""

    from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    async def _fake_embed_tracked(
        self: OpenAIEmbedder, texts: list[str], usage_recorder: object, budget_tracker: object
    ) -> list[list[float]]:
        del self, texts, usage_recorder, budget_tracker
        raise EmbeddingIdentityMismatchError("collection already has vectors but no identity")

    monkeypatch.setattr(OpenAIEmbedder, "embed_tracked", _fake_embed_tracked)
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "semantic",
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["code"] == 4015


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_semantic_strategy_reports_other_embedding_provider_errors_clearly(
    mock_get_object_bytes: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any OTHER `EmbeddingProviderError` (auth/network/budget-rejection) must also reach the
    client as a specific error, not a generic 500 - `EmbeddingIdentityMismatchError`'s more
    specific handler must not swallow this broader case."""

    from app.core.errors.llm_error_classifier import EmbeddingProviderError
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    async def _fake_embed_tracked(
        self: OpenAIEmbedder, texts: list[str], usage_recorder: object, budget_tracker: object
    ) -> list[list[float]]:
        del self, texts, usage_recorder, budget_tracker
        raise EmbeddingProviderError("provider auth failed")

    monkeypatch.setattr(OpenAIEmbedder, "embed_tracked", _fake_embed_tracked)
    mock_get_object_bytes.return_value = make_pdf_bytes("A paragraph for chunking.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "semantic",
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 502
    assert response.json()["code"] == 5006


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_excel_row_on_non_xlsx_returns_422(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Not a spreadsheet.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "excel_row",
        },
        headers=_TRUSTED_HEADERS,
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
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
        },
        headers=_TRUSTED_HEADERS,
    )
    assert response.status_code == 200

    job = client.get("/api/v1/ingestion/jobs/doc-draft-1", headers=_TRUSTED_HEADERS)
    assert job.status_code == 200
    assert job.json()["data"]["chunking_strategy"] == "recursive"
    assert job.json()["data"]["chunks"] == response.json()["data"]["chunks"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_failed_chunking_call_does_not_write_a_draft_row(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Not a spreadsheet.")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-draft-failed",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "excel_row",
        },
        headers=_TRUSTED_HEADERS,
    )
    assert response.status_code == 422

    job = client.get("/api/v1/ingestion/jobs/doc-draft-failed", headers=_TRUSTED_HEADERS)
    assert job.status_code == 404


def test_chunking_returns_403_when_missing_document_permission(client: TestClient) -> None:
    headers = {**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
        },
        headers=headers,
    )

    assert response.status_code == 403


def test_chunking_returns_403_when_department_not_granted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "KHOA_KINH_TE",
            "object_key": "docs/handbook.pdf",
            "strategy": "recursive",
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 403


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_chunking_scanned_pdf_without_text_returns_a_clear_error(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    # A scanned PDF has pages but no text layer; a blank page stands in for it.
    mock_get_object_bytes.return_value = make_pdf_bytes("")

    response = client.post(
        "/api/v1/ingestion/chunking",
        json={
            "document_id": "doc-1",
            "department_id": "CNTT",
            "object_key": "docs/scan.pdf",
            "strategy": "recursive",
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["code"] == 4221
