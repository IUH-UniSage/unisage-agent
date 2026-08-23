import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.rag.ingestion.minio_client import ObjectNotFoundException
from tests.fixtures.documents import make_pdf_bytes

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_raw_text_for_valid_object_key(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Hello from PDF fixture.")

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.pdf"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 200
    assert "Hello from PDF fixture." in response.json()["raw_text"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_404_when_object_missing(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.side_effect = ObjectNotFoundException("docs/missing.pdf")

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/missing.pdf"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 404


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_415_for_unsupported_filetype(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = b"whatever"

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.doc"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 415


def test_preview_returns_403_when_missing_document_permission(client: TestClient) -> None:
    headers = {**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.pdf"},
        headers=headers,
    )

    assert response.status_code == 403


def test_preview_returns_403_when_department_not_granted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "KHOA_KINH_TE", "object_key": "docs/handbook.pdf"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 403
