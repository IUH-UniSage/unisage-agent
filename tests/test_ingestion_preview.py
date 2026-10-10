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
    assert "Hello from PDF fixture." in response.json()["data"]["raw_text"]


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


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_422_for_non_utf8_text(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = "điểm".encode("utf-16")

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.txt"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["code"] == 4222


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_reports_storage_failure_specifically(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    from app.core.errors.exceptions import StorageUnavailableException

    mock_get_object_bytes.side_effect = StorageUnavailableException("S3 AccessDenied")

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.txt"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 502
    assert response.json()["code"] == 5018
    assert "AccessDenied" in response.json()["message"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_infrastructure_failures_name_the_component(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    """A backing service failing (here Qdrant-style, via its own exception type) must say
    which service, not a generic 500."""

    from qdrant_client.http.exceptions import ResponseHandlingException

    mock_get_object_bytes.side_effect = ResponseHandlingException(ConnectionError("down"))

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.txt"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 502
    assert response.json()["code"] == 5017
    assert response.json()["errors"] == {"component": "qdrant"}
