from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.rag.ingestion.minio_client import ObjectNotFoundException
from tests.fixtures.documents import make_pdf_bytes


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_raw_text_for_valid_object_key(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = make_pdf_bytes("Hello from PDF fixture.")

    response = client.post("/api/v1/ingestion/preview", json={"object_key": "docs/handbook.pdf"})

    assert response.status_code == 200
    assert "Hello from PDF fixture." in response.json()["raw_text"]


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_404_when_object_missing(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.side_effect = ObjectNotFoundException("docs/missing.pdf")

    response = client.post("/api/v1/ingestion/preview", json={"object_key": "docs/missing.pdf"})

    assert response.status_code == 404


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_preview_returns_415_for_unsupported_filetype(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = b"whatever"

    response = client.post("/api/v1/ingestion/preview", json={"object_key": "docs/handbook.doc"})

    assert response.status_code == 415
