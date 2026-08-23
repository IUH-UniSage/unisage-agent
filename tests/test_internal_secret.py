import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings

_PREVIEW_PAYLOAD = {"department_id": "CNTT", "object_key": "docs/handbook.pdf"}
_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


def test_request_without_secret_header_is_rejected(client: TestClient) -> None:
    client.headers.pop("X-Internal-Secret", None)

    response = client.post("/api/v1/ingestion/preview", json=_PREVIEW_PAYLOAD)

    assert response.status_code == 403


def test_request_with_wrong_secret_is_rejected(client: TestClient) -> None:
    client.headers["X-Internal-Secret"] = "not-the-right-secret"

    response = client.post("/api/v1/ingestion/preview", json=_PREVIEW_PAYLOAD)

    assert response.status_code == 403


@patch("app.api.v1.ingestion.minio_client.get_object_bytes")
def test_request_with_correct_secret_reaches_the_route(
    mock_get_object_bytes: MagicMock, client: TestClient
) -> None:
    mock_get_object_bytes.return_value = b"Hello world."
    client.headers["X-Internal-Secret"] = settings.INTERNAL_SECRET_KEY

    response = client.post(
        "/api/v1/ingestion/preview",
        json={"department_id": "CNTT", "object_key": "docs/handbook.txt"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 200
    mock_get_object_bytes.assert_called_once()


def test_health_check_does_not_require_internal_secret(client: TestClient) -> None:
    client.headers.pop("X-Internal-Secret", None)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
