import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.fixtures.documents import make_pdf_bytes

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


def _chunk_a_document(client: TestClient, document_id: str, *, department_id: str = "CNTT") -> None:
    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as mock_get_object_bytes:
        mock_get_object_bytes.return_value = make_pdf_bytes(
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        )
        client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": document_id,
                "department_id": department_id,
                "object_key": "docs/handbook.pdf",
                "strategy": "recursive",
                "params": {"chunk_size": 40, "overlap": 0},
            },
            headers={
                "X-User-Department-Access": json.dumps(
                    [{"department_id": department_id, "access_level": 3}]
                ),
                "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
            },
        )


def test_list_chunks_returns_the_page_after_chunking(client: TestClient) -> None:
    _chunk_a_document(client, "doc-chunks-1")

    response = client.get("/api/v1/documents/doc-chunks-1/chunks", headers=_TRUSTED_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 1000
    data = body["data"]
    assert data["data"]
    assert data["total_items"] == len(data["data"])
    assert data["limit"] == 50
    assert data["page"] == 1
    assert data["total_pages"] == 1


def test_list_chunks_paginates_with_page_and_limit(client: TestClient) -> None:
    _chunk_a_document(client, "doc-chunks-page")

    first_page = client.get(
        "/api/v1/documents/doc-chunks-page/chunks",
        params={"limit": 1, "page": 1},
        headers=_TRUSTED_HEADERS,
    )
    second_page = client.get(
        "/api/v1/documents/doc-chunks-page/chunks",
        params={"limit": 1, "page": 2},
        headers=_TRUSTED_HEADERS,
    )

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    first_data = first_page.json()["data"]
    second_data = second_page.json()["data"]
    assert len(first_data["data"]) == 1
    assert first_data["total_items"] == second_data["total_items"]
    assert first_data["total_items"] > 1
    assert first_data["total_pages"] == first_data["total_items"]
    assert first_data["data"][0]["chunk_index"] != second_data["data"][0]["chunk_index"]


def test_list_chunks_returns_404_when_document_never_chunked(client: TestClient) -> None:
    response = client.get(
        "/api/v1/documents/no-such-document/chunks", headers=_TRUSTED_HEADERS
    )

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 4042


def test_list_chunks_returns_403_when_missing_document_permission(client: TestClient) -> None:
    _chunk_a_document(client, "doc-chunks-perm")
    headers = {**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}

    response = client.get("/api/v1/documents/doc-chunks-perm/chunks", headers=headers)

    assert response.status_code == 403


def test_list_chunks_returns_403_for_a_caller_not_in_the_documents_department(
    client: TestClient,
) -> None:
    _chunk_a_document(client, "doc-chunks-dept", department_id="CNTT")
    headers = {
        "X-User-Department-Access": json.dumps(
            [{"department_id": "KHOA_KINH_TE", "access_level": 5}]
        ),
        "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
    }

    response = client.get("/api/v1/documents/doc-chunks-dept/chunks", headers=headers)

    assert response.status_code == 403
