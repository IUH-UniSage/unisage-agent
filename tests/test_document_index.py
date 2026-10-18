"""Backend-only document index sync routes (changes/17-10-2026-Embedding-Reindex R2): an access
edit rewrites the payload, a new file or a delete drops the points - in the live collection and
in an open re-index's target alike."""

import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from app.api.v1 import document_index
from app.core.registry import model_registry
from app.main import app
from app.rag.vectorstore import qdrant_store
from tests.fixtures.documents import make_pdf_bytes

LIVE = "unisage_chunks"
TARGET = "unisage_chunks__20261017030000"

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    for collection in (LIVE, TARGET):
        qdrant_store.ensure_collection(client, collection, 2)
        qdrant_store.upsert_points(
            client,
            collection,
            [_point("doc-1", index) for index in range(3)] + [_point("doc-2", 0)],
        )
    app.dependency_overrides[document_index.get_qdrant_client] = lambda: client
    yield client
    app.dependency_overrides.pop(document_index.get_qdrant_client, None)


@pytest.fixture(autouse=True)
def _reindex_open() -> Iterator[None]:
    previous = model_registry._current_snapshot
    model_registry._current_snapshot = model_registry.parse_snapshot(
        {
            "version": 1,
            "generatedAt": "2026-10-17T03:00:00Z",
            "purposes": {},
            "embeddingIndexIdentity": None,
            "embeddingCollection": LIVE,
            "embeddingReindex": {"runId": "r", "status": "RUNNING", "targetCollection": TARGET},
        }
    )
    yield
    model_registry._current_snapshot = previous


def _point(document_id: str, index: int) -> models.PointStruct:
    chunk_id = f"{document_id}:{index}"
    return models.PointStruct(
        id=qdrant_store.chunk_point_id(chunk_id),
        vector={name: [0.1, 0.2] for name in qdrant_store._VECTOR_NAMES},
        payload={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "content": "x",
            "department": "CNTT",
            "access_level": 2,
            "is_public": False,
        },
    )


def _payloads(client: QdrantClient, collection: str, document_id: str) -> list[dict[str, object]]:
    return [
        record.payload or {}
        for record in qdrant_store.scroll_chunks_by_document(client, document_id, collection)
    ]


def test_access_edit_rewrites_every_point_in_both_collections(
    client: TestClient, qdrant: QdrantClient
) -> None:
    response = client.patch(
        "/api/v1/internal/documents/doc-1/index-metadata",
        json={"department_id": "KINH_TE", "access_level": 0, "is_public": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"] == {"updated_points": 6}
    for collection in (LIVE, TARGET):
        payloads = _payloads(qdrant, collection, "doc-1")
        assert len(payloads) == 3
        assert all(p["department"] == "KINH_TE" for p in payloads)
        assert all(p["access_level"] == 0 and p["is_public"] is True for p in payloads)
        assert _payloads(qdrant, collection, "doc-2")[0]["department"] == "CNTT"


def test_access_edit_moves_the_draft_to_the_new_department(
    client: TestClient, qdrant: QdrantClient
) -> None:
    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as get_object_bytes:
        get_object_bytes.return_value = make_pdf_bytes("Mot doan van ban.")
        chunked = client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": "doc-1",
                "department_id": "CNTT",
                "object_key": "docs/a.pdf",
                "strategy": "recursive",
            },
            headers={
                "X-User-Department-Access": json.dumps([{"department_id": "*", "access_level": 9}]),
                "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
            },
        )
    assert chunked.status_code == 200, chunked.text

    client.patch(
        "/api/v1/internal/documents/doc-1/index-metadata",
        json={"department_id": "KINH_TE", "access_level": 1, "is_public": False},
    )

    job = client.get(
        "/api/v1/ingestion/jobs/doc-1",
        headers={
            "X-User-Department-Access": json.dumps(
                [{"department_id": "KINH_TE", "access_level": 1}]
            ),
            "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
        },
    )
    assert job.status_code == 200, job.text


def test_delete_drops_the_points_and_the_draft(client: TestClient, qdrant: QdrantClient) -> None:
    response = client.delete("/api/v1/internal/documents/doc-1/index")

    assert response.status_code == 200, response.text
    assert response.json()["data"] == {"deleted_points": 6}
    for collection in (LIVE, TARGET):
        assert _payloads(qdrant, collection, "doc-1") == []
        assert len(_payloads(qdrant, collection, "doc-2")) == 1


def test_routes_need_the_internal_secret(qdrant: QdrantClient) -> None:
    with TestClient(app) as anonymous:
        response = anonymous.delete("/api/v1/internal/documents/doc-1/index")

    assert response.status_code == 403
    assert len(_payloads(qdrant, LIVE, "doc-1")) == 3


def test_delete_document_points_keeps_the_chunks_still_in_the_set(qdrant: QdrantClient) -> None:
    dropped = qdrant_store.delete_document_points(
        qdrant, "doc-1", collection=LIVE, keep_chunk_ids={"doc-1:0", "doc-1:1"}
    )

    assert dropped == 1
    assert sorted(str(p["chunk_id"]) for p in _payloads(qdrant, LIVE, "doc-1")) == [
        "doc-1:0",
        "doc-1:1",
    ]
