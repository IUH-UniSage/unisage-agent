from unittest.mock import MagicMock

from app.core.config import settings
from app.rag.vectorstore.qdrant_store import ChunkPoint, ensure_collection, upsert_chunk


def test_ensure_collection_creates_when_absent() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_collection(client)

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == settings.QDRANT_COLLECTION
    assert set(kwargs["vectors_config"].keys()) == {
        "content_vector",
        "summary_vector",
        "questions_vector",
    }


def test_ensure_collection_is_idempotent_when_already_present() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True

    ensure_collection(client)

    client.create_collection.assert_not_called()


def test_upsert_chunk_builds_expected_payload_and_vector_shape() -> None:
    client = MagicMock()
    point = ChunkPoint(
        point_id="doc-1:0",
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        chunk_id="doc-1:0",
        content="chunk text",
        summary="a summary",
        questions=["Q1?", "Q2?"],
        department="CNTT",
        access_level=2,
        region_type="text",
        content_vector=[0.1, 0.2],
        summary_vector=[0.3, 0.4],
        questions_vector=[0.5, 0.6],
    )

    upsert_chunk(client, point)

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == settings.QDRANT_COLLECTION
    (upserted_point,) = kwargs["points"]
    assert upserted_point.id == "doc-1:0"
    assert upserted_point.vector == {
        "content_vector": [0.1, 0.2],
        "summary_vector": [0.3, 0.4],
        "questions_vector": [0.5, 0.6],
    }
    assert upserted_point.payload == {
        "document_id": "doc-1",
        "object_key": "docs/handbook.pdf",
        "chunk_id": "doc-1:0",
        "content": "chunk text",
        "summary": "a summary",
        "questions": ["Q1?", "Q2?"],
        "department": "CNTT",
        "access_level": 2,
        "region_type": "text",
    }
