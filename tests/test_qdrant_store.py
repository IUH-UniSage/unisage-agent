from unittest.mock import MagicMock

from qdrant_client.http.models import QueryResponse, ScoredPoint

from app.core.config import settings
from app.rag.vectorstore.qdrant_store import (
    ChunkPoint,
    ensure_collection,
    search_chunks,
    upsert_chunk,
)


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
        category="HOC_VU",
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
        "category": "HOC_VU",
        "region_type": "text",
    }


def test_search_chunks_returns_empty_list_when_collection_missing() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert points == []
    client.query_points.assert_not_called()


def test_search_chunks_queries_all_three_named_vectors_with_payload_attached() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    scored_point = ScoredPoint(id="c1", version=0, score=0.9, payload={"chunk_id": "c1"})
    client.query_points.return_value = QueryResponse(points=[scored_point])

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert points == [scored_point]
    assert client.query_points.call_count == 3
    used_vectors = {call.kwargs["using"] for call in client.query_points.call_args_list}
    assert used_vectors == {"content_vector", "summary_vector", "questions_vector"}
    for call in client.query_points.call_args_list:
        assert call.kwargs["collection_name"] == settings.QDRANT_COLLECTION
        assert call.kwargs["query"] == [0.1, 0.2]
        assert call.kwargs["with_payload"] is True


def test_search_chunks_keeps_the_best_score_per_chunk_across_vectors() -> None:
    """A chunk that scores low on content_vector but high on questions_vector
    (its precomputed questions closely match the user's query) must surface
    with the HIGHER score, not be lost or duplicated."""

    client = MagicMock()
    client.collection_exists.return_value = True
    low_score = ScoredPoint(id="c1", version=0, score=0.4, payload={"chunk_id": "c1"})
    high_score = ScoredPoint(id="c1", version=0, score=0.85, payload={"chunk_id": "c1"})
    other_chunk = ScoredPoint(id="c2", version=0, score=0.5, payload={"chunk_id": "c2"})

    client.query_points.side_effect = [
        QueryResponse(points=[low_score]),  # content_vector
        QueryResponse(points=[other_chunk]),  # summary_vector
        QueryResponse(points=[high_score]),  # questions_vector
    ]

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert len(points) == 2
    assert points[0].id == "c1"
    assert points[0].score == 0.85  # kept the higher of the two c1 scores
    assert points[1].id == "c2"


def test_search_chunks_respects_limit_after_merging() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = [
        QueryResponse(
            points=[
                ScoredPoint(id=f"c{i}", version=0, score=1.0 - i * 0.1, payload={})
                for i in range(3)
            ]
        ),
        QueryResponse(points=[]),
        QueryResponse(points=[]),
    ]

    points = search_chunks(client, query_vector=[0.1], limit=2)

    assert len(points) == 2
    assert [p.id for p in points] == ["c0", "c1"]
