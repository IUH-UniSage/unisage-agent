"""Hybrid (dense + BM25) search and the sparse-vector plumbing of the Qdrant store
(docs/specs/SPEC-hybrid-retrieval.md). Fake clients only - the fusion is client-side."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, QueryResponse, ScoredPoint, SparseVector

from app.core.config import settings
from app.rag.retrieval.lexical import BM25_VECTOR, sparse_query_vector, sparse_vector
from app.rag.vectorstore import qdrant_store
from app.rag.vectorstore.qdrant_store import (
    ChunkPoint,
    ensure_collection,
    has_sparse_vector,
    point_vectors,
    search_chunks_batch,
    upsert_chunk,
)


@pytest.fixture(autouse=True)
def _fresh_sparse_cache() -> None:
    qdrant_store.reset_sparse_cache()


def _scored(point_id: str, score: float) -> ScoredPoint:
    return ScoredPoint(id=point_id, version=0, score=score, payload={"chunk_id": point_id})


# --------------------------------------------------------------------------- search


def test_hybrid_sends_a_bm25_request_per_query_with_the_same_filter() -> None:
    client = MagicMock()
    client.query_batch_points.return_value = [QueryResponse(points=[])] * 8
    access = Filter(must=[])

    results = search_chunks_batch(
        client,
        query_vectors=[[0.1], [0.2]],
        limits=[4, 4],
        query_filter=access,
        sparse_queries=[sparse_query_vector("học phần 2105618"), sparse_vector("học phí")],
        candidate_multiplier=2,
    )

    assert results == [[], []]
    requests = client.query_batch_points.call_args.kwargs["requests"]
    assert len(requests) == 8  # 3 dense + 1 sparse, twice
    assert [request.using for request in requests[:4]] == [
        "content_vector",
        "summary_vector",
        "questions_vector",
        BM25_VECTOR,
    ]
    assert isinstance(requests[3].query, SparseVector)
    assert all(request.filter is access for request in requests)
    assert all(request.limit == 8 for request in requests)  # limit x multiplier


def test_a_query_without_terms_sends_no_sparse_request() -> None:
    client = MagicMock()
    client.query_batch_points.return_value = [QueryResponse(points=[])] * 7

    search_chunks_batch(
        client,
        query_vectors=[[0.1], [0.2]],
        limits=[4, 4],
        sparse_queries=[sparse_query_vector("???"), sparse_vector("học phí")],
    )

    requests = client.query_batch_points.call_args.kwargs["requests"]
    assert len(requests) == 7
    assert [request.using for request in requests].count(BM25_VECTOR) == 1


def test_without_sparse_queries_the_search_is_the_dense_only_one() -> None:
    client = MagicMock()
    client.query_batch_points.return_value = [QueryResponse(points=[_scored("a", 0.9)])] * 3

    results = search_chunks_batch(client, query_vectors=[[0.1]], limits=[4])

    requests = client.query_batch_points.call_args.kwargs["requests"]
    assert len(requests) == 3 and all(request.limit == 4 for request in requests)
    assert [point.id for point in results[0]] == ["a"]


def test_fusion_orders_by_rrf_but_keeps_the_dense_cosine_as_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_LEXICAL_HIT_SCORE", 0.66)
    client = MagicMock()
    # dense lists: a > b in content, b > a in summary, questions has only c (0.95)
    client.query_batch_points.return_value = [
        QueryResponse(points=[_scored("a", 0.80), _scored("b", 0.70)]),
        QueryResponse(points=[_scored("b", 0.75), _scored("a", 0.60)]),
        QueryResponse(points=[_scored("c", 0.95)]),
        # bm25: the code matched only in "d" (not in any dense list) and in "b"
        QueryResponse(points=[_scored("d", 12.0), _scored("b", 3.0)]),
    ]

    (ranked,) = search_chunks_batch(
        client,
        query_vectors=[[0.1]],
        limits=[3],
        sparse_queries=[sparse_query_vector("mã 2105618")],
        rrf_k=60,
    )

    ids = [point.id for point in ranked]
    # Dense merged by best cosine: c (0.95), a (0.80), b (0.75); BM25: d, b. RRF over the
    # two lists: b (rank 3 + rank 2) wins, then c and d (rank 1 of one list each, c first
    # as it appeared first), a (rank 2 of one list) is cut by limit=3.
    assert ids == ["b", "c", "d"]
    scores = {point.id: point.score for point in ranked}
    assert scores["b"] == 0.75  # best dense cosine, not the BM25 or RRF number
    assert scores["c"] == 0.95
    assert scores["d"] == 0.66  # BM25-only hit: the lexical score


def test_a_bm25_only_hit_enters_the_result_with_the_lexical_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_LEXICAL_HIT_SCORE", 0.66)
    client = MagicMock()
    client.query_batch_points.return_value = [
        QueryResponse(points=[_scored("a", 0.80)]),
        QueryResponse(points=[]),
        QueryResponse(points=[]),
        QueryResponse(points=[_scored("code-chunk", 9.0), _scored("a", 1.0)]),
    ]

    (ranked,) = search_chunks_batch(
        client, query_vectors=[[0.1]], limits=[5], sparse_queries=[sparse_query_vector("2105618")]
    )

    by_id = {point.id: point for point in ranked}
    assert set(by_id) == {"a", "code-chunk"}
    assert by_id["code-chunk"].score == 0.66
    # The stand-in score is not a cosine: the hit says so in its payload.
    assert by_id["code-chunk"].payload == {"chunk_id": "code-chunk", "lexical_only": True}
    assert by_id["a"].score == 0.80
    assert "lexical_only" not in (by_id["a"].payload or {})


def test_limit_applies_after_fusion() -> None:
    client = MagicMock()
    client.query_batch_points.return_value = [
        QueryResponse(points=[_scored("a", 0.9), _scored("b", 0.8), _scored("c", 0.7)]),
        QueryResponse(points=[]),
        QueryResponse(points=[]),
        QueryResponse(points=[_scored("z", 5.0)]),
    ]

    (ranked,) = search_chunks_batch(
        client, query_vectors=[[0.1]], limits=[2], sparse_queries=[sparse_query_vector("x y")]
    )

    assert len(ranked) == 2


# --------------------------------------------------------------------------- schema


def test_ensure_collection_configures_the_bm25_sparse_vector() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_collection(client, "c1", 8)

    sparse = client.create_collection.call_args.kwargs["sparse_vectors_config"]
    assert set(sparse) == {BM25_VECTOR}
    assert has_sparse_vector(client, "c1") is True  # cached by ensure_collection
    client.get_collection.assert_not_called()


def test_has_sparse_vector_reads_the_collection_once() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.params.sparse_vectors = {BM25_VECTOR: object()}

    assert has_sparse_vector(client, "c1") is True
    assert has_sparse_vector(client, "c1") is True
    client.get_collection.assert_called_once()

    client.get_collection.return_value.config.params.sparse_vectors = None
    assert has_sparse_vector(client, "c2") is False
    client.collection_exists.return_value = False
    assert has_sparse_vector(client, "c3") is False


def test_copy_points_with_bm25_moves_dense_vectors_and_adds_the_sparse_one() -> None:
    """Qdrant 1.15 refuses to add a vector to an existing collection (verified 21/10/2026:
    `update_collection` -> "Not existing vector name"), so the points are copied into a
    collection created with `bm25`."""

    client = QdrantClient(":memory:")
    ensure_collection(client, "old", 4)
    dense = [0.1, 0.2, 0.3, 0.4]
    client.upsert(
        collection_name="old",
        points=[
            models.PointStruct(
                id=qdrant_store.chunk_point_id(f"d:{index}"),
                vector=dict.fromkeys(qdrant_store._VECTOR_NAMES, dense),
                payload={
                    "chunk_id": f"d:{index}",
                    "content": text,
                    "heading_path": ["Mục 2105618"],
                },
            )
            for index, text in enumerate(["Học phần 2105618", "Điều kiện tốt nghiệp"])
        ],
    )
    ensure_collection(client, "new", 4)
    seen: list[int] = []

    copied = qdrant_store.copy_points_with_bm25(client, "old", "new", batch=1, on_batch=seen.append)

    assert copied == 2 and seen == [1, 2]
    assert qdrant_store.count_points(client, "new") == 2
    (record,) = client.retrieve(
        collection_name="new",
        ids=[qdrant_store.chunk_point_id("d:1")],
        with_vectors=True,
        with_payload=True,
    )
    vectors = cast(dict[str, Any], record.vector)
    assert set(vectors) == {*qdrant_store._VECTOR_NAMES, BM25_VECTOR}
    # Qdrant stores cosine vectors normalized: same direction as what went in.
    norm = sum(x * x for x in dense) ** 0.5
    assert vectors["content_vector"] == pytest.approx([x / norm for x in dense], rel=1e-5)
    assert (record.payload or {})["content"] == "Điều kiện tốt nghiệp"
    # The heading path's code is searchable: it went into bm25.
    (ranked,) = search_chunks_batch(
        client,
        query_vectors=[dense],
        limits=[1],
        collection="new",
        sparse_queries=[sparse_query_vector("2105618")],
    )
    assert ranked


def test_point_vectors_and_upsert_include_bm25_when_present() -> None:
    dense = [0.1] * 4
    assert BM25_VECTOR not in point_vectors(dense, dense, dense)
    assert BM25_VECTOR not in point_vectors(dense, dense, dense, bm25=([], []))
    vectors = point_vectors(dense, dense, dense, bm25=sparse_vector("2105618"))
    assert isinstance(vectors[BM25_VECTOR], SparseVector)

    client = MagicMock()
    upsert_chunk(
        client,
        ChunkPoint(
            point_id="p1",
            document_id="d",
            object_key="k",
            chunk_id="d:0",
            content="Học phần 2105618",
            summary="",
            questions=[],
            department="CNTT",
            access_level=1,
            category="HOC_VU",
            region_type="text",
            content_vector=dense,
            summary_vector=dense,
            questions_vector=dense,
            bm25=sparse_vector("Học phần 2105618"),
        ),
        "c1",
    )
    point = client.upsert.call_args.kwargs["points"][0]
    assert BM25_VECTOR in point.vector


def test_in_memory_qdrant_accepts_sparse_points_and_hybrid_queries() -> None:
    """End to end on the embedded engine: a point with `bm25`, a sparse query finds it."""

    client = QdrantClient(":memory:")
    ensure_collection(client, "mem", 4)
    dense = [0.1, 0.2, 0.3, 0.4]
    for index, text in enumerate(["Học phần 2105618 có 3 tín chỉ", "Điều kiện tốt nghiệp"]):
        upsert_chunk(
            client,
            ChunkPoint(
                point_id=qdrant_store.chunk_point_id(f"d:{index}"),
                document_id="d",
                object_key="k",
                chunk_id=f"d:{index}",
                content=text,
                summary="",
                questions=[],
                department="CNTT",
                access_level=1,
                category="HOC_VU",
                region_type="text",
                content_vector=dense,
                summary_vector=dense,
                questions_vector=dense,
                bm25=sparse_vector(text),
            ),
            "mem",
        )
    assert has_sparse_vector(client, "mem") is True

    (ranked,) = search_chunks_batch(
        client,
        query_vectors=[dense],
        limits=[2],
        collection="mem",
        sparse_queries=[sparse_query_vector("2105618")],
    )
    assert (ranked[0].payload or {})["chunk_id"] == "d:0"
