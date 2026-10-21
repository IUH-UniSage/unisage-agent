"""The re-index worker against an in-memory Qdrant and a fake backend-java / embedding API."""

from collections.abc import Callable
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.integrations.backend_java_client import BackendJavaHTTPError
from app.rag.vectorstore import qdrant_store
from app.worker import embedding_reindex
from app.worker.embedding_reindex import embedding_texts, run_embedding_reindex_once

SOURCE = "unisage_chunks"
TARGET = "unisage_chunks__20261017030000"
OLD_DIMENSION = 4
NEW_DIMENSION = 3


class FakeBackend:
    def __init__(self, *, claims: list[dict[str, Any]] | None = None) -> None:
        self.claims = claims if claims is not None else [_claim()]
        self.progress: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.identities: dict[str, dict[str, Any]] = {}
        self.progress_status = "RUNNING"
        self.result_status = "SUCCEEDED"

    async def claim_embedding_reindex(self, *, limit: int = 1) -> list[dict[str, Any]]:
        claims, self.claims = self.claims, []
        return claims

    async def post_embedding_reindex_progress(self, **kwargs: Any) -> str:
        self.progress.append(kwargs)
        return self.progress_status

    async def post_embedding_reindex_result(self, **kwargs: Any) -> str:
        self.results.append(kwargs)
        return self.result_status if kwargs["succeeded"] else "FAILED"

    async def put_embedding_index_identity(
        self, *, collection: str, **kwargs: Any
    ) -> dict[str, Any]:
        if collection in self.identities:
            raise BackendJavaHTTPError("PUT", "/identity", 409, {})
        self.identities[collection] = {
            "provider": kwargs["provider"],
            "modelName": kwargs["model_name"],
            "dimension": kwargs["dimension"],
            "establishedBy": kwargs["established_by"],
        }
        return self.identities[collection]

    async def get_embedding_index_identity(self, *, collection: str) -> dict[str, Any] | None:
        return self.identities.get(collection)


def _claim() -> dict[str, Any]:
    return {
        "runId": "run-1",
        "leaseToken": "lease-1",
        "sourceCollection": SOURCE,
        "targetCollection": TARGET,
        "credentialRevision": 2,
        "dimension": NEW_DIMENSION,
        "fingerprint": [0.1] * (NEW_DIMENSION * 3),
        "credential": {
            "chatModelId": "model-new",
            "provider": "openai",
            "modelName": "text-embedding-new",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "apiKey": "sk-test-not-real",
            "sourceType": "CLOUD_API",
        },
    }


class FakeEmbedder:
    def __init__(self, *, fail: bool = False, on_call: Callable[[], None] | None = None) -> None:
        self.fail = fail
        self.on_call = on_call
        self.calls: list[list[str]] = []

    async def embed_tracked(self, texts: list[str], *_: Any) -> list[list[float]]:
        self.calls.append(texts)
        if self.on_call is not None:
            self.on_call()
        if self.fail:
            raise RuntimeError("provider down")
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class _Recorder:
    def __init__(self, **_: Any) -> None:
        self.request_id = "req"

    async def close(self, **_: Any) -> None:
        return None


class _Budget:
    def __init__(self, **_: Any) -> None:
        pass

    async def reserve_request(self, **_: Any) -> str:
        return "OK"


class _Redis:
    async def aclose(self) -> None:
        return None


pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    qdrant_store.ensure_collection(client, SOURCE, OLD_DIMENSION)
    qdrant_store.upsert_points(client, SOURCE, [_point(index) for index in range(5)])
    return client


def _point(index: int, document_id: str = "doc-1") -> models.PointStruct:
    chunk_id = f"{document_id}:{index}"
    return models.PointStruct(
        id=qdrant_store.chunk_point_id(chunk_id),
        vector={name: [0.5] * OLD_DIMENSION for name in qdrant_store._VECTOR_NAMES},
        payload={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "content": f"noi dung {index}",
            "summary": f"tom tat {index}" if index else "",
            "questions": [f"cau hoi {index}?"],
            "department": "CNTT",
            "access_level": 2,
            "is_public": False,
            "embedding_identity_key": "old-key",
        },
    )


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(embedding_reindex, "build_embedder_for", lambda credential, key: fake)
    monkeypatch.setattr(embedding_reindex, "UsageRecorder", _Recorder)
    monkeypatch.setattr(embedding_reindex, "BudgetTracker", _Budget)
    monkeypatch.setattr(
        "app.worker.embedding_reindex.redis_asyncio.Redis.from_url", lambda url: _Redis()
    )
    monkeypatch.setattr(settings, "EMBEDDING_REINDEX_BATCH_SIZE", 2)
    monkeypatch.setattr(settings, "EMBEDDING_REINDEX_MAX_ATTEMPTS", 1)
    return fake


@pytest.mark.asyncio
async def test_no_queued_run_does_nothing(qdrant: QdrantClient, embedder: FakeEmbedder) -> None:
    backend = FakeBackend(claims=[])

    assert await run_embedding_reindex_once(backend=backend, qdrant=qdrant) is False  # type: ignore[arg-type]
    assert not embedder.calls


@pytest.mark.asyncio
async def test_every_point_is_re_embedded_into_the_target_collection(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend()

    assert await run_embedding_reindex_once(backend=backend, qdrant=qdrant) is True  # type: ignore[arg-type]

    assert qdrant_store.get_collection_dimension(qdrant, TARGET) == NEW_DIMENSION
    assert qdrant_store.count_points(qdrant, TARGET) == 5
    assert qdrant_store.point_ids(qdrant, TARGET) == qdrant_store.point_ids(qdrant, SOURCE)
    record = qdrant_store.retrieve_points(qdrant, TARGET, [qdrant_store.chunk_point_id("doc-1:3")])[
        0
    ]
    assert record.payload is not None
    assert record.payload["content"] == "noi dung 3"
    assert record.payload["department"] == "CNTT"
    assert record.payload["embedding_identity_key"] not in (None, "old-key")
    # The source collection is left as it was: it stays live until Java cuts over.
    assert qdrant_store.count_points(qdrant, SOURCE) == 5
    assert backend.identities[TARGET]["establishedBy"] == "reindex"
    assert backend.identities[TARGET]["dimension"] == NEW_DIMENSION
    assert backend.results == [
        {
            "run_id": "run-1",
            "lease_token": "lease-1",
            "succeeded": True,
            "total_points": 5,
            "processed_points": 5,
            "failed_points": 0,
        }
    ]
    assert backend.progress[-1]["processed_points"] == 5
    # 3 batches of at most 2 points, 3 texts per point.
    assert [len(call) for call in embedder.calls] == [6, 6, 3]


def test_embedding_texts_fall_back_to_the_content_like_ingestion() -> None:
    assert embedding_texts({"content": "a", "summary": "", "questions": []}) == ["a", "a", "a"]
    assert embedding_texts({"content": "a", "summary": "b", "questions": ["c", "d"]}) == [
        "a",
        "b",
        "c d",
    ]


@pytest.mark.asyncio
async def test_a_cancelled_run_drops_the_target_and_reports_nothing(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend()
    backend.progress_status = "CANCELLED"

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert not qdrant.collection_exists(TARGET)
    assert backend.results == []


@pytest.mark.asyncio
async def test_an_embedding_failure_fails_the_run_and_drops_the_target(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    embedder.fail = True
    backend = FakeBackend()

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert not qdrant.collection_exists(TARGET)
    assert len(backend.results) == 1
    assert backend.results[0]["succeeded"] is False
    assert "provider down" in backend.results[0]["error_message"]
    assert "sk-test-not-real" not in backend.results[0]["error_message"]


@pytest.mark.asyncio
async def test_points_added_to_the_source_meanwhile_are_reconciled(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    added: list[bool] = []

    def embed_job_writes_to_the_source() -> None:
        if not added:
            added.append(True)
            qdrant_store.upsert_points(qdrant, SOURCE, [_point(0, document_id="doc-0")])

    embedder.on_call = embed_job_writes_to_the_source
    backend = FakeBackend()

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert qdrant_store.point_ids(qdrant, TARGET) == qdrant_store.point_ids(qdrant, SOURCE)
    assert qdrant_store.count_points(qdrant, TARGET) == 6
    assert backend.results[0]["succeeded"] is True


@pytest.mark.asyncio
async def test_a_refused_cutover_drops_the_unused_target(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend()
    backend.result_status = "FAILED"

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert not qdrant.collection_exists(TARGET)


@pytest.mark.asyncio
async def test_a_retried_run_reuses_its_target_and_identity(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend()
    backend.identities[TARGET] = {
        "provider": "openai",
        "modelName": "text-embedding-new",
        "dimension": NEW_DIMENSION,
        "establishedBy": "reindex",
    }

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert backend.results[0]["succeeded"] is True
    assert qdrant_store.count_points(qdrant, TARGET) == 5


@pytest.mark.asyncio
async def test_a_target_identity_of_another_model_fails_the_run(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend()
    backend.identities[TARGET] = {"provider": "google", "modelName": "x", "dimension": 768}

    await run_embedding_reindex_once(backend=backend, qdrant=qdrant)  # type: ignore[arg-type]

    assert backend.results[0]["succeeded"] is False
    assert not embedder.calls


def _rebuild_claim() -> dict[str, Any]:
    return {
        **_claim(),
        "rebuild": True,
        "dimension": OLD_DIMENSION,
        "fingerprint": [0.1] * (OLD_DIMENSION * 3),
    }


@pytest.mark.asyncio
async def test_a_rebuild_copies_the_dense_vectors_and_adds_bm25_without_embedding(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    """Production path of SPEC-hybrid-retrieval.md: the live model's own collection is rebuilt
    with the sparse vector - no embedding call, same dimension, same identity, pointer moves."""

    from app.rag.retrieval.lexical import BM25_VECTOR

    backend = FakeBackend(claims=[_rebuild_claim()])

    assert await run_embedding_reindex_once(backend=backend, qdrant=qdrant) is True  # type: ignore[arg-type]

    assert not embedder.calls
    assert qdrant_store.get_collection_dimension(qdrant, TARGET) == OLD_DIMENSION
    assert qdrant_store.count_points(qdrant, TARGET) == 5
    assert qdrant_store.point_ids(qdrant, TARGET) == qdrant_store.point_ids(qdrant, SOURCE)
    (record,) = qdrant.retrieve(
        collection_name=TARGET,
        ids=[qdrant_store.chunk_point_id("doc-1:3")],
        with_vectors=True,
        with_payload=True,
    )
    assert isinstance(record.vector, dict)
    assert set(record.vector) == {*qdrant_store._VECTOR_NAMES, BM25_VECTOR}
    (source,) = qdrant.retrieve(
        collection_name=SOURCE, ids=[record.id], with_vectors=True, with_payload=False
    )
    assert isinstance(source.vector, dict)
    assert record.vector["content_vector"] == pytest.approx(source.vector["content_vector"])
    assert len(record.vector[BM25_VECTOR].indices) > 0  # type: ignore[union-attr]
    assert record.payload is not None and record.payload["content"] == "noi dung 3"
    assert backend.identities[TARGET]["dimension"] == OLD_DIMENSION
    assert backend.results == [
        {
            "run_id": "run-1",
            "lease_token": "lease-1",
            "succeeded": True,
            "total_points": 5,
            "processed_points": 5,
            "failed_points": 0,
        }
    ]


@pytest.mark.asyncio
async def test_a_rebuild_reconciles_points_added_meanwhile(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    backend = FakeBackend(claims=[_rebuild_claim()])
    original = qdrant_store.point_ids
    added = False

    def point_ids_adding_one(
        client: QdrantClient, collection: str, page_size: int = 1000
    ) -> set[str]:
        nonlocal added
        if collection == SOURCE and not added:
            added = True
            qdrant_store.upsert_points(client, SOURCE, [_point(9, "doc-2")])
        return original(client, collection, page_size)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(qdrant_store, "point_ids", point_ids_adding_one)
        assert await run_embedding_reindex_once(backend=backend, qdrant=qdrant) is True  # type: ignore[arg-type]

    assert qdrant_store.point_ids(qdrant, TARGET) == qdrant_store.point_ids(qdrant, SOURCE)
    assert qdrant_store.count_points(qdrant, TARGET) == 6
    assert not embedder.calls


@pytest.mark.asyncio
async def test_the_target_collection_carries_the_bm25_sparse_vector(
    qdrant: QdrantClient, embedder: FakeEmbedder
) -> None:
    """Hybrid retrieval (SPEC-hybrid-retrieval.md): a re-index writes the `bm25` vector of
    every point from its payload, so the new collection is searchable lexically at once."""

    from app.rag.retrieval.lexical import BM25_VECTOR

    assert await run_embedding_reindex_once(backend=FakeBackend(), qdrant=qdrant) is True  # type: ignore[arg-type]

    assert qdrant_store.has_sparse_vector(qdrant, TARGET) is True
    (record,) = qdrant.retrieve(
        collection_name=TARGET,
        ids=[qdrant_store.chunk_point_id("doc-1:3")],
        with_vectors=[BM25_VECTOR],
        with_payload=False,
    )
    assert record.vector is not None and BM25_VECTOR in record.vector
    assert len(record.vector[BM25_VECTOR].indices) > 0
