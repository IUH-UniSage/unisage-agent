"""Invalidation hooks of the retrieval cache (docs/specs/SPEC-retrieval-cache.md): an
embed, an access-payload edit, an index delete and a single-chunk delete each reach the
cache; nothing here needs a live Redis or Qdrant."""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import asyncio as redis_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations import retrieval_cache as rc
from app.integrations.retrieval_cache import RedisRetrievalCache
from app.rag.vectorstore import qdrant_store
from app.schemas.document_index import IndexMetadataUpdateRequest
from app.services import document_index_service
from app.services import retrieval_cache_invalidation as inv
from tests.fakes.fake_redis import FakeRedisStore


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRedisStore, RedisRetrievalCache]:
    store = FakeRedisStore()
    cache = RedisRetrievalCache(
        "redis://unused",
        client_factory=store.client,
        ttl_seconds=600,
        write_on_second_sighting=False,
        rng=random.Random(1),
    )
    monkeypatch.setattr(inv, "default_cache", lambda: cache)
    return store, cache


async def _seed(cache: RedisRetrievalCache, key: str, document_ids: list[str]) -> None:
    assert await cache.set("l2", key, "{}", description="d", document_ids=document_ids)


@pytest.mark.asyncio
async def test_invalidate_document_purges_only_entries_referencing_it(
    fake_cache: tuple[FakeRedisStore, RedisRetrievalCache],
) -> None:
    _store, cache = fake_cache
    await _seed(cache, rc.entry_key("l2", "a"), ["doc-1", "doc-2"])
    await _seed(cache, rc.entry_key("l2", "b"), ["doc-3"])

    assert await inv.invalidate_document("doc-2") == 1
    assert await cache.get("l2", rc.entry_key("l2", "a")) is None
    assert await cache.get("l2", rc.entry_key("l2", "b")) == "{}"
    assert await inv.invalidate_document("doc-404") == 0


@pytest.mark.asyncio
async def test_invalidate_after_embed_purges_and_bumps_the_documents_scope(
    fake_cache: tuple[FakeRedisStore, RedisRetrievalCache],
) -> None:
    _store, cache = fake_cache
    await _seed(cache, rc.entry_key("l2", "a"), ["doc-1"])

    await inv.invalidate_after_embed("doc-1", department_id="CNTT", is_public=False)
    assert await cache.get("l2", rc.entry_key("l2", "a")) is None
    assert await cache.read_generations(["public", "dept:CNTT", "dept:KT"]) == {
        "public": 0,
        "dept:CNTT": 1,
        "dept:KT": 0,
    }

    await inv.invalidate_after_embed("doc-2", department_id="CNTT", is_public=True)
    assert await cache.read_generations(["public", "dept:CNTT"]) == {"public": 1, "dept:CNTT": 1}


@pytest.mark.asyncio
async def test_hooks_are_no_ops_without_a_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inv, "default_cache", lambda: None)
    assert await inv.invalidate_document("doc-1") == 0
    await inv.invalidate_after_embed("doc-1", department_id="CNTT", is_public=False)


@pytest.mark.asyncio
async def test_hooks_fail_open(fake_cache: tuple[FakeRedisStore, RedisRetrievalCache]) -> None:
    store, _cache = fake_cache
    store.fail = True
    assert await inv.invalidate_document("doc-1") == 0
    await inv.invalidate_after_embed("doc-1", department_id="CNTT", is_public=False)


@pytest.mark.asyncio
async def test_index_metadata_update_and_delete_invalidate_the_document(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    purged: list[str] = []

    async def fake_invalidate(document_id: str) -> int:
        purged.append(document_id)
        return 1

    monkeypatch.setattr(document_index_service, "invalidate_document", fake_invalidate)
    monkeypatch.setattr(document_index_service, "index_collections", lambda: ("unisage_chunks",))
    client = MagicMock()
    with (
        patch.object(qdrant_store, "set_document_payload", return_value=3),
        patch.object(qdrant_store, "delete_document_points", return_value=3),
    ):
        await document_index_service.update_index_metadata(
            client,
            db_session,
            "doc-1",
            IndexMetadataUpdateRequest(department_id="CNTT", access_level=2, is_public=False),
        )
        await document_index_service.delete_index(client, db_session, "doc-1")
    assert purged == ["doc-1", "doc-1"]


@pytest.mark.asyncio
async def test_embed_task_invalidates_after_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_run_embed_chunks` ends with the hook, carrying the document's own access payload."""

    from app.worker.tasks import ingestion

    hook = AsyncMock()
    monkeypatch.setattr(ingestion, "invalidate_after_embed", hook)
    monkeypatch.setattr(ingestion, "_unchanged_chunk_indexes", lambda *a, **k: set())
    monkeypatch.setattr(ingestion, "_drop_stale_points", lambda *a, **k: None)
    monkeypatch.setattr(ingestion, "index_collections", lambda: ("unisage_chunks",))

    class _Tracker:
        async def reserve_request(self, **_: Any) -> str:
            return "OK"

    class _Recorder:
        request_id = "r"

        def __init__(self, **_: Any) -> None:
            pass

        async def close(self, **_: Any) -> None:
            pass

    monkeypatch.setattr(ingestion, "BudgetTracker", lambda **_: _Tracker())
    monkeypatch.setattr(ingestion, "UsageRecorder", _Recorder)
    monkeypatch.setattr(redis_asyncio.Redis, "from_url", lambda *_a, **_k: AsyncMock())

    embedder = MagicMock()
    embedder.identity_key = "openai:text-embedding-3-small:1536"
    embedder.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    enricher = MagicMock()
    enricher.enrich_tracked = AsyncMock(return_value=MagicMock(summary="s", questions=["q"]))
    monkeypatch.setattr(ingestion, "_pause_between_extractions", AsyncMock(return_value=None))
    monkeypatch.setattr(qdrant_store, "upsert_chunk", lambda *a, **k: None)

    result = await ingestion._run_embed_chunks(
        embedder=embedder,
        enricher=enricher,
        qdrant_client=MagicMock(),
        chunks=[{"chunk_index": 0, "content": "x", "region_type": "text"}],
        document_id="doc-1",
        object_key="k",
        department_id="CNTT",
        access_level=2,
        is_public=False,
        category="HOC_VU",
        user_id=None,
        update_state=lambda **_: None,
        publish=lambda *_: None,
    )
    assert result["percent"] == 100
    hook.assert_awaited_once_with("doc-1", department_id="CNTT", is_public=False)
