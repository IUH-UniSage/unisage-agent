"""Retrieval cache: the pure key helpers and the Redis store on an in-memory fake
(docs/specs/SPEC-retrieval-cache.md). No live Redis anywhere here."""

from __future__ import annotations

import json
import random
import unicodedata
from datetime import UTC, datetime

import pytest

from app.core.config import settings
from app.integrations import retrieval_cache as rc
from app.integrations.retrieval_cache import RedisRetrievalCache
from app.schemas.retrieval_cache import (
    CachedChunkRef,
    CachedRetrievalBundle,
    CachedSubQuery,
    CachedTurnBundle,
)
from tests.fakes.fake_redis import FakeRedisStore

# --------------------------------------------------------------------------- pure helpers


def test_normalize_question_is_nfc_casefolded_and_punctuation_free() -> None:
    composed = "Điều kiện tốt nghiệp là gì ?"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert rc.normalize_question(composed) == "điều kiện tốt nghiệp là gì"
    assert rc.normalize_question(decomposed) == rc.normalize_question(composed)
    assert (
        rc.normalize_question("  HỌC   PHÍ\tkỳ 1 ngành CNTT 2026?!")
        == "học phí kỳ 1 ngành cntt 2026"
    )


def test_normalize_question_keeps_digits_and_codes() -> None:
    assert (
        rc.normalize_question("Môn 2101234 học kỳ 2 năm 2025-2026.")
        == "môn 2101234 học kỳ 2 năm 2025 2026"
    )


def test_normalize_question_ignores_punctuation_the_classifier_varies() -> None:
    """Measured 20/10/2026: the same standalone question came back with and without
    parentheses, quotes or a colon - none of that changes what is asked."""

    with_parens = (
        "Ngành CNKT máy tính (chương trình chất lượng cao 2022): "
        'học phần "Hệ thống máy tính" học trước môn nào?'
    )
    without = (
        "ngành cnkt máy tính chương trình chất lượng cao 2022 "
        "học phần hệ thống máy tính học trước môn nào"
    )
    assert rc.normalize_question(with_parens) == without
    assert rc.normalize_question("học kỳ 1") != rc.normalize_question("học kỳ 2")


def _parts(**overrides: object) -> dict[str, object]:
    parts: dict[str, object] = {
        "v": "1",
        "qs": ["học phí đại học chính quy khóa 2025 ngành cntt là bao nhiêu"],
        "scope": [["CNTT", 2], False],
        "coll": "unisage_chunks",
        "emb": "openai:text-embedding-3-small:1536",
        "gen": {"public": 3, "dept:CNTT": 1},
        "cfg": "abc123",
    }
    parts.update(overrides)
    return parts


def test_cache_key_is_stable_and_prefixed() -> None:
    key = rc.cache_key("l2", _parts())
    assert key == rc.cache_key("l2", _parts())
    assert key.startswith("rc:v1:l2:")
    assert len(key) == len("rc:v1:l2:") + 64
    assert rc.cache_key("l1", _parts()) != key


@pytest.mark.parametrize(
    "override",
    [
        {"qs": ["học phí đại học chính quy khóa 2024 ngành cntt là bao nhiêu"]},
        {"scope": [["CNTT", 3], False]},
        {"scope": [[], True]},
        {"coll": "unisage_chunks__20261020120000"},
        {"emb": "google:gemini-embedding-001:768"},
        {"gen": {"public": 4, "dept:CNTT": 1}},
        {"cfg": "def456"},
        {"date": "2026-10-21"},
    ],
)
def test_cache_key_changes_when_any_part_changes(override: dict[str, object]) -> None:
    assert rc.cache_key("l2", _parts(**override)) != rc.cache_key("l2", _parts())


def test_cache_key_ignores_dict_order_but_not_list_order() -> None:
    a = rc.cache_key("l2", {"gen": {"public": 1, "dept:CNTT": 2}, "qs": ["a", "b"]})
    b = rc.cache_key("l2", {"qs": ["a", "b"], "gen": {"dept:CNTT": 2, "public": 1}})
    c = rc.cache_key("l2", {"qs": ["b", "a"], "gen": {"dept:CNTT": 2, "public": 1}})
    assert a == b
    assert a != c


def test_describe_key_names_every_part() -> None:
    text = rc.describe_key("l2", _parts())
    assert text.startswith("key=rc:v1:l2:")
    for name in ("qs=", "scope=", "coll=", "emb=", "gen=", "cfg="):
        assert name in text


def test_config_fingerprint_tracks_settings_and_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    before_l1, before_l2 = rc.config_fingerprint("l1"), rc.config_fingerprint("l2")
    assert before_l1 != before_l2
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.42)
    assert rc.config_fingerprint("l2") != before_l2
    assert rc.config_fingerprint("l1") != before_l1
    monkeypatch.setattr(
        settings, "CHAT_RERANK_SCORE_THRESHOLD", settings.CHAT_RERANK_SCORE_THRESHOLD
    )
    monkeypatch.setattr(
        settings, "CHAT_CLASSIFY_WITH_RETRIEVAL", not settings.CHAT_CLASSIFY_WITH_RETRIEVAL
    )
    assert rc.config_fingerprint("l2") == rc.config_fingerprint("l2")
    assert rc.config_fingerprint("l1") != before_l1


def test_config_fingerprint_changes_with_the_rerank_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    before = rc.config_fingerprint("l2")
    monkeypatch.setattr(settings, "CHAT_RERANK_MODE", "cross_encoder")
    with_ce = rc.config_fingerprint("l2")
    assert with_ce != before
    monkeypatch.setattr(settings, "CHAT_CROSS_ENCODER_KEEP", settings.CHAT_CROSS_ENCODER_KEEP + 1)
    assert rc.config_fingerprint("l2") != with_ce


def test_ttl_with_jitter_stays_within_ten_percent() -> None:
    rng = random.Random(7)
    for _ in range(50):
        value = rc.ttl_with_jitter(21600, rng)
        assert 21600 * 0.9 <= value <= 21600 * 1.1
    assert rc.ttl_with_jitter(5, rng) == 5


# --------------------------------------------------------------------------- bundles


def _bundle() -> CachedRetrievalBundle:
    return CachedRetrievalBundle(
        created_at=datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
        questions=["Điều kiện tốt nghiệp là gì?"],
        per_query=[
            [
                CachedChunkRef(chunk_id="doc-a:0", score=0.91),
                CachedChunkRef(chunk_id="doc-b:3", score=0.8),
            ]
        ],
        best_scores=[0.91],
        merged=["doc-a:0", "doc-b:3"],
    )


def test_bundle_round_trips_and_lists_ids() -> None:
    bundle = _bundle()
    again = CachedRetrievalBundle.model_validate_json(bundle.model_dump_json())
    assert again == bundle
    assert again.schema_version == 1
    assert again.chunk_ids() == ["doc-a:0", "doc-b:3"]
    assert again.document_ids() == ["doc-a", "doc-b"]
    assert (
        CachedChunkRef(chunk_id="uuid-with:colons:7", score=0.5).document_id == "uuid-with:colons"
    )


def test_turn_bundle_keeps_classifier_output() -> None:
    bundle = CachedTurnBundle(
        **_bundle().model_dump(),
        tasks=[
            {
                "intent": "academic_advisory",
                "query": "q",
                "routing_mode": "SINGLE",
                "hyde_text": "H\n\nP",
                "web": {"topics": ["quy_che_thu_tuc"], "subject": "điều kiện tốt nghiệp"},
            }
        ],
        confidence=0.9,
        sub_queries=[
            CachedSubQuery(question="Điều kiện tốt nghiệp là gì?", retrieval_text="H\n\nP")
        ],
    )
    again = CachedTurnBundle.model_validate_json(bundle.model_dump_json())
    assert again.tasks[0]["web"]["subject"] == "điều kiện tốt nghiệp"
    assert again.sub_queries[0].task_index == 0


# --------------------------------------------------------------------------- the store


@pytest.fixture
def store() -> FakeRedisStore:
    clock = {"now": 1_000.0}
    fake = FakeRedisStore(clock=lambda: clock["now"])
    fake.advance = lambda seconds: clock.__setitem__("now", clock["now"] + seconds)  # type: ignore[attr-defined]
    return fake


def _cache(store: FakeRedisStore, **kwargs: object) -> RedisRetrievalCache:
    kwargs.setdefault("ttl_seconds", 600)
    kwargs.setdefault("rng", random.Random(1))
    return RedisRetrievalCache("redis://unused", client_factory=store.client, **kwargs)  # type: ignore[arg-type]


KEY = rc.cache_key("l2", _parts())
PAYLOAD = _bundle().model_dump_json()


@pytest.mark.asyncio
async def test_second_sighting_policy_writes_on_the_second_miss(store: FakeRedisStore) -> None:
    cache = _cache(store)
    assert await cache.get("l2", KEY) is None
    assert (
        await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=["doc-a", "doc-b"])
        is False
    )
    assert await cache.get("l2", KEY) is None
    digest = KEY.removeprefix("rc:v1:l2:")
    assert rc.seen_key("l2", digest) in store.values

    assert (
        await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=["doc-a", "doc-b"])
        is True
    )
    assert rc.seen_key("l2", digest) not in store.values
    assert await cache.get("l2", KEY) == PAYLOAD
    assert store.values[rc.description_key("l2", digest)] == b"d"
    assert store.sets[rc.document_key("doc-a")] == {KEY.encode()}
    assert store.sets[rc.document_key("doc-b")] == {KEY.encode()}
    # Companion keys outlive the entry a little.
    assert store.ttl(rc.document_key("doc-a")) > store.ttl(KEY)
    assert 540 <= store.ttl(KEY) <= 660


@pytest.mark.asyncio
async def test_marker_expiry_restarts_the_count(store: FakeRedisStore) -> None:
    cache = _cache(store)
    assert await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=[]) is False
    store.advance(700)  # type: ignore[attr-defined]
    assert await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=[]) is False
    assert await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=[]) is True


@pytest.mark.asyncio
async def test_policy_off_writes_at_once(store: FakeRedisStore) -> None:
    cache = _cache(store, write_on_second_sighting=False)
    assert await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=["doc-a"]) is True
    assert await cache.get("l2", KEY) == PAYLOAD


@pytest.mark.asyncio
async def test_entry_expires_absolutely_and_is_not_refreshed_by_reads(
    store: FakeRedisStore,
) -> None:
    cache = _cache(store, write_on_second_sighting=False)
    await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=[])
    store.advance(500)  # type: ignore[attr-defined]
    assert await cache.get("l2", KEY) == PAYLOAD
    store.advance(200)  # type: ignore[attr-defined]
    assert await cache.get("l2", KEY) is None


@pytest.mark.asyncio
async def test_purge_document_drops_only_entries_referencing_it(store: FakeRedisStore) -> None:
    cache = _cache(store, write_on_second_sighting=False)
    other = rc.cache_key("l2", _parts(qs=["khác"]))
    await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=["doc-a", "doc-b"])
    await cache.set("l2", other, PAYLOAD, description="d", document_ids=["doc-c"])
    l1 = rc.cache_key("l1", _parts())
    await cache.set("l1", l1, PAYLOAD, description="d", document_ids=["doc-b"])

    assert await cache.purge_document("doc-b") == 2
    assert await cache.get("l2", KEY) is None
    assert await cache.get("l1", l1) is None
    assert await cache.get("l2", other) == PAYLOAD
    assert rc.document_key("doc-b") not in store.sets
    assert rc.description_key("l2", KEY.removeprefix("rc:v1:l2:")) not in store.values
    assert await cache.purge_document("doc-b") == 0


@pytest.mark.asyncio
async def test_delete_removes_entry_and_companions(store: FakeRedisStore) -> None:
    cache = _cache(store, write_on_second_sighting=False)
    await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=[])
    await cache.delete("l2", KEY)
    assert await cache.get("l2", KEY) is None
    assert not [k for k in store.values if k.endswith(KEY.removeprefix("rc:v1:l2:"))]


@pytest.mark.asyncio
async def test_generations_bump_and_read(store: FakeRedisStore) -> None:
    cache = _cache(store)
    assert await cache.read_generations(["public", "dept:CNTT"]) == {"public": 0, "dept:CNTT": 0}
    await cache.bump_generation("dept:CNTT")
    await cache.bump_generation("dept:CNTT")
    await cache.bump_generation("public")
    assert await cache.read_generations(["public", "dept:CNTT", "dept:KT"]) == {
        "public": 1,
        "dept:CNTT": 2,
        "dept:KT": 0,
    }
    assert await cache.read_generations([]) == {}


@pytest.mark.asyncio
async def test_kill_switch_lock_and_stats(store: FakeRedisStore) -> None:
    cache = _cache(store)
    assert await cache.is_disabled() is False
    store.values[rc.DISABLED_KEY] = b"1"
    assert await cache.is_disabled() is True

    assert await cache.acquire_lock("l2", KEY) is True
    assert await cache.acquire_lock("l2", KEY) is False
    store.advance(11)  # type: ignore[attr-defined]
    assert await cache.acquire_lock("l2", KEY) is True

    await cache.record("l2", "hit")
    await cache.record("l2", "hit")
    await cache.record("l1", "miss_first")
    counters = {k: v for k, v in store.values.items() if ":stats:" in k}
    assert sorted(counters.values()) == [b"1", b"2"]
    assert all(store.ttl(k) > 0 for k in counters)


@pytest.mark.asyncio
async def test_fails_open_when_redis_is_down(
    store: FakeRedisStore, caplog: pytest.LogCaptureFixture
) -> None:
    cache = _cache(store, write_on_second_sighting=False)
    store.fail = True
    assert await cache.get("l2", KEY) is None
    assert await cache.set("l2", KEY, PAYLOAD, description="d", document_ids=["doc-a"]) is False
    await cache.delete("l2", KEY)
    assert await cache.purge_document("doc-a") == 0
    await cache.bump_generation("public")
    assert await cache.is_disabled() is False
    assert await cache.acquire_lock("l2", KEY) is True
    await cache.record("l2", "error")
    with pytest.raises(ConnectionError):
        await cache.read_generations(["public"])
    assert any("Retrieval cache" in message for message in caplog.messages)


def test_default_cache_follows_the_url_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_CACHE_REDIS_URL", "")
    assert rc.default_cache() is None
    monkeypatch.setattr(settings, "RETRIEVAL_CACHE_REDIS_URL", "redis://localhost:6380/2")
    assert isinstance(rc.default_cache(), RedisRetrievalCache)


def test_entry_key_split() -> None:
    assert rc._split_entry_key("rc:v1:l1:abc") == ("l1", "abc")
    with pytest.raises(ValueError):
        rc._split_entry_key("rc:v1:q:l1:abc")
    assert json.loads(rc.canonical_json({"b": 1, "a": [2, 3]})) == {"a": [2, 3], "b": 1}
