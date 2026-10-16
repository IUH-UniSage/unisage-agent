"""RedisWebSearchCache - keys, round trip and failing open, against a fake Redis."""

import unicodedata
from typing import Any

import pytest
from redis import asyncio as redis_asyncio

from app.core.config import settings
from app.integrations import web_search_cache
from app.integrations.web_search_cache import RedisWebSearchCache, cache_key, query_key
from app.schemas.web_search import WebSearchResult

_PAGE = WebSearchResult(title="Học phí", url="https://pdt.iuh.edu.vn/a", content="x", score=0.4)


class _FakeRedis:
    def __init__(self, store: dict[str, Any], *, broken: bool = False) -> None:
        self.store = store
        self.broken = broken
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> Any:
        if self.broken:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: Any, *, ex: int) -> None:
        if self.broken:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.ttls[key] = ex

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "TAVILY_INCLUDE_DOMAINS", ["iuh.edu.vn"])
    monkeypatch.setattr(settings, "TAVILY_SEARCH_DEPTH", "advanced")
    monkeypatch.setattr(settings, "TAVILY_CHUNKS_PER_SOURCE", 3)


def _redis(monkeypatch: pytest.MonkeyPatch, redis: _FakeRedis) -> None:
    monkeypatch.setattr(redis_asyncio.Redis, "from_url", lambda _url: redis)


def test_key_ignores_case_accent_encoding_and_punctuation() -> None:
    decomposed = unicodedata.normalize("NFD", "Mức thu học phí")
    assert cache_key(decomposed, max_results=10) == cache_key("mức thu học phí?", max_results=10)
    assert cache_key("MỨC THU   học phí", max_results=10) == cache_key(
        "mức thu học phí", max_results=10
    )


def test_key_changes_with_the_search_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    before = cache_key("học phí", max_results=10)
    assert cache_key("học phí", max_results=5) != before
    monkeypatch.setattr(settings, "TAVILY_SEARCH_DEPTH", "basic")
    assert cache_key("học phí", max_results=10) != before


@pytest.mark.asyncio
async def test_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _FakeRedis({})
    _redis(monkeypatch, redis)
    cache = RedisWebSearchCache("redis://x/3")

    assert await cache.get("học phí", max_results=10) is None
    await cache.set("học phí", [_PAGE], max_results=10, ttl_seconds=60)

    assert await cache.get("Học phí", max_results=10) == [_PAGE]
    assert list(redis.ttls.values()) == [60, 60]


@pytest.mark.asyncio
async def test_the_query_is_kept_beside_its_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _FakeRedis({})
    _redis(monkeypatch, redis)

    await RedisWebSearchCache("redis://x/3").set(
        "Học phí IUH năm học 2025-2026", [_PAGE], max_results=10, ttl_seconds=60
    )

    key = cache_key("học phí iuh năm học 2025-2026", max_results=10)
    assert redis.store[query_key(key)] == "Học phí IUH năm học 2025-2026"
    assert not query_key(key).startswith(web_search_cache.KEY_PREFIX)


@pytest.mark.asyncio
async def test_redis_down_is_a_miss_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _redis(monkeypatch, _FakeRedis({}, broken=True))
    cache = RedisWebSearchCache("redis://x/3")

    await cache.set("q", [_PAGE], max_results=10, ttl_seconds=60)
    assert await cache.get("q", max_results=10) is None


@pytest.mark.asyncio
async def test_unreadable_entry_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    _redis(monkeypatch, _FakeRedis({cache_key("q", max_results=10): b"not json"}))

    assert await RedisWebSearchCache("redis://x/3").get("q", max_results=10) is None


def test_empty_url_means_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "WEB_SEARCH_CACHE_REDIS_URL", " ")
    assert web_search_cache.default_cache() is None
