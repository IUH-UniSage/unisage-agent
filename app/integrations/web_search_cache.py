"""Redis cache of raw web search results (W3, changes/14-10-2026-Web-Search-Optimization).

Holds what Tavily returned for a query, never a judged or generated answer: the pages are
public and the same for every user, while what is relevant depends on the question. Keyed
by the normalized query and every Tavily parameter that changes the answer, so changing the
configuration never serves stale results; the academic year is part of the query, so a new
year never does either.

Fails open: Redis down or a corrupt entry means a cache miss, never a failed search.
"""

import hashlib
import json
import logging
import re
import unicodedata
from typing import Protocol

from pydantic import TypeAdapter, ValidationError
from redis import asyncio as redis_asyncio

from app.core.config import settings
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)

KEY_PREFIX = "websearch:v1:"
# The query text an entry was searched with, under the same digest and TTL: the entry's key
# is a hash, so this is the only place `task db:web-cache` can read the query back from.
QUERY_KEY_PREFIX = "websearch:query:v1:"
_RESULTS = TypeAdapter(list[WebSearchResult])


class WebSearchCache(Protocol):
    async def get(self, query: str, *, max_results: int) -> list[WebSearchResult] | None: ...

    async def set(
        self, query: str, results: list[WebSearchResult], *, max_results: int, ttl_seconds: int
    ) -> None: ...


def normalize_query(query: str) -> str:
    """Case, accent encoding (NFC) and punctuation never change what Tavily is asked."""

    text = unicodedata.normalize("NFC", query).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _key_parts(query: str, *, max_results: int) -> list[object]:
    return [
        normalize_query(query),
        sorted(settings.TAVILY_INCLUDE_DOMAINS),
        settings.TAVILY_SEARCH_DEPTH,
        settings.TAVILY_CHUNKS_PER_SOURCE if settings.TAVILY_SEARCH_DEPTH == "advanced" else None,
        max_results,
    ]


def cache_key(query: str, *, max_results: int) -> str:
    parts = _key_parts(query, max_results=max_results)
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def describe_key(query: str, *, max_results: int) -> str:
    """What a lookup compares, for the log: the Redis key and every part it is hashed
    from - two queries share an entry only when all of these match."""

    normalized, domains, depth, chunks, _ = _key_parts(query, max_results=max_results)
    return (
        f"key={cache_key(query, max_results=max_results)} normalized={normalized!r} "
        f"depth={depth} chunks_per_source={chunks} max_results={max_results} "
        f"domains={domains}"
    )


def query_key(entry_key: str) -> str:
    return QUERY_KEY_PREFIX + entry_key.removeprefix(KEY_PREFIX)


class RedisWebSearchCache:
    """One short-lived connection per call: a search can run on the request's loop or keep
    running in the background after the turn, and an asyncio Redis connection is bound to
    the loop that opened it."""

    def __init__(self, url: str) -> None:
        self._url = url

    async def get(self, query: str, *, max_results: int) -> list[WebSearchResult] | None:
        try:
            client = redis_asyncio.Redis.from_url(self._url)
            try:
                raw = await client.get(cache_key(query, max_results=max_results))
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Web search cache unavailable - searching without it", exc_info=True)
            return None
        if raw is None:
            return None
        try:
            return _RESULTS.validate_json(raw)
        except ValidationError:
            logger.warning("Web search cache: unreadable entry for query=%r, ignored", query)
            return None

    async def set(
        self, query: str, results: list[WebSearchResult], *, max_results: int, ttl_seconds: int
    ) -> None:
        try:
            client = redis_asyncio.Redis.from_url(self._url)
            key = cache_key(query, max_results=max_results)
            try:
                await client.set(key, _RESULTS.dump_json(results), ex=ttl_seconds)
                await client.set(query_key(key), query, ex=ttl_seconds)
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Web search cache: could not store query=%r", query, exc_info=True)


def default_cache() -> WebSearchCache | None:
    url = settings.WEB_SEARCH_CACHE_REDIS_URL.strip()
    return RedisWebSearchCache(url) if url else None
