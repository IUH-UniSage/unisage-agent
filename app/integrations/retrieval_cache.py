"""Redis cache of a turn's retrieval outcome (docs/specs/SPEC-retrieval-cache.md).

Two exact-match layers share one store: `l1` keyed by the student's own question (turns
with no history) and `l2` keyed by the standalone questions the classifier/HyDE wrote.
Entries hold chunk ids and scores only (`app.schemas.retrieval_cache`); the graph fetches
the chunks again with the caller's permission filter on every hit.

Mirrors `app.integrations.web_search_cache`: one short-lived connection per call (a write
may finish on a background task after the turn's loop moved on), pure key helpers that
need no Redis, and fail-open everywhere - Redis down or an unreadable entry is a miss,
never a failed turn.

Key layout (all under `rc:v1:`):
    {layer}:{digest}        the entry (JSON bundle), absolute TTL with jitter
    q:{layer}:{digest}      `describe_key()` text, same TTL - what the CLI reads back
    seen:{layer}:{digest}   first-sighting marker (write-on-second-sighting policy)
    doc:{document_id}       set of entry keys referencing the document (purge on edit/delete)
    gen:{scope}             generation counters: `public` or `dept:{department_id}`
    stats:{date}:{layer}:{outcome}   daily counters the CLI turns into hit rates
    lock:{layer}:{digest}   single-flight marker for a miss being computed
    disabled                kill switch ("1" = every lookup is a bypass)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Literal, Protocol

from redis import asyncio as redis_asyncio

from app.core.config import settings
from app.core.timezone import now_ict
from app.rag.prompting.loader import get_known_metadata_fields, get_templates

logger = logging.getLogger(__name__)

Layer = Literal["l1", "l2"]
Outcome = Literal["hit", "miss_first", "miss_second", "stale", "bypass", "error"]
OUTCOMES: tuple[Outcome, ...] = ("hit", "miss_first", "miss_second", "stale", "bypass", "error")

KEY_PREFIX = "rc:v1:"
PUBLIC_SCOPE = "public"
# Reverse index / description keys outlive the entry a little so a purge that races an
# expiry still finds nothing dangling.
COMPANION_TTL_EXTRA_SECONDS = 300
STATS_TTL_SECONDS = 30 * 24 * 3600
LOCK_TTL_MS = 10_000
TTL_JITTER = 0.10

# Every punctuation mark goes (not only the trailing ones): the classifier writes the same
# standalone question with or without parentheses, quotes or a colon from one call to the
# next (measured 20/10/2026, changes/20-10-2026-RetrievalCache/plan.md), and none of them
# changes what is asked. Letters, digits and spaces are all that is left.
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


# --------------------------------------------------------------------------- pure helpers


def normalize_question(text: str) -> str:
    """The exact-match form of a question. NFC first: Vietnamese typed on different
    devices arrives composed or decomposed and must hash the same. Then casefold, every
    punctuation mark dropped, one space between words. Digits, course codes and years
    stay as they are - "học kỳ 1" and "học kỳ 2" are different questions."""

    text = unicodedata.normalize("NFC", text).casefold()
    text = _PUNCTUATION.sub(" ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def canonical_json(value: Any) -> str:
    """Stable serialization: sorted keys, no whitespace, UTF-8 as-is."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(parts: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


def entry_key(layer: Layer, digest: str) -> str:
    return f"{KEY_PREFIX}{layer}:{digest}"


def description_key(layer: Layer, digest: str) -> str:
    return f"{KEY_PREFIX}q:{layer}:{digest}"


def seen_key(layer: Layer, digest: str) -> str:
    return f"{KEY_PREFIX}seen:{layer}:{digest}"


def lock_key(layer: Layer, digest: str) -> str:
    return f"{KEY_PREFIX}lock:{layer}:{digest}"


def document_key(document_id: str) -> str:
    return f"{KEY_PREFIX}doc:{document_id}"


def generation_key(scope: str) -> str:
    return f"{KEY_PREFIX}gen:{scope}"


def stats_key(day: date, layer: Layer, outcome: Outcome) -> str:
    return f"{KEY_PREFIX}stats:{day.isoformat()}:{layer}:{outcome}"


DISABLED_KEY = f"{KEY_PREFIX}disabled"


def department_scope(department_id: str) -> str:
    return f"dept:{department_id}"


def cache_key(layer: Layer, parts: Mapping[str, Any]) -> str:
    """`rc:v1:{layer}:{sha256}` of the canonical JSON of `parts`. Every field that can
    change the retrieval outcome is in `parts` (see the spec's "Khoá" table), so two
    turns share an entry only when all of them match."""

    return entry_key(layer, digest_of(parts))


def describe_key(layer: Layer, parts: Mapping[str, Any]) -> str:
    """What a lookup compares, for the log and the CLI: the key plus every part it is
    hashed from, so a miss on a question that "should" hit shows which part differed."""

    fields = " ".join(f"{name}={canonical_json(value)}" for name, value in sorted(parts.items()))
    return f"key={cache_key(layer, parts)} {fields}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_fingerprint(layer: Layer) -> str:
    """Hash of everything in settings and prompt templates that shapes the cached
    outcome: a changed top-k, threshold or prompt gets a new key on its own, with no
    version bump to remember. L1 adds the classifier/HyDE side."""

    templates = get_templates()
    parts: dict[str, Any] = {
        "v": settings.RETRIEVAL_CACHE_KEY_VERSION,
        "max_chunks": settings.CHAT_RETRIEVAL_MAX_CHUNKS,
        "context_max_chunks": settings.CHAT_CONTEXT_MAX_CHUNKS,
        "score_threshold": settings.CHAT_RERANK_SCORE_THRESHOLD,
        "llm_rerank": settings.CHAT_LLM_RERANK_ENABLED,
        "rescue_min_score": settings.CHAT_LLM_RERANK_RESCUE_MIN_SCORE,
        "rescue_keep": settings.CHAT_LLM_RERANK_RESCUE_KEEP,
        # Cross-encoder rerank (SPEC-cross-encoder-rerank.md): which chunks survive 09a
        # depends on the mode and on how many the cross-encoder keeps.
        "rerank_mode": settings.CHAT_RERANK_MODE,
        "ce_keep": settings.CHAT_CROSS_ENCODER_KEEP,
        "evidence_min_cosine": settings.CHAT_EVIDENCE_MIN_COSINE,
        "evidence_min_ce": settings.CHAT_EVIDENCE_MIN_CE,
        "rerank_template": _sha(templates.agent_reranker_compressor),
        # Hybrid retrieval (SPEC-hybrid-retrieval.md): a different candidate set per setting.
        "hybrid": settings.RETRIEVAL_HYBRID_ENABLED,
        "dense_floor": settings.CHAT_RETRIEVAL_DENSE_FLOOR,
        "lexical_hit_score": settings.CHAT_RETRIEVAL_LEXICAL_HIT_SCORE,
        "rrf_k": settings.CHAT_RETRIEVAL_RRF_K,
        "candidates_multiplier": settings.CHAT_RETRIEVAL_HYBRID_CANDIDATES_MULTIPLIER,
    }
    if layer == "l1":
        parts.update(
            {
                "classify_with_retrieval": settings.CHAT_CLASSIFY_WITH_RETRIEVAL,
                "classification_template": _sha(templates.agent_message_classification),
                "classification_retrieval_template": _sha(
                    templates.agent_message_classification_retrieval
                ),
                "hyde_template": _sha(templates.agent_hyde_generator),
                "decomposer_template": _sha(templates.agent_multi_query_decomposer),
                "declared_attributes_template": _sha(templates.declared_attributes),
                "known_fields": _sha(canonical_json(get_known_metadata_fields())),
            }
        )
    return _sha(canonical_json(parts))


def ttl_with_jitter(ttl_seconds: int, rng: random.Random | None = None) -> int:
    """+/-10% so entries written together do not expire together."""

    spread = int(ttl_seconds * TTL_JITTER)
    if spread <= 0:
        return ttl_seconds
    return ttl_seconds + (rng or random).randint(-spread, spread)


# --------------------------------------------------------------------------- the store


class RetrievalCache(Protocol):
    """What the graph and the invalidation hooks need. Values are the JSON text of a
    bundle; parsing into `CachedRetrievalBundle`/`CachedTurnBundle` is the caller's."""

    async def get(self, layer: Layer, key: str) -> str | None: ...

    async def set(
        self,
        layer: Layer,
        key: str,
        payload: str,
        *,
        description: str,
        document_ids: Iterable[str],
    ) -> bool: ...

    async def delete(self, layer: Layer, key: str) -> None: ...

    async def purge_document(self, document_id: str) -> int: ...

    async def bump_generation(self, scope: str) -> None: ...

    async def read_generations(self, scopes: Sequence[str]) -> dict[str, int]: ...

    async def is_disabled(self) -> bool: ...

    async def acquire_lock(self, layer: Layer, key: str) -> bool: ...

    async def record(self, layer: Layer, outcome: Outcome) -> None: ...


class RedisRetrievalCache:
    """`RetrievalCache` on a Redis URL. `client_factory` exists for tests (an in-memory
    fake); production opens one asyncio client per call, bound to the calling loop."""

    def __init__(
        self,
        url: str,
        *,
        client_factory: Callable[[], Any] | None = None,
        ttl_seconds: int | None = None,
        write_on_second_sighting: bool | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._url = url
        self._client_factory = client_factory
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else settings.RETRIEVAL_CACHE_TTL_SECONDS
        )
        self._second_sighting = (
            write_on_second_sighting
            if write_on_second_sighting is not None
            else settings.RETRIEVAL_CACHE_WRITE_ON_SECOND_SIGHTING
        )
        self._rng = rng

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        timeout = settings.RETRIEVAL_CACHE_TIMEOUT_MS / 1000
        return redis_asyncio.Redis.from_url(
            self._url, socket_timeout=timeout, socket_connect_timeout=timeout
        )

    @staticmethod
    def _digest(layer: Layer, key: str) -> str:
        return key.removeprefix(entry_key(layer, ""))

    async def get(self, layer: Layer, key: str) -> str | None:
        try:
            client = self._client()
            try:
                raw = await client.get(key)
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Retrieval cache unavailable - retrieving without it", exc_info=True)
            return None
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    async def set(
        self,
        layer: Layer,
        key: str,
        payload: str,
        *,
        description: str,
        document_ids: Iterable[str],
    ) -> bool:
        """True when the entry was written. Under the write-on-second-sighting policy the
        first call for a key only leaves a marker and returns False."""

        digest = self._digest(layer, key)
        ttl = ttl_with_jitter(self._ttl_seconds, self._rng)
        try:
            client = self._client()
            try:
                if self._second_sighting:
                    first = await client.set(seen_key(layer, digest), "1", ex=ttl, nx=True)
                    if first:
                        # The marker is the whole write; free the single-flight lock so the
                        # next sighting (which may come seconds later) can store.
                        await client.delete(lock_key(layer, digest))
                        return False
                pipe = client.pipeline(transaction=True)
                pipe.set(key, payload, ex=ttl)
                pipe.set(
                    description_key(layer, digest),
                    description,
                    ex=ttl + COMPANION_TTL_EXTRA_SECONDS,
                )
                for document_id in dict.fromkeys(document_ids):
                    pipe.sadd(document_key(document_id), key)
                    pipe.expire(document_key(document_id), ttl + COMPANION_TTL_EXTRA_SECONDS)
                pipe.delete(seen_key(layer, digest), lock_key(layer, digest))
                await pipe.execute()
                return True
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Retrieval cache: could not store %s", key, exc_info=True)
            return False

    async def delete(self, layer: Layer, key: str) -> None:
        digest = self._digest(layer, key)
        try:
            client = self._client()
            try:
                await client.delete(key, description_key(layer, digest), seen_key(layer, digest))
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Retrieval cache: could not delete %s", key, exc_info=True)

    async def purge_document(self, document_id: str) -> int:
        """Drop every entry that references `document_id` (edited, deleted, re-embedded,
        or its access payload changed). Returns how many entries went."""

        try:
            client = self._client()
            try:
                members = await client.smembers(document_key(document_id))
                keys = [m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members]
                if not keys:
                    await client.delete(document_key(document_id))
                    return 0
                pipe = client.pipeline(transaction=True)
                for key in keys:
                    layer, digest = _split_entry_key(key)
                    pipe.delete(key, description_key(layer, digest), seen_key(layer, digest))
                pipe.delete(document_key(document_id))
                await pipe.execute()
                logger.info(
                    "Retrieval cache: purged %d entr(y/ies) of document %s", len(keys), document_id
                )
                return len(keys)
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Retrieval cache: purge of %s failed", document_id, exc_info=True)
            return 0

    async def bump_generation(self, scope: str) -> None:
        """New content in `scope` (`public` or `dept:{id}`): every key of a user who can
        see that scope changes, nothing else does."""

        try:
            client = self._client()
            try:
                await client.incr(generation_key(scope))
            finally:
                await client.aclose()
        except Exception:
            logger.warning("Retrieval cache: generation bump for %s failed", scope, exc_info=True)

    async def read_generations(self, scopes: Sequence[str]) -> dict[str, int]:
        """Current counters, 0 for a scope never bumped. Raises on a Redis error: the
        caller treats that as "cache unavailable" for the turn (no key can be built)."""

        if not scopes:
            return {}
        client = self._client()
        try:
            values = await client.mget([generation_key(scope) for scope in scopes])
        finally:
            await client.aclose()
        return {
            scope: int(value) if value is not None else 0
            for scope, value in zip(scopes, values, strict=True)
        }

    async def is_disabled(self) -> bool:
        try:
            client = self._client()
            try:
                return bool(await client.exists(DISABLED_KEY))
            finally:
                await client.aclose()
        except Exception:
            return False

    async def acquire_lock(self, layer: Layer, key: str) -> bool:
        """Single-flight marker for a miss, released by `set` (or by its 10 s TTL if the
        write never came). The loser does not wait: it computes too and skips the write.
        A Redis error counts as "acquired" so a turn is never blocked."""

        try:
            client = self._client()
            try:
                return bool(
                    await client.set(
                        lock_key(layer, self._digest(layer, key)), "1", px=LOCK_TTL_MS, nx=True
                    )
                )
            finally:
                await client.aclose()
        except Exception:
            return True

    async def record(self, layer: Layer, outcome: Outcome) -> None:
        try:
            client = self._client()
            try:
                key = stats_key(now_ict().date(), layer, outcome)
                pipe = client.pipeline(transaction=True)
                pipe.incr(key)
                pipe.expire(key, STATS_TTL_SECONDS)
                await pipe.execute()
            finally:
                await client.aclose()
        except Exception:
            logger.debug("Retrieval cache: stats write failed", exc_info=True)


def _split_entry_key(key: str) -> tuple[Layer, str]:
    rest = key.removeprefix(KEY_PREFIX)
    layer, _, digest = rest.partition(":")
    if layer not in ("l1", "l2"):
        raise ValueError(f"not an entry key: {key}")
    return layer, digest  # type: ignore[return-value]


def default_cache() -> RetrievalCache | None:
    url = settings.RETRIEVAL_CACHE_REDIS_URL.strip()
    return RedisRetrievalCache(url) if url else None
