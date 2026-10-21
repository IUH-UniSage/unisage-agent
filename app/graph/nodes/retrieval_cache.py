"""RetrievalCacheNode helpers - the graph side of docs/specs/SPEC-retrieval-cache.md.

Two exact-match lookups into `app.integrations.retrieval_cache`:

- L1 (`lookup_l1`/`store_l1`): keyed by the student's own question on a turn with no
  history; a hit returns the classifier's tasks, the sub-queries and the reranked chunks,
  so the graph skips classification, HyDE, embedding, Qdrant and both reranks.
- L2 (`lookup_l2`/`store_l2`): keyed by the standalone questions the classifier/HyDE
  wrote; a hit returns the reranked chunks, so the graph skips embedding, Qdrant and
  both reranks.

A hit never carries chunk text: the chunks are fetched again by id through
`RetrievalServiceProtocol.fetch`, which applies the caller's permission filter. Any id
that does not come back (deleted, re-chunked, no longer visible) turns the hit into a
miss and drops the entry. Everything here is fail-open: a cache error is a miss.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.registry.model_registry import active_collection, get_current_snapshot
from app.core.timezone import now_ict
from app.graph.metadata_fields import canonical_profile
from app.graph.nodes.post_retrieval_rerank import TurnRerankResult, turn_result
from app.graph.nodes.query_transformation import SubQuery
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.retrieval_cache import (
    PUBLIC_SCOPE,
    Layer,
    Outcome,
    RetrievalCache,
    cache_key,
    config_fingerprint,
    default_cache,
    department_scope,
    describe_key,
    normalize_question,
)
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.intent import ClassifiedTask, IntentClassification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_cache import (
    CachedChunkRef,
    CachedRetrievalBundle,
    CachedSubQuery,
    CachedTurnBundle,
)
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

ADVISORY_INTENT = "academic_advisory"


@dataclass(frozen=True)
class TurnCacheContext:
    """What every lookup and store of one turn shares: the store, the caller's scope and
    the generation counters read once at the start of the turn (so a bump mid-turn
    cannot make the turn's own write land under a key nobody will look up)."""

    cache: RetrievalCache
    scope: list[Any]
    generations: dict[str, int]
    collection: str
    embedding: str
    l1_eligible: bool


@dataclass(frozen=True)
class L1Pending:
    """An L1 miss waiting for the turn to finish so its outcome can be stored."""

    key: str
    parts: dict[str, Any]
    confidence: float | None = None


@dataclass(frozen=True)
class CachedTurn:
    """An L1 hit, rebuilt: what the graph would otherwise compute in nodes 03-09a."""

    classification: IntentClassification
    sub_queries: list[SubQuery]
    rerank_result: TurnRerankResult


@dataclass
class L2Lookup:
    key: str
    parts: dict[str, Any]
    result: TurnRerankResult | None = None
    questions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- context


def scope_of(security: AcademicSecurityContext) -> tuple[list[Any], list[str]]:
    """The permission scope as it goes into a key (sorted department access entries plus
    the guest flag) and the generation scopes it depends on."""

    entries = sorted(
        (entry.department_id, entry.access_level) for entry in security.department_access
    )
    scope: list[Any] = [[list(entry) for entry in entries], security.role == "KHACH"]
    generation_scopes = [PUBLIC_SCOPE, *(department_scope(dept) for dept, _ in entries)]
    return scope, list(dict.fromkeys(generation_scopes))


def embedding_identity(models: GraphModels) -> str:
    """The embedding model behind retrieval: the embedder's identity key when the
    service exposes one, else the registry's identity, else the configured model name."""

    embedder = getattr(models.retrieval, "embedder", None)
    identity = getattr(embedder, "identity_key", None)
    if isinstance(identity, str) and identity:
        return identity
    snapshot = get_current_snapshot()
    if snapshot is not None and snapshot.embedding_index_identity is not None:
        found = snapshot.embedding_index_identity
        return f"{found.provider}:{found.model_name}:{found.dimension}"
    return "unknown"


async def open_turn_cache(graph_input: GraphInput, models: GraphModels) -> TurnCacheContext | None:
    """None when the cache is off, disabled by the kill switch, or unreachable."""

    cache = default_cache()
    if cache is None:
        return None
    started = time.perf_counter()
    try:
        if await cache.is_disabled():
            logger.info("retrieval_cache outcome=bypass reason=disabled")
            await cache.record("l2", "bypass")
            return None
        scope, generation_scopes = scope_of(graph_input.security)
        generations = await cache.read_generations(generation_scopes)
    except Exception:
        logger.warning(
            "retrieval_cache outcome=error reason=unavailable ms=%d",
            _ms(started),
            exc_info=True,
        )
        return None
    return TurnCacheContext(
        cache=cache,
        scope=scope,
        generations=generations,
        collection=active_collection(),
        embedding=embedding_identity(models),
        l1_eligible=not graph_input.history and graph_input.resume is None,
    )


def _common_parts(ctx: TurnCacheContext, layer: Layer) -> dict[str, Any]:
    return {
        "scope": ctx.scope,
        "coll": ctx.collection,
        "emb": ctx.embedding,
        "gen": ctx.generations,
        "cfg": config_fingerprint(layer),
    }


def l2_parts(ctx: TurnCacheContext, questions: Sequence[str]) -> dict[str, Any]:
    return {**_common_parts(ctx, "l2"), "qs": [normalize_question(q) for q in questions]}


def l1_parts(
    ctx: TurnCacheContext, graph_input: GraphInput, confirmed_metadata: dict[str, str]
) -> dict[str, Any]:
    today = now_ict().date()
    return {
        **_common_parts(ctx, "l1"),
        "q": normalize_question(graph_input.user_message),
        "role": graph_input.security.role,
        "profile": canonical_profile(confirmed_metadata, today),
        "date": today.isoformat(),
    }


# --------------------------------------------------------------------------- rebuild


async def _rebuild(
    bundle: CachedRetrievalBundle,
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
) -> TurnRerankResult | None:
    """The rerank result the bundle describes, with chunks fetched fresh under the
    caller's filter. None when any id is missing - the entry is stale."""

    wanted = bundle.chunk_ids()
    fetched = await asyncio.to_thread(retrieval_service.fetch, wanted, security=security)
    by_id = {chunk.chunk_id: chunk for chunk in fetched}
    missing = [chunk_id for chunk_id in wanted if chunk_id not in by_id]
    if missing:
        logger.info("retrieval_cache stale missing=%d of %d", len(missing), len(wanted))
        return None
    per_query_kept = [
        [by_id[ref.chunk_id].model_copy(update={"score": ref.score}) for ref in refs]
        for refs in bundle.per_query
    ]
    return turn_result(per_query_kept, best_scores=list(bundle.best_scores))


def _bundle_of(questions: Sequence[str], rerank_result: TurnRerankResult) -> CachedRetrievalBundle:
    return CachedRetrievalBundle(
        created_at=datetime.now(UTC),
        questions=list(questions),
        per_query=[
            [CachedChunkRef(chunk_id=chunk.chunk_id, score=chunk.score) for chunk in result.chunks]
            for result in rerank_result.per_query
        ],
        best_scores=list(rerank_result.best_scores),
        merged=[chunk.chunk_id for chunk in rerank_result.chunks],
    )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def _record(ctx: TurnCacheContext, layer: Layer, outcome: Outcome) -> None:
    await ctx.cache.record(layer, outcome)


# --------------------------------------------------------------------------- L2


async def lookup_l2(
    ctx: TurnCacheContext,
    questions: Sequence[str],
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
) -> L2Lookup:
    started = time.perf_counter()
    parts = l2_parts(ctx, questions)
    key = cache_key("l2", parts)
    lookup = L2Lookup(key=key, parts=parts, questions=list(questions))
    raw = await ctx.cache.get("l2", key)
    if raw is None:
        logger.info("retrieval_cache layer=l2 outcome=miss key=%s ms=%d", key, _ms(started))
        logger.debug("retrieval_cache layer=l2 %s", describe_key("l2", parts))
        return lookup
    try:
        bundle = CachedRetrievalBundle.model_validate_json(raw)
        result = await _rebuild(bundle, retrieval_service, security)
    except Exception:
        logger.warning("retrieval_cache layer=l2 outcome=error key=%s", key, exc_info=True)
        await ctx.cache.delete("l2", key)
        await _record(ctx, "l2", "error")
        return lookup
    if result is None:
        await ctx.cache.delete("l2", key)
        await _record(ctx, "l2", "stale")
        logger.info("retrieval_cache layer=l2 outcome=stale key=%s ms=%d", key, _ms(started))
        return lookup
    lookup.result = result
    await _record(ctx, "l2", "hit")
    logger.info(
        "retrieval_cache layer=l2 outcome=hit key=%s chunks=%d ms=%d",
        key,
        len(result.chunks),
        _ms(started),
    )
    return lookup


async def store_l2(
    ctx: TurnCacheContext, lookup: L2Lookup, rerank_result: TurnRerankResult
) -> None:
    """After a miss whose retrieval and LLM rerank succeeded with at least one chunk."""

    if not rerank_result.has_valid_context:
        return
    if not await ctx.cache.acquire_lock("l2", lookup.key):
        # Another turn is computing the same miss right now: one write is enough.
        return
    bundle = _bundle_of(lookup.questions, rerank_result)
    written = await ctx.cache.set(
        "l2",
        lookup.key,
        bundle.model_dump_json(),
        description=describe_key("l2", lookup.parts),
        document_ids=bundle.document_ids(),
    )
    await _record(ctx, "l2", "miss_second" if written else "miss_first")
    logger.info(
        "retrieval_cache layer=l2 outcome=%s key=%s",
        "miss_second" if written else "miss_first",
        lookup.key,
    )


# --------------------------------------------------------------------------- L1


async def lookup_l1(
    ctx: TurnCacheContext,
    graph_input: GraphInput,
    confirmed_metadata: dict[str, str],
    retrieval_service: RetrievalServiceProtocol,
) -> tuple[CachedTurn | None, L1Pending | None]:
    """(hit, None) on a hit; (None, pending) on a miss the turn may store later; (None,
    None) when the cache said nothing usable and the turn should not store either."""

    started = time.perf_counter()
    parts = l1_parts(ctx, graph_input, confirmed_metadata)
    key = cache_key("l1", parts)
    pending = L1Pending(key=key, parts=parts)
    raw = await ctx.cache.get("l1", key)
    if raw is None:
        logger.info("retrieval_cache layer=l1 outcome=miss key=%s ms=%d", key, _ms(started))
        logger.debug("retrieval_cache layer=l1 %s", describe_key("l1", parts))
        return None, pending
    try:
        bundle = CachedTurnBundle.model_validate_json(raw)
        result = await _rebuild(bundle, retrieval_service, graph_input.security)
        classification = IntentClassification(
            tasks=[ClassifiedTask.model_validate(task) for task in bundle.tasks],
            confidence=bundle.confidence,
        )
    except Exception:
        logger.warning("retrieval_cache layer=l1 outcome=error key=%s", key, exc_info=True)
        await ctx.cache.delete("l1", key)
        await _record(ctx, "l1", "error")
        return None, None
    if result is None:
        await ctx.cache.delete("l1", key)
        await _record(ctx, "l1", "stale")
        logger.info("retrieval_cache layer=l1 outcome=stale key=%s ms=%d", key, _ms(started))
        return None, pending
    sub_queries = [
        SubQuery(question=sq.question, retrieval_text=sq.retrieval_text, task_index=sq.task_index)
        for sq in bundle.sub_queries
    ]
    await _record(ctx, "l1", "hit")
    logger.info(
        "retrieval_cache layer=l1 outcome=hit key=%s chunks=%d ms=%d",
        key,
        len(result.chunks),
        _ms(started),
    )
    return CachedTurn(
        classification=classification, sub_queries=sub_queries, rerank_result=result
    ), None


def l1_storable(classification: IntentClassification) -> bool:
    """Only a turn the classifier read as advisory-only, with nothing to save to the
    profile: a hit replays the classification, so it must have had no side effect."""

    return not classification.metadata_updates and all(
        task.intent == ADVISORY_INTENT for task in classification.tasks
    )


async def store_l1(
    ctx: TurnCacheContext,
    pending: L1Pending,
    tasks: Sequence[ClassifiedTask],
    sub_queries: Sequence[SubQuery],
    rerank_result: TurnRerankResult,
) -> None:
    if not rerank_result.has_valid_context:
        return
    if not await ctx.cache.acquire_lock("l1", pending.key):
        return
    questions = [sq.question for sq in sub_queries]
    base = _bundle_of(questions, rerank_result)
    bundle = CachedTurnBundle(
        **base.model_dump(),
        tasks=[_dump_task(task) for task in tasks],
        confidence=pending.confidence,
        sub_queries=[
            CachedSubQuery(
                question=sq.question, retrieval_text=sq.retrieval_text, task_index=sq.task_index
            )
            for sq in sub_queries
        ],
    )
    written = await ctx.cache.set(
        "l1",
        pending.key,
        bundle.model_dump_json(),
        description=describe_key("l1", pending.parts),
        document_ids=bundle.document_ids(),
    )
    await _record(ctx, "l1", "miss_second" if written else "miss_first")
    logger.info(
        "retrieval_cache layer=l1 outcome=%s key=%s",
        "miss_second" if written else "miss_first",
        pending.key,
    )


def _dump_task(task: ClassifiedTask) -> dict[str, Any]:
    """Every field, including the ones the model excludes from its own dump."""

    return {name: getattr(task, name) for name in ClassifiedTask.model_fields} | {
        "web": task.web.model_dump(mode="json") if task.web is not None else None,
        "sub_query_web": (
            [
                slots.model_dump(mode="json") if slots is not None else None
                for slots in task.sub_query_web
            ]
            if task.sub_query_web is not None
            else None
        ),
    }


def chunks_of(result: TurnRerankResult) -> list[RetrievedChunk]:
    return list(result.chunks)
