"""Embedding re-index worker (changes/17-10-2026-Embedding-Reindex in unisage-backend, plan.md §3).

Claims one queued run from backend-java, re-embeds every point of the live collection into a new
collection with the target model, then reports the result - on success Java cuts over (live
collection -> target, target model ACTIVE) and bumps the registry version.

Re-embeds from the payload each point already holds (`content`, `summary`, `questions`), with the
same three texts ingestion embedded (`app.worker.tasks.ingestion`), so no PDF is parsed and no
EXTRACTION call is made. The claimed credential (plaintext API key) lives only in this call stack:
the Celery task that runs this takes no arguments.

A claim with `rebuild=true` (docs/specs/SPEC-hybrid-retrieval.md, the production way to give a
collection the `bm25` sparse vector) targets the live model itself: every point is copied with
its dense vectors as they are and gets `bm25` from its payload - no embedding call at all. Java
then only moves the pointer.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from qdrant_client import QdrantClient, models
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker, RequestBudgetRejectedError
from app.core.config import settings
from app.core.registry.embedding_identity import identity_key
from app.core.registry.model_registry import CredentialConfig
from app.core.security.redaction import safe_error_message
from app.core.usage.usage_recorder import UsageRecorder
from app.integrations.backend_java_client import BackendJavaClient, BackendJavaHTTPError
from app.rag.embeddings.provider import EmbeddingProvider, build_embedder_for
from app.rag.retrieval.lexical import lexical_text, sparse_vector
from app.rag.vectorstore import qdrant_store

logger = logging.getLogger(__name__)

CANCELLED = "CANCELLED"


class ReindexError(Exception):
    """The run cannot complete; its (secret-free) message becomes the run's error."""


class _RunCancelled(Exception):
    pass


@dataclass
class _Progress:
    total: int | None = None
    processed: int = 0
    failed: int = 0


@dataclass(frozen=True)
class _Claim:
    run_id: str
    lease_token: str
    source: str
    target: str
    credential: CredentialConfig
    model_source_ref: str | None
    dimension: int
    fingerprint: list[float]
    rebuild: bool = False


def _parse_claim(raw: dict[str, Any]) -> _Claim:
    candidate = raw["credential"]
    return _Claim(
        run_id=str(raw["runId"]),
        lease_token=str(raw["leaseToken"]),
        source=str(raw["sourceCollection"]),
        target=str(raw["targetCollection"]),
        credential=CredentialConfig(
            id=str(candidate["chatModelId"]),
            revision=int(raw.get("credentialRevision") or 0),
            source_type=str(candidate.get("sourceType") or ""),
            provider=candidate.get("provider"),
            model_name=candidate.get("modelName"),
            api_base_url=candidate.get("apiBaseUrl"),
            priority=None,
            max_rpm=candidate.get("maxRpm"),
            api_key=candidate.get("apiKey") or "",
            max_concurrency=candidate.get("maxConcurrency"),
        ),
        model_source_ref=candidate.get("modelSourceRef"),
        dimension=int(raw["dimension"]),
        fingerprint=[float(value) for value in raw.get("fingerprint") or []],
        rebuild=bool(raw.get("rebuild") or False),
    )


def sparse_of_payload(payload: dict[str, Any]) -> tuple[list[int], list[float]]:
    """The `bm25` vector a point gets in the target collection, from the same payload fields
    ingestion used (`app.worker.tasks.ingestion`): heading path + content."""

    return sparse_vector(
        lexical_text(str(payload.get("content") or ""), list(payload.get("heading_path") or []))
    )


def embedding_texts(payload: dict[str, Any]) -> list[str]:
    """The three texts ingestion embedded for a point: content, summary, questions - each empty
    one falling back to the content (`app.worker.tasks.ingestion._run_embed_chunks`)."""

    content = str(payload.get("content") or "")
    summary = str(payload.get("summary") or "") or content
    questions = " ".join(str(q) for q in payload.get("questions") or []) or content
    return [content, summary, questions]


async def run_embedding_reindex_once(
    *, backend: BackendJavaClient | None = None, qdrant: QdrantClient | None = None
) -> bool:
    """Claims and runs at most one re-index. Returns whether there was one to run."""

    backend = backend or BackendJavaClient()
    claims = await backend.claim_embedding_reindex(limit=1)
    if not claims:
        return False
    claim = _parse_claim(claims[0])
    await _run(claim, backend, qdrant or qdrant_store.get_client())
    return True


async def _run(claim: _Claim, backend: BackendJavaClient, qdrant: QdrantClient) -> None:
    progress = _Progress()
    logger.info(
        "embedding re-index %s%s: %s -> %s with %s/%s (dimension %d)",
        claim.run_id,
        " (rebuild, copy + bm25)" if claim.rebuild else "",
        claim.source,
        claim.target,
        claim.credential.provider,
        claim.credential.model_name,
        claim.dimension,
    )
    try:
        key = await _prepare_target(claim, backend, qdrant)
        progress.total = qdrant_store.count_points(qdrant, claim.source)
        await _report(claim, backend, progress)
        if claim.rebuild:
            await _copy_collection(claim, backend, qdrant, key, progress)
        else:
            embedder = build_embedder_for(claim.credential, key)
            await _reembed_collection(claim, backend, qdrant, embedder, key, progress)
        if progress.failed:
            raise ReindexError(f"{progress.failed} point(s) could not be re-embedded")
    except _RunCancelled:
        logger.info("embedding re-index %s: cancelled, dropping %s", claim.run_id, claim.target)
        qdrant_store.delete_collection(qdrant, claim.target)
        return
    except BackendJavaHTTPError as exc:
        if exc.status_code == 409:
            # Lease lost: another worker owns the run now - leave its target collection alone.
            logger.warning("embedding re-index %s: lease lost, stopping", claim.run_id)
            return
        await _fail(claim, backend, qdrant, progress, exc)
        return
    except Exception as exc:
        await _fail(claim, backend, qdrant, progress, exc)
        return

    status = await backend.post_embedding_reindex_result(
        run_id=claim.run_id,
        lease_token=claim.lease_token,
        succeeded=True,
        total_points=progress.total,
        processed_points=progress.processed,
        failed_points=progress.failed,
    )
    logger.info(
        "embedding re-index %s: %d point(s) %s into %s, run %s",
        claim.run_id,
        progress.processed,
        "copied with bm25" if claim.rebuild else "re-embedded",
        claim.target,
        status,
    )
    if status != "SUCCEEDED":
        # Java refused the cutover (model edited meanwhile, ...): the target is unused.
        qdrant_store.delete_collection(qdrant, claim.target)


async def _fail(
    claim: _Claim,
    backend: BackendJavaClient,
    qdrant: QdrantClient,
    progress: _Progress,
    exc: Exception,
) -> None:
    message = safe_error_message(exc, claim.credential.api_key)
    logger.error("embedding re-index %s failed: %s", claim.run_id, message)
    try:
        qdrant_store.delete_collection(qdrant, claim.target)
    except Exception:
        logger.warning(
            "embedding re-index %s: dropping %s failed", claim.run_id, claim.target, exc_info=True
        )
    try:
        await backend.post_embedding_reindex_result(
            run_id=claim.run_id,
            lease_token=claim.lease_token,
            succeeded=False,
            total_points=progress.total,
            processed_points=progress.processed,
            failed_points=progress.failed,
            error_message=message[:500],
        )
    except Exception:
        # The lease then expires and the run is claimed again from scratch.
        logger.exception("embedding re-index %s: reporting the failure failed", claim.run_id)


async def _prepare_target(claim: _Claim, backend: BackendJavaClient, qdrant: QdrantClient) -> str:
    """Creates the target collection and registers its identity; returns the identity key its
    points carry. A target left by an earlier attempt of the same run is reused (upserts are
    idempotent per point id); its identity must then match."""

    if not claim.fingerprint:
        raise ReindexError("the target model has no measured fingerprint - verify it again")
    existing_dimension = qdrant_store.get_collection_dimension(qdrant, claim.target)
    if existing_dimension is not None and existing_dimension != claim.dimension:
        qdrant_store.delete_collection(qdrant, claim.target)
    qdrant_store.ensure_collection(qdrant, claim.target, claim.dimension)

    credential = claim.credential
    try:
        await backend.put_embedding_index_identity(
            collection=claim.target,
            provider=credential.provider or "",
            model_name=credential.model_name or "",
            model_source_ref=claim.model_source_ref,
            api_base_url=credential.api_base_url,
            dimension=claim.dimension,
            fingerprint=claim.fingerprint,
            established_by="reindex",
        )
    except BackendJavaHTTPError as exc:
        if exc.status_code != 409:
            raise
        registered = await backend.get_embedding_index_identity(collection=claim.target) or {}
        if (
            registered.get("provider") != credential.provider
            or registered.get("modelName") != credential.model_name
            or registered.get("dimension") != claim.dimension
        ):
            raise ReindexError(
                f"collection {claim.target} already has another model's identity"
            ) from exc

    return identity_key(
        credential.provider or "",
        credential.model_name or "",
        claim.model_source_ref,
        credential.api_base_url,
        claim.dimension,
    )


async def _report(claim: _Claim, backend: BackendJavaClient, progress: _Progress) -> None:
    """Heartbeat: renews the lease; raises `_RunCancelled` when an admin cancelled the run."""

    status = await backend.post_embedding_reindex_progress(
        run_id=claim.run_id,
        lease_token=claim.lease_token,
        total_points=progress.total,
        processed_points=progress.processed,
        failed_points=progress.failed,
    )
    if status == CANCELLED:
        raise _RunCancelled()


async def _copy_collection(
    claim: _Claim,
    backend: BackendJavaClient,
    qdrant: QdrantClient,
    key: str,
    progress: _Progress,
) -> None:
    """The rebuild: every point of the source goes to the target with its dense vectors as
    they are and a `bm25` vector from its payload (`qdrant_store.point_with_bm25`). No embedding
    API, no budget. Points added to or removed from the source meanwhile are reconciled like the
    re-embed does, so the cutover sees the same set of ids."""

    batch_size = max(1, settings.EMBEDDING_REINDEX_BATCH_SIZE)

    async def copy(records: list[models.Record]) -> None:
        points = [
            qdrant_store.point_with_bm25(record, payload_update={"embedding_identity_key": key})
            for record in records
        ]
        qdrant_store.upsert_points(qdrant, claim.target, points)
        progress.processed += len(points)
        await _report(claim, backend, progress)

    offset: qdrant_store.PointOffset = None
    while True:
        records, offset = qdrant_store.scroll_points(
            qdrant, claim.source, limit=batch_size, offset=offset, with_vectors=True
        )
        if records:
            await copy(records)
        if offset is None:
            break

    source_ids = qdrant_store.point_ids(qdrant, claim.source)
    target_ids = qdrant_store.point_ids(qdrant, claim.target)
    missing = sorted(source_ids - target_ids)
    for start in range(0, len(missing), batch_size):
        await copy(
            qdrant_store.retrieve_points(
                qdrant, claim.source, missing[start : start + batch_size], with_vectors=True
            )
        )
    extra = sorted(target_ids - source_ids)
    qdrant_store.delete_points(qdrant, claim.target, extra)
    if missing or extra:
        logger.info(
            "embedding re-index %s: reconciled %d missing and %d extra point(s)",
            claim.run_id,
            len(missing),
            len(extra),
        )


async def _reembed_collection(
    claim: _Claim,
    backend: BackendJavaClient,
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    key: str,
    progress: _Progress,
) -> None:
    redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    budget_tracker = BudgetTracker(redis_client=redis_client)
    recorder = UsageRecorder(
        request_id=str(uuid.uuid4()), purpose="INGEST", budget_tracker=budget_tracker
    )
    batch_size = max(1, settings.EMBEDDING_REINDEX_BATCH_SIZE)
    batches = -(-(progress.total or 0) // batch_size)
    status = "ERROR"
    try:
        reserve_result = await budget_tracker.reserve_request(
            request_id=recorder.request_id,
            purpose="INGEST",
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)) * max(batches, 1),
        )
        if reserve_result != "OK":
            raise RequestBudgetRejectedError("INGEST", reserve_result)

        async def reembed(records: list[models.Record]) -> None:
            await _reembed_batch(
                records, claim, qdrant, embedder, key, recorder, budget_tracker, progress
            )
            await _report(claim, backend, progress)

        offset: qdrant_store.PointOffset = None
        while True:
            records, offset = qdrant_store.scroll_points(
                qdrant, claim.source, limit=batch_size, offset=offset
            )
            if records:
                await reembed(records)
            if offset is None:
                break

        # A job that embedded into the source while it was being scrolled, or a point deleted
        # meanwhile: bring the target in line with the source before the cutover.
        source_ids = qdrant_store.point_ids(qdrant, claim.source)
        target_ids = qdrant_store.point_ids(qdrant, claim.target)
        missing = sorted(source_ids - target_ids)
        for start in range(0, len(missing), batch_size):
            await reembed(
                qdrant_store.retrieve_points(
                    qdrant, claim.source, missing[start : start + batch_size]
                )
            )
        extra = sorted(target_ids - source_ids)
        qdrant_store.delete_points(qdrant, claim.target, extra)
        if missing or extra:
            logger.info(
                "embedding re-index %s: reconciled %d missing and %d extra point(s)",
                claim.run_id,
                len(missing),
                len(extra),
            )
        status = "SUCCESS"
    finally:
        try:
            await recorder.close(status=status)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                logger.debug("embedding re-index: closing the Redis client failed", exc_info=True)


async def _reembed_batch(
    records: list[models.Record],
    claim: _Claim,
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    key: str,
    recorder: UsageRecorder,
    budget_tracker: BudgetTracker,
    progress: _Progress,
) -> None:
    embeddable = [record for record in records if (record.payload or {}).get("content")]
    progress.failed += len(records) - len(embeddable)
    if not embeddable:
        return
    texts = [text for record in embeddable for text in embedding_texts(record.payload or {})]
    vectors = await _embed_with_retry(embedder, texts, recorder, budget_tracker, claim.run_id)
    points = [
        models.PointStruct(
            id=str(record.id),
            vector=qdrant_store.point_vectors(
                vectors[3 * index],
                vectors[3 * index + 1],
                vectors[3 * index + 2],
                bm25=sparse_of_payload(record.payload or {}),
            ),
            payload={**(record.payload or {}), "embedding_identity_key": key},
        )
        for index, record in enumerate(embeddable)
    ]
    qdrant_store.upsert_points(qdrant, claim.target, points)
    progress.processed += len(points)


async def _embed_with_retry(
    embedder: EmbeddingProvider,
    texts: list[str],
    recorder: UsageRecorder,
    budget_tracker: BudgetTracker,
    run_id: str,
) -> list[list[float]]:
    """A long run meets the odd rate limit or timeout: retry a batch with backoff before giving
    up on the whole run."""

    attempts = max(1, settings.EMBEDDING_REINDEX_MAX_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            vectors = await embedder.embed_tracked(texts, recorder, budget_tracker)
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = 2.0**attempt
            logger.warning(
                "embedding re-index %s: batch failed (attempt %d/%d), retrying in %.0fs: %s",
                run_id,
                attempt,
                attempts,
                delay,
                safe_error_message(exc),
            )
            await asyncio.sleep(delay)
            continue
        if len(vectors) != len(texts):
            raise ReindexError(
                f"the embedding API returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors
    raise AssertionError("unreachable")
