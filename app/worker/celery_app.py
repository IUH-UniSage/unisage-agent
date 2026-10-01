import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import redis
from celery import Celery
from celery.signals import worker_process_init
from kombu import Queue
from redis import asyncio as redis_asyncio

from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.budget.tracker import BudgetTracker, RequestBudgetRejectedError
from app.core.config import settings
from app.core.errors.error_codes import ErrorCode
from app.core.errors.llm_error_classifier import (
    EmbeddingProviderError,
    ErrorType,
    classify_llm_error,
)
from app.core.errors.llm_failure import describe_llm_failure, is_model_failure
from app.core.llm.provider_models import UnsupportedProviderError
from app.core.observability.alerting import alert_credential_failure
from app.core.observability.events import publish_ingestion_event
from app.core.observability.logging_config import configure_logging
from app.core.pricing.snapshot import refresh_pricing_snapshot
from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
from app.core.registry.model_registry import (
    ModelRegistryError,
    get_current_snapshot,
    init_model_registry,
)
from app.core.registry.model_router import NoAvailableCredentialError, NoBudgetAvailableError
from app.core.registry.registry_subscriber import start_thread_registry_subscriber
from app.core.security.redaction import safe_error_message
from app.integrations.backend_java_client import BackendJavaClient
from app.rag.embeddings.provider import EmbeddingProvider, build_embedder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.core.usage.usage_recorder import UsageRecorder
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk
from app.worker.budget_reconciliation_tasks import (
    reconcile_budget_committed_once,
    release_expired_reservations_once,
)
from app.worker.usage_outbox_tasks import drain_usage_outbox_once
from app.worker.verification_subscriber import start_thread_verification_subscriber
from app.worker.verification_tasks import run_verification_batch, try_acquire_verification_lock

# Same redaction filter + noisy-logger silencing as the API process (see
# app/main.py) - Celery worker/beat is a separate process that never imports
# app.main, so it needs its own call.
configure_logging()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "unisage_ingestion",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# Keep task results well past a wizard tab's lifetime so the client's
# reconciliation sweep can still read a terminal state days later.
celery_app.conf.result_expires = 60 * 60 * 24 * 7
# Every task this service declares runs in a queue named
# `<CELERY_QUEUE_PREFIX>-<queue>` - lets the integration harness give each test
# run its own queue namespace and `celery purge -Q <that queue>` without ever
# touching another run's queue or (since purge is queue-scoped, not DB-scoped)
# another Redis DB's keys. Single default queue today; task_routes can split
# further later without changing the prefix mechanism.
_default_queue = f"{settings.CELERY_QUEUE_PREFIX}-default"
celery_app.conf.task_default_queue = _default_queue
celery_app.conf.task_queues = (Queue(_default_queue, routing_key=_default_queue),)

# Beat heartbeat: exists so the integration harness has something observable to
# assert "Beat is actually ticking, not just started" - writes the current time to a
# registry-namespaced (`mr:`) Redis key on DB 0, not the Celery broker/backend DB, and
# never touches task args/results (Celery tasks in this feature never carry
# credentials as arguments).
BEAT_HEARTBEAT_REDIS_KEY = "mr:beat:last_tick"
celery_app.conf.beat_schedule = {
    "model-registry-beat-heartbeat": {
        "task": "beat_heartbeat",
        "schedule": settings.CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS,
    },
    # Verify-before-active claim loop - the backstop that guarantees a queued job
    # eventually gets claimed even if the verification-requested pub/sub nudge (see
    # worker_process_init below) is missed entirely.
    "model-registry-verify-pending": {
        "task": "verify_pending_credentials",
        "schedule": settings.MODEL_REGISTRY_VERIFICATION_INTERVAL_SECONDS,
    },
    # Usage outbox drain - moves UsageRecorder payloads from Redis to backend-java.
    # Short interval on purpose: the outbox is the only thing standing between a
    # Chat response and its cost ever reaching Java.
    "usage-outbox-drain": {
        "task": "drain_usage_outbox",
        "schedule": settings.USAGE_OUTBOX_DRAIN_INTERVAL_SECONDS,
    },
    # Refreshes THIS worker process's own BudgetSnapshot cache - a prefork worker
    # has no long-lived event loop to run the FastAPI process's background poller
    # on, so Beat is what keeps it from going stale between worker restarts.
    "budget-snapshot-refresh": {
        "task": "refresh_budget_snapshot",
        "schedule": settings.BUDGET_SNAPSHOT_REFRESH_SECONDS,
    },
    "pricing-snapshot-refresh": {
        "task": "refresh_pricing_snapshot",
        "schedule": settings.MODEL_PRICING_SNAPSHOT_REFRESH_SECONDS,
    },
    "budget-release-expired-reservations": {
        "task": "release_expired_reservations",
        "schedule": 60.0,
    },
    "budget-reconcile-committed": {
        "task": "reconcile_budget_committed",
        "schedule": 60.0 * 60.0,
    },
}


@worker_process_init.connect
def _load_model_registry_on_worker_start(**kwargs: Any) -> None:
    """Mirrors `app.main`'s lifespan load — each prefork worker process gets its own
    snapshot, since it doesn't share memory with gunicorn
    workers or other worker processes. No-op when `MODEL_REGISTRY_ENABLED=false`; when true, an
    uncaught `ModelRegistryError` here is deliberately fatal (Celery aborts the worker process
    that raised out of a bootstep signal), same fail-loud contract as the FastAPI side.
    """

    del kwargs
    asyncio.run(init_model_registry())
    if settings.MODEL_REGISTRY_ENABLED:
        asyncio.run(refresh_budget_snapshot())
        asyncio.run(refresh_pricing_snapshot())
    # Same hot-reload as the FastAPI side, but as a daemon thread running its own
    # event loop - this prefork worker process has no asyncio loop of its own to schedule
    # tasks on. No-op when MODEL_REGISTRY_ENABLED=false.
    start_thread_registry_subscriber()
    # Verify-before-active: wakes the claim loop immediately on a verification-requested
    # message instead of waiting for the next Beat tick. Only the Celery worker process needs
    # this - the FastAPI process never runs verification. No-op when MODEL_REGISTRY_ENABLED=false.
    start_thread_verification_subscriber()


@celery_app.task(name="beat_heartbeat", ignore_result=True)
def beat_heartbeat() -> None:
    """Writes the current time to Redis - see `BEAT_HEARTBEAT_REDIS_KEY` above."""

    try:
        client = redis.Redis.from_url(settings.REDIS_URL)
        client.set(BEAT_HEARTBEAT_REDIS_KEY, str(time.time()))
        client.close()
    except Exception:
        logger.exception("Beat heartbeat failed to write to Redis")


@celery_app.task(name="verify_pending_credentials", ignore_result=True)
def verify_pending_credentials() -> None:
    """Claims and verifies pending model-registry credentials - triggered by Beat on
    a schedule and by an immediate wake-up on the
    verification-requested channel (see `worker_process_init` above), always through this
    same task so the Redis lock below is the only thing that needs to serialize them.

    Deliberately takes **no arguments** and returns nothing: the claimed candidate
    credentials (plaintext API keys) live only in local variables inside
    `run_verification_batch()`'s call stack for the duration of this one run, and never cross
    the Celery broker or result backend.
    """

    if not try_acquire_verification_lock():
        logger.info("verify_pending_credentials: another run already holds the lock, skipping")
        return
    try:
        asyncio.run(run_verification_batch())
    except Exception:
        logger.exception("verify_pending_credentials: run_verification_batch failed")


@celery_app.task(name="drain_usage_outbox", ignore_result=True)
def drain_usage_outbox() -> None:
    """Beat-scheduled - see `app.worker.usage_outbox_tasks.drain_usage_outbox_once`
    for the actual logic (kept as a plain function there so tests call it directly)."""

    result = drain_usage_outbox_once()
    if result["sent"] or result["dead"]:
        logger.info("drain_usage_outbox: sent=%d dead=%d", result["sent"], result["dead"])


@celery_app.task(name="refresh_budget_snapshot", ignore_result=True)
def refresh_budget_snapshot_task() -> None:
    """Beat-scheduled - keeps this worker process's own `BudgetSnapshot` cache
    from going stale between restarts (see `app.core.budget.poller` for the
    FastAPI process's equivalent, which uses a background asyncio task instead
    since it has a long-lived event loop Beat doesn't give a worker process).
    No-op when `MODEL_REGISTRY_ENABLED=false` - same as the initial worker-start
    load, there is no live backend-java to fetch a snapshot from in that mode."""

    if settings.MODEL_REGISTRY_ENABLED:
        asyncio.run(refresh_budget_snapshot())


@celery_app.task(name="refresh_pricing_snapshot", ignore_result=True)
def refresh_pricing_snapshot_task() -> None:
    """Beat-scheduled - same role as `refresh_budget_snapshot_task`, for model prices."""

    if settings.MODEL_REGISTRY_ENABLED:
        asyncio.run(refresh_pricing_snapshot())


@celery_app.task(name="release_expired_reservations", ignore_result=True)
def release_expired_reservations() -> None:
    """Beat-scheduled - see
    `app.worker.budget_reconciliation_tasks.release_expired_reservations_once`."""

    release_expired_reservations_once()


@celery_app.task(name="reconcile_budget_committed", ignore_result=True)
def reconcile_budget_committed() -> None:
    """Beat-scheduled - see
    `app.worker.budget_reconciliation_tasks.reconcile_budget_committed_once`."""

    result = reconcile_budget_committed_once()
    if result["reconciled"] or result["skipped"]:
        logger.info(
            "reconcile_budget_committed: reconciled=%d skipped=%d",
            result["reconciled"],
            result["skipped"],
        )


class IngestionJobFailedError(Exception):
    """Terminal failure of an `embed_chunks` job, raised so Celery records the task FAILED.

    `args` are `(message, error_code)` - `message` is already the friendly, client-facing
    reason (see `describe_llm_failure`). Celery's result backend stores the exception as
    its type + args and rebuilds it on read (this module is imported by the API process,
    so the type resolves), which is how `GET /ingestion/jobs/{id}` (`_read_task_progress`)
    shows the same specific reason the live WebSocket frame did.
    """

    def __init__(self, message: str, error_code: int = ErrorCode.EMBEDDING_JOB_FAILED.code) -> None:
        super().__init__(message, error_code)
        self.message = message
        self.error_code = error_code


# Failures that mean the EMBEDDING/EXTRACTION model is unusable for the whole job (not just
# for one chunk) - see `_run_embed_chunks`.
_JOB_FATAL_ERRORS: tuple[type[Exception], ...] = (
    EmbeddingProviderError,
    NoAvailableCredentialError,
    NoBudgetAvailableError,
    RequestBudgetRejectedError,
    ModelRegistryError,
    UnsupportedProviderError,
)


def _chunk_failure_reason(exc: Exception) -> str:
    """A client-safe explanation of one chunk's failure (no provider text)."""

    if is_model_failure(exc):
        return describe_llm_failure(exc, purpose="EXTRACTION").message
    if type(exc).__module__.startswith("qdrant_client"):
        return ErrorCode.VECTOR_STORE_ERROR.message
    return f"Lỗi nội bộ khi xử lý đoạn ({type(exc).__name__})."


def partial_failure_message(results: list[dict[str, Any]], total: int) -> str:
    """The "N/M đoạn nạp liệu thất bại." line plus the first failed chunk's reason, so the client
    learns why, not just how many."""

    failed = [r for r in results if r.get("status") == "FAILED"]
    message = f"{len(failed)}/{total} đoạn nạp liệu thất bại."
    first_reason = next((r.get("reason") for r in failed if r.get("reason")), None)
    if first_reason:
        message += f" Lỗi đầu tiên (đoạn #{failed[0]['chunk_index']}): {first_reason}"
    return message


@celery_app.task(bind=True, name="embed_chunks")
def embed_chunks(
    self: Any,
    document_id: str,
    object_key: str,
    chunks: list[dict[str, Any]],
    department_id: str,
    access_level: int,
    is_public: bool = False,
    category: str = "HOC_VU",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Enrich, embed, and upsert a client-approved chunk list into Qdrant.

    Reports percent-complete via `update_state` (for the client's
    reconciliation sweep) and a Redis `progress` event per chunk (for the
    live `WS /ingestion/events` relay), then a `completed` event on finish.
    One chunk's enrichment/embedding failure is recorded in the returned
    per-chunk results rather than raised, so it doesn't abort the batch -
    UNLESS the failure is `EmbeddingProviderError` (the ACTIVE EMBEDDING
    credential itself is broken, or its identity guard refused it). Embedding
    never auto-fails-over - that kind of
    failure aborts the whole job instead: chunks already upserted stay in
    Qdrant untouched, no further chunk is embedded, no `completed` event is
    published, and this re-raises so Celery records the task as FAILED (the
    client's reconciliation sweep, `_draft_task_progress`, reads that state
    straight off Celery - no separate document-status bookkeeping needed).

    Every embed/enrich call for this whole document shares ONE `UsageRecorder`
    (`purpose="INGEST"`, `document_id=document_id`, `user_id` = whoever called
    `POST /ingestion/embedding`) - one request-level budget reservation covering
    the entire job, not one per chunk/call - so the cost history shows a single
    INGEST record per document ingestion, with every embedding/extraction call
    as its lines (see `UsageRecorder`/`RequestUsageLog`).
    """

    task_id = self.request.id

    def _publish(payload: dict[str, Any]) -> None:
        publish_ingestion_event(
            {
                "task_id": task_id,
                "document_id": document_id,
                "department_id": department_id,
                **payload,
            }
        )

    embedder = build_embedder()
    enricher = MultiRepresentationEnricher()
    try:
        client = qdrant_store.get_client()
        qdrant_store.ensure_collection(client)
    except Exception as exc:
        # Without this the task would die before `_run_embed_chunks` ever publishes a
        # terminal frame - the wizard would wait forever with no reason shown.
        logger.exception("embed_chunks: Qdrant unavailable for document %s", document_id)
        message = ErrorCode.VECTOR_STORE_ERROR.message
        _publish(
            {
                "type": "completed",
                "state": "FAILURE",
                "error_code": ErrorCode.VECTOR_STORE_ERROR.code,
                "reason": "VECTOR_STORE_ERROR",
                "message": message,
            }
        )
        raise IngestionJobFailedError(message, ErrorCode.VECTOR_STORE_ERROR.code) from exc

    def _update_state(*, state: str, meta: dict[str, Any]) -> None:
        # `task_id=` explicitly, not left to `update_state`'s own `self.request.id` default -
        # `self.request` is a thread-local stack that a `ThreadPoolExecutor`-offloaded call
        # below would see as an empty/unpushed context (Celery only pushes it on the thread
        # that actually received the task), which would store the result under task_id=None.
        self.update_state(task_id=task_id, state=state, meta=meta)

    coro = _run_embed_chunks(
        embedder=embedder,
        enricher=enricher,
        qdrant_client=client,
        chunks=chunks,
        document_id=document_id,
        object_key=object_key,
        department_id=department_id,
        access_level=access_level,
        is_public=is_public,
        category=category,
        user_id=user_id,
        update_state=_update_state,
        publish=_publish,
    )
    # The real Celery worker process has no event loop of its own (Celery's sync task
    # machinery), so `asyncio.run()` is the normal path - but `task_always_eager=True`
    # test callers can invoke this from inside a running loop (e.g. an async test client
    # hitting `POST /ingestion/embedding`), where `asyncio.run()` would raise. Same
    # loop-detection fallback as `OpenAIEmbedder.embed()`/`MultiRepresentationEnricher.enrich()`.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _run_embed_chunks(
    *,
    embedder: EmbeddingProvider,
    enricher: MultiRepresentationEnricher,
    qdrant_client: Any,
    chunks: list[dict[str, Any]],
    document_id: str,
    object_key: str,
    department_id: str,
    access_level: int,
    is_public: bool,
    category: str,
    user_id: str | None,
    update_state: Any,
    publish: Any,
) -> dict[str, Any]:
    redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    budget_tracker = BudgetTracker(redis_client=redis_client)
    recorder = UsageRecorder(
        request_id=str(uuid.uuid4()),
        purpose="INGEST",
        document_id=document_id,
        user_id=user_id,
        budget_tracker=budget_tracker,
    )

    total = len(chunks)
    results: list[dict[str, Any]] = []
    overall_status = "SUCCESS"
    try:
        reserve_result = await budget_tracker.reserve_request(
            request_id=recorder.request_id,
            purpose="INGEST",
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD))
            * max(total, 1),
        )
        if reserve_result != "OK":
            raise RequestBudgetRejectedError("INGEST", reserve_result)

        for position, raw_chunk in enumerate(chunks):
            chunk = Chunk.model_validate(raw_chunk)
            try:
                enriched = await enricher.enrich_tracked(chunk, recorder, budget_tracker)
                # Empty summary/questions (the enrichment fallback) would send an
                # empty string to the embeddings API; fall back to the chunk's own
                # content so every point still gets three valid vectors.
                summary_text = enriched.summary or chunk.content
                questions_text = " ".join(enriched.questions) or chunk.content
                content_vector, summary_vector, questions_vector = await embedder.embed_tracked(
                    [chunk.content, summary_text, questions_text], recorder, budget_tracker
                )
                chunk_id = f"{document_id}:{chunk.chunk_index}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
                qdrant_store.upsert_chunk(
                    qdrant_client,
                    qdrant_store.ChunkPoint(
                        point_id=point_id,
                        document_id=document_id,
                        object_key=object_key,
                        chunk_id=chunk_id,
                        content=chunk.content,
                        summary=enriched.summary,
                        questions=enriched.questions,
                        department=department_id,
                        access_level=access_level,
                        is_public=is_public,
                        category=category,
                        region_type=chunk.region_type.value,
                        content_vector=content_vector,
                        summary_vector=summary_vector,
                        questions_vector=questions_vector,
                        source_type=chunk.source_type.value if chunk.source_type else None,
                        block_index=chunk.block_index,
                        heading_path=list(chunk.heading_path),
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        source_locator=(
                            chunk.source_locator.model_dump(mode="json")
                            if chunk.source_locator is not None
                            else None
                        ),
                        column_names=chunk.column_names,
                        has_header=chunk.has_header,
                        header_source=chunk.header_source.value,
                        chunking_version=chunk.chunking_version,
                        structure_confidence=chunk.structure_confidence,
                        parse_warnings=list(chunk.parse_warnings),
                        embedding_identity_key=embedder.identity_key,
                    ),
                )
                results.append({"chunk_index": chunk.chunk_index, "status": "SUCCESS"})
            except _JOB_FATAL_ERRORS:
                # The EMBEDDING/EXTRACTION model itself is unusable (bad key, not
                # configured, quota, budget, identity guard, every credential cooling
                # down...) - never a per-chunk data problem, and every later chunk would
                # fail the same way. Stops the batch; handled once by the outer `except`.
                logger.error(
                    "AI model failure for document %s at chunk %s - aborting the job",
                    document_id,
                    chunk.chunk_index,
                )
                raise
            except Exception as exc:
                logger.exception("Failed to embed chunk %s of %s", chunk.chunk_index, document_id)
                # This dict is the task's return value, which Celery persists to the
                # result backend (`result_expires` above keeps it there for a week) -
                # exactly the kind of DB-like sink a raw exception message must never
                # reach unredacted, since it may embed provider credentials.
                results.append(
                    {
                        "chunk_index": chunk.chunk_index,
                        "status": "FAILED",
                        "error": safe_error_message(exc),
                        "reason": _chunk_failure_reason(exc),
                    }
                )

            percent = round((position + 1) / total * 100)
            update_state(state="PROGRESS", meta={"percent": percent})
            publish({"type": "progress", "percent": percent})

        failed_chunk_count = sum(1 for r in results if r["status"] == "FAILED")
        if failed_chunk_count > 0:
            # Every chunk failure is isolated (see per-chunk try/except above), so the
            # task itself still ends Celery-SUCCESS with percent=100 - but a caller
            # reading only that would see a false "100% done" for a job where some or
            # all chunks never actually got embedded. The terminal event must say so.
            publish(
                {
                    "type": "completed",
                    "state": "FAILURE",
                    "error_code": ErrorCode.EMBEDDING_JOB_FAILED.code,
                    "message": partial_failure_message(results, total),
                    "failed_chunk_count": failed_chunk_count,
                    "total_chunk_count": total,
                }
            )
        else:
            publish({"type": "completed", "state": "SUCCESS"})
        return {
            "percent": 100,
            "results": results,
            "failed_chunk_count": failed_chunk_count,
            "total_chunk_count": total,
        }
    except _JOB_FATAL_ERRORS as exc:
        overall_status = "ERROR"
        failure = describe_llm_failure(exc)
        # One terminal shape - "completed" with state=FAILURE - not a separate "failed"
        # frame: the web wizard's `ingestionEventSchema` only has one terminal variant.
        publish(
            {
                "type": "completed",
                "state": "FAILURE",
                "error_code": failure.error_code.code,
                "reason": failure.reason,
                "message": failure.message,
            }
        )
        if isinstance(exc, EmbeddingProviderError):
            await _report_embedding_provider_failure(exc)
        # Raised (not returned) so Celery records the task FAILED; the message survives
        # the result backend so `GET /ingestion/jobs/{id}` can show the same reason.
        raise IngestionJobFailedError(failure.message, failure.error_code.code) from exc
    finally:
        try:
            await recorder.close(status=overall_status)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                logger.debug("embed_chunks: closing the Redis client failed", exc_info=True)


async def _report_embedding_provider_failure(exc: EmbeddingProviderError) -> None:
    """Best-effort health report to `backend-java`, same shape CHAT/EXTRACTION
    failures already report via `app.core.registry.model_router`. A credential
    identity mismatch is always `PERMANENT` (retrying never fixes a wrong model/provider); any
    other embedding provider failure is classified from its underlying cause the same way
    `model_router` classifies CHAT/EXTRACTION failures.

    A plain `async def` (not wrapped in its own `asyncio.run`) - the caller,
    `_run_embed_chunks`, is already running inside `embed_chunks`'s one outer
    `asyncio.run`, and `asyncio.run` cannot be nested inside a running loop.
    """

    credential = exc.credential
    snapshot = get_current_snapshot()
    if credential is None or snapshot is None:
        logger.warning(
            "embed_chunks: cannot report embedding provider health - no credential/snapshot in "
            "context for this failure"
        )
        return

    if isinstance(exc, EmbeddingIdentityMismatchError):
        error_type = ErrorType.PERMANENT
    else:
        cause = exc.__cause__
        error_type = classify_llm_error(cause) if cause is not None else ErrorType.TRANSIENT

    reason = safe_error_message(exc, credential.api_key)
    # Embedding never auto-fails-over, so this job ends FAILED regardless of whether the
    # underlying cause classifies as TRANSIENT or PERMANENT - unlike CHAT/EXTRACTION, there is
    # no retry-with-a-different-credential path here that could still recover on its own, so
    # the "don't alert on a self-recovering TRANSIENT" exception doesn't apply.
    await alert_credential_failure(
        credential, "EMBEDDING_PROVIDER_FAILURE", reason, purpose="EMBEDDING"
    )

    try:
        await BackendJavaClient().report_health(
            credential_id=credential.id,
            credential_revision=credential.revision,
            snapshot_version=snapshot.version,
            error_type=error_type.value,
            error_code=type(exc.__cause__ or exc).__name__,
            message=reason,
            occurred_at=datetime.now(UTC).isoformat(),
        )
    except Exception:
        # Best-effort - Java not hearing about this failure right now is not a reason to swallow
        # the actual embedding failure this function was called to report.
        logger.warning(
            "embed_chunks: failed to report embedding provider health for credential=%s",
            credential.id,
            exc_info=True,
        )
