"""The `embed_chunks` ingest task: enrich (EXTRACTION), embed (EMBEDDING) and upsert a
document's approved chunks into Qdrant, publishing progress for the ingest wizard."""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker, RequestBudgetRejectedError
from app.core.config import settings
from app.core.errors.error_codes import ErrorCode
from app.core.errors.llm_error_classifier import ErrorType, classify_llm_error
from app.core.errors.llm_failure import describe_llm_failure
from app.core.errors.provider_errors import EmbeddingProviderError
from app.core.observability.alerting import alert_credential_failure
from app.core.observability.events import publish_ingestion_event
from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import get_current_snapshot
from app.core.security.redaction import safe_error_message
from app.core.usage.usage_recorder import UsageRecorder
from app.integrations.backend_java_client import BackendJavaClient
from app.rag.embeddings.provider import EmbeddingProvider, build_embedder
from app.rag.enrichment.multi_representation import EnrichedChunk, MultiRepresentationEnricher
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk
from app.worker.celery_app import celery_app
from app.worker.embedding_job_errors import (
    JOB_FATAL_ERRORS,
    IngestionJobFailedError,
    chunk_failure_reason,
    partial_failure_message,
)

logger = logging.getLogger(__name__)


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


async def _pause_between_extractions(last_started: float | None) -> float:
    """Sleep out what is left of `INGEST_EXTRACTION_MIN_INTERVAL_SECONDS`, counted from when
    the previous extraction started (so a slow call is not paused again). Returns the start
    time of the extraction that is about to run."""

    interval = settings.INGEST_EXTRACTION_MIN_INTERVAL_SECONDS
    if last_started is not None and interval > 0:
        remaining = interval - (time.monotonic() - last_started)
        if remaining > 0:
            await asyncio.sleep(remaining)
    return time.monotonic()


def _is_temporary_exhaustion(exc: NoAvailableCredentialError) -> bool:
    """True when waiting can help: the last provider failure was transient, or credentials are
    suspended with no failure to judge by (cooldowns from an earlier chunk). A permanent
    failure, or no credential configured at all, never recovers by waiting."""

    if exc.last_error is not None:
        return (
            isinstance(exc.last_error, Exception)
            and classify_llm_error(exc.last_error) is ErrorType.TRANSIENT
        )
    return bool(exc.suspension_reasons)


async def _enrich_waiting_for_credentials(
    enricher: MultiRepresentationEnricher,
    chunk: Chunk,
    recorder: UsageRecorder,
    budget_tracker: BudgetTracker,
) -> EnrichedChunk:
    """`enrich_tracked`, but while every EXTRACTION credential is cooling down it waits
    `INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS` and retries the same chunk, at most
    `INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS` times before letting the error fail the job."""

    waits = 0
    while True:
        try:
            return await enricher.enrich_tracked(chunk, recorder, budget_tracker)
        except NoAvailableCredentialError as exc:
            if waits >= settings.INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS or not (
                _is_temporary_exhaustion(exc)
            ):
                raise
            waits += 1
            logger.warning(
                "All EXTRACTION credentials are cooling down at chunk %s - waiting %.0fs (%d/%d)",
                chunk.chunk_index,
                settings.INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS,
                waits,
                settings.INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS,
            )
            await asyncio.sleep(settings.INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS)


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
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)) * max(total, 1),
        )
        if reserve_result != "OK":
            raise RequestBudgetRejectedError("INGEST", reserve_result)

        last_extraction_started: float | None = None
        for position, raw_chunk in enumerate(chunks):
            chunk = Chunk.model_validate(raw_chunk)
            try:
                last_extraction_started = await _pause_between_extractions(last_extraction_started)
                enriched = await _enrich_waiting_for_credentials(
                    enricher, chunk, recorder, budget_tracker
                )
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
            except JOB_FATAL_ERRORS:
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
                        "reason": chunk_failure_reason(exc),
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
    except JOB_FATAL_ERRORS as exc:
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
