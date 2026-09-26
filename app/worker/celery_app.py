import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import redis
from celery import Celery
from celery.signals import worker_process_init
from kombu import Queue

from app.core.config import settings
from app.core.embedding_identity import EmbeddingIdentityMismatchError
from app.core.events import publish_ingestion_event
from app.core.llm_error_classifier import ErrorType, EmbeddingProviderError, classify_llm_error
from app.core.logging_config import configure_logging
from app.core.model_registry import get_current_snapshot, init_model_registry
from app.core.redaction import safe_error_message
from app.core.registry_subscriber import start_thread_registry_subscriber
from app.integrations.backend_java_client import BackendJavaClient
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk

# Same redaction filter + noisy-logger silencing as the API process (see
# app/main.py) - Celery worker/beat is a separate process that never imports
# app.main, so it needs its own call. plan.md "Secret redaction".
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

# Beat heartbeat: the only thing on Beat's schedule until Task 8 (hot-reload
# poll) adds the real verify-poll entry. Exists so the integration harness has
# something observable to assert "Beat is actually ticking, not just started"
# (plan.md Task 0.5's smoke test) - writes the current time to a registry-namespaced
# (`mr:`) Redis key on DB 0, not the Celery broker/backend DB, and never touches
# task args/results (plan.md "Secret redaction" - Celery tasks in this feature
# never carry credentials as arguments).
BEAT_HEARTBEAT_REDIS_KEY = "mr:beat:last_tick"
celery_app.conf.beat_schedule = {
    "model-registry-beat-heartbeat": {
        "task": "beat_heartbeat",
        "schedule": settings.CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS,
    },
}


@worker_process_init.connect
def _load_model_registry_on_worker_start(**kwargs: Any) -> None:
    """Mirrors `app.main`'s lifespan load (plan.md "Cutover khỏi cấu hình .env tĩnh") — each
    prefork worker process gets its own snapshot, since it doesn't share memory with gunicorn
    workers or other worker processes. No-op when `MODEL_REGISTRY_ENABLED=false`; when true, an
    uncaught `ModelRegistryError` here is deliberately fatal (Celery aborts the worker process
    that raised out of a bootstep signal), same fail-loud contract as the FastAPI side.
    """

    del kwargs
    asyncio.run(init_model_registry())
    # Task 8: same hot-reload as the FastAPI side, but as a daemon thread running its own
    # event loop - this prefork worker process has no asyncio loop of its own to schedule
    # tasks on. No-op when MODEL_REGISTRY_ENABLED=false.
    start_thread_registry_subscriber()


@celery_app.task(name="beat_heartbeat", ignore_result=True)
def beat_heartbeat() -> None:
    """Writes the current time to Redis - see `BEAT_HEARTBEAT_REDIS_KEY` above."""

    try:
        client = redis.Redis.from_url(settings.REDIS_URL)
        client.set(BEAT_HEARTBEAT_REDIS_KEY, str(time.time()))
        client.close()
    except Exception:
        logger.exception("Beat heartbeat failed to write to Redis")


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
) -> dict[str, Any]:
    """Enrich, embed, and upsert a client-approved chunk list into Qdrant.

    Reports percent-complete via `update_state` (for the client's
    reconciliation sweep) and a Redis `progress` event per chunk (for the
    live `WS /ingestion/events` relay), then a `completed` event on finish.
    One chunk's enrichment/embedding failure is recorded in the returned
    per-chunk results rather than raised, so it doesn't abort the batch -
    UNLESS the failure is `EmbeddingProviderError` (the ACTIVE EMBEDDING
    credential itself is broken, or its identity guard refused it). Embedding
    never auto-fails-over (plan.md "Embedding identity guard") - that kind of
    failure aborts the whole job instead: chunks already upserted stay in
    Qdrant untouched, no further chunk is embedded, no `completed` event is
    published, and this re-raises so Celery records the task as FAILED (the
    client's reconciliation sweep, `_draft_task_progress`, reads that state
    straight off Celery - no separate document-status bookkeeping needed).
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

    embedder = OpenAIEmbedder()
    enricher = MultiRepresentationEnricher()
    client = qdrant_store.get_client()
    qdrant_store.ensure_collection(client)

    total = len(chunks)
    results: list[dict[str, Any]] = []

    for position, raw_chunk in enumerate(chunks):
        chunk = Chunk.model_validate(raw_chunk)
        try:
            enriched = enricher.enrich(chunk)
            # Empty summary/questions (the enrichment fallback) would send an
            # empty string to the embeddings API; fall back to the chunk's own
            # content so every point still gets three valid vectors.
            summary_text = enriched.summary or chunk.content
            questions_text = " ".join(enriched.questions) or chunk.content
            content_vector, summary_vector, questions_vector = embedder.embed(
                [chunk.content, summary_text, questions_text]
            )
            chunk_id = f"{document_id}:{chunk.chunk_index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
            qdrant_store.upsert_chunk(
                client,
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
        except EmbeddingProviderError as exc:
            # The provider itself (or the identity guard) is broken - never a per-chunk data
            # problem. Stops the batch entirely: report health, tell the client, mark the task
            # FAILED, and let no further chunk get embedded.
            logger.error(
                "Embedding provider failed for document %s at chunk %s: %s",
                document_id,
                chunk.chunk_index,
                exc,
            )
            reason = safe_error_message(exc)
            _publish({"type": "failed", "reason": reason})
            _report_embedding_provider_failure(exc)
            raise
        except Exception as exc:
            logger.exception("Failed to embed chunk %s of %s", chunk.chunk_index, document_id)
            # This dict is the task's return value, which Celery persists to the
            # result backend (`result_expires` above keeps it there for a week) -
            # exactly the kind of DB-like sink plan.md "Secret redaction" warns
            # about, so the raw exception text never goes in unredacted.
            results.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "status": "FAILED",
                    "error": safe_error_message(exc),
                }
            )

        percent = round((position + 1) / total * 100)
        self.update_state(state="PROGRESS", meta={"percent": percent})
        _publish({"type": "progress", "percent": percent})

    _publish({"type": "completed", "state": "SUCCESS"})
    return {"percent": 100, "results": results}


def _report_embedding_provider_failure(exc: EmbeddingProviderError) -> None:
    """Best-effort health report to `backend-java` (plan.md "Internal API contract" endpoint #3),
    same shape CHAT/EXTRACTION failures already report via `app.core.model_router`. A credential
    identity mismatch is always `PERMANENT` (retrying never fixes a wrong model/provider); any
    other embedding provider failure is classified from its underlying cause the same way
    `model_router` classifies CHAT/EXTRACTION failures.
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

    try:
        asyncio.run(
            BackendJavaClient().report_health(
                credential_id=credential.id,
                credential_revision=credential.revision,
                snapshot_version=snapshot.version,
                error_type=error_type.value,
                error_code=type(exc.__cause__ or exc).__name__,
                message=safe_error_message(exc, credential.api_key),
                occurred_at=datetime.now(UTC).isoformat(),
            )
        )
    except Exception:
        # Best-effort - Java not hearing about this failure right now is not a reason to swallow
        # the actual embedding failure this function was called to report.
        logger.warning(
            "embed_chunks: failed to report embedding provider health for credential=%s",
            credential.id,
            exc_info=True,
        )
