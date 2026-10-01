"""Beat-scheduled maintenance tasks (see `celery_app.conf.beat_schedule`). Each one is a
thin wrapper: the actual logic lives in a plain function its own module tests directly."""

import asyncio
import logging
import time

import redis

from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.config import settings
from app.core.pricing.snapshot import refresh_pricing_snapshot
from app.worker.budget_reconciliation_tasks import (
    reconcile_budget_committed_once,
    release_expired_reservations_once,
)
from app.worker.celery_app import celery_app
from app.worker.usage_outbox_tasks import drain_usage_outbox_once
from app.worker.verification_tasks import run_verification_batch, try_acquire_verification_lock

logger = logging.getLogger(__name__)

# Beat heartbeat: exists so the integration harness has something observable to
# assert "Beat is actually ticking, not just started" - writes the current time to a
# registry-namespaced (`mr:`) Redis key on DB 0, not the Celery broker/backend DB, and
# never touches task args/results (Celery tasks in this feature never carry
# credentials as arguments).
BEAT_HEARTBEAT_REDIS_KEY = "mr:beat:last_tick"


@celery_app.task(name="beat_heartbeat", ignore_result=True)
def beat_heartbeat() -> None:
    """Writes the current time to Redis - see `BEAT_HEARTBEAT_REDIS_KEY`."""

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
    verification-requested channel (see `app.worker.signals`), always through this
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
