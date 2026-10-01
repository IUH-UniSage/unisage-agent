"""The Celery application: broker/backend, queues and the Beat schedule.

Start the worker/beat with `celery -A app.worker.celery_app ...`. Tasks live in
`app.worker.tasks` and the worker-process bootstrap in `app.worker.signals`; both are
loaded through `include` when a worker starts, so this module imports neither (and
neither creates an import cycle back to it).
"""

from celery import Celery
from kombu import Queue

from app.core.config import settings
from app.core.observability.logging_config import configure_logging

# Same redaction filter + noisy-logger silencing as the API process (see
# app/main.py) - Celery worker/beat is a separate process that never imports
# app.main, so it needs its own call.
configure_logging()

celery_app = Celery(
    "unisage_ingestion",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.signals", "app.worker.tasks"],
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

celery_app.conf.beat_schedule = {
    "model-registry-beat-heartbeat": {
        "task": "beat_heartbeat",
        "schedule": settings.CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS,
    },
    # Verify-before-active claim loop - the backstop that guarantees a queued job
    # eventually gets claimed even if the verification-requested pub/sub nudge (see
    # `app.worker.signals`) is missed entirely.
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
