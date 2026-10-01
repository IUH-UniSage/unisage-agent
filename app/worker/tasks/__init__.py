"""Every Celery task of this service. Importing this package registers them all on
`app.worker.celery_app.celery_app` (the worker does so through its `include`)."""

from app.worker.tasks.ingestion import embed_chunks
from app.worker.tasks.periodic import (
    beat_heartbeat,
    drain_usage_outbox,
    reconcile_budget_committed,
    refresh_budget_snapshot_task,
    refresh_pricing_snapshot_task,
    release_expired_reservations,
    verify_pending_credentials,
)

__all__ = [
    "beat_heartbeat",
    "drain_usage_outbox",
    "embed_chunks",
    "reconcile_budget_committed",
    "refresh_budget_snapshot_task",
    "refresh_pricing_snapshot_task",
    "release_expired_reservations",
    "verify_pending_credentials",
]
