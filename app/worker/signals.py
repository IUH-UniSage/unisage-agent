"""Per-worker-process bootstrap, run in every prefork child right after it starts.

Loaded through `celery_app`'s `include`, so the handler is connected before the pool
forks its children.
"""

import asyncio
from typing import Any

from celery.signals import worker_process_init

from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.config import settings
from app.core.pricing.snapshot import refresh_pricing_snapshot
from app.core.registry.model_registry import init_model_registry
from app.core.registry.registry_subscriber import start_thread_registry_subscriber
from app.worker.verification_subscriber import start_thread_verification_subscriber


@worker_process_init.connect
def _load_model_registry_on_worker_start(**kwargs: Any) -> None:
    """Mirrors `app.core.lifespan` load — each prefork worker process gets its own
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
