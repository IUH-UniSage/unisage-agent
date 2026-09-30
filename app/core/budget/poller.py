"""Periodic `BudgetSnapshot` refresh for the FastAPI process - keeps
`app.core.budget.snapshot`'s process-local cache from going stale for the whole
process lifetime without a restart. The Celery worker side of this same need is
met differently (see `app.worker.celery_app`'s `refresh_budget_snapshot` beat
task) since a worker process has no long-lived event loop of its own to run a
background task on between task invocations.
"""

from __future__ import annotations

import asyncio
import logging

from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.config import settings

logger = logging.getLogger(__name__)


async def _poll_loop(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_budget_snapshot()
        except Exception:
            logger.exception("budget snapshot poll failed - keeping previous snapshot")


def start_budget_snapshot_poller(*, interval: float | None = None) -> asyncio.Task[None]:
    """Starts the periodic refresh loop as a background task on the CURRENT event
    loop. The caller owns cancelling it on shutdown (see `app.main`'s lifespan)."""

    resolved_interval = (
        interval if interval is not None else settings.BUDGET_SNAPSHOT_REFRESH_SECONDS
    )
    return asyncio.create_task(_poll_loop(resolved_interval))
