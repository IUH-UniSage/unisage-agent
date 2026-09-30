"""Periodic `PricingSnapshot` refresh for the FastAPI process. Celery workers refresh through the
`refresh_pricing_snapshot` beat task instead - they have no long-lived event loop."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.pricing.snapshot import refresh_pricing_snapshot

logger = logging.getLogger(__name__)


async def _poll_loop(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_pricing_snapshot()
        except Exception:
            logger.exception("pricing snapshot poll failed - keeping previous snapshot")


def start_pricing_snapshot_poller(*, interval: float | None = None) -> asyncio.Task[None]:
    """The caller owns cancelling the task on shutdown."""

    resolved_interval = (
        interval if interval is not None else settings.MODEL_PRICING_SNAPSHOT_REFRESH_SECONDS
    )
    return asyncio.create_task(_poll_loop(resolved_interval))
