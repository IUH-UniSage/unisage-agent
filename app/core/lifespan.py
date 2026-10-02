"""FastAPI startup/shutdown: load the model registry, start its hot-reload subscriber
and the budget/pricing snapshot pollers, and stop them again on shutdown."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.budget.poller import start_budget_snapshot_poller
from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.config import settings
from app.core.pricing.poller import start_pricing_snapshot_poller
from app.core.pricing.snapshot import refresh_pricing_snapshot
from app.core.registry.model_registry import init_model_registry
from app.core.registry.registry_subscriber import start_asyncio_registry_subscriber

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log application startup and shutdown boundaries."""

    del app
    logger.info("Starting %s in [%s] mode", settings.APP_NAME, settings.APP_ENV)
    # One-time load of the model registry snapshot from backend-java. No-op when
    # MODEL_REGISTRY_ENABLED=false; when true, raises (and is deliberately left
    # uncaught, failing startup) if there is no ACTIVE CHAT credential.
    await init_model_registry()
    # Hot-reload the cached snapshot without a restart - subscribes to Java's
    # after-commit pub/sub signal and independently polls /version as a self-healing
    # fallback. No-op when the flag above is off.
    subscriber = start_asyncio_registry_subscriber()
    # Same soft-limit posture as the reservation itself: an empty/never-loaded
    # snapshot means no budgets are enforced, fail-open by absence - never fatal to
    # startup even if Java is unreachable right now. Gated on the same flag as the
    # model registry above - a process running on static .env credentials has no
    # live backend-java to fetch a budget snapshot from either.
    # Prices follow the same gate: without a live backend-java every call is UNPRICED.
    budget_poller: asyncio.Task[None] | None = None
    pricing_poller: asyncio.Task[None] | None = None
    if settings.MODEL_REGISTRY_ENABLED:
        await refresh_budget_snapshot()
        budget_poller = start_budget_snapshot_poller()
        await refresh_pricing_snapshot()
        pricing_poller = start_pricing_snapshot_poller()
    yield
    if budget_poller is not None:
        budget_poller.cancel()
    if pricing_poller is not None:
        pricing_poller.cancel()
    await subscriber.stop()
    logger.info("Shutting down %s", settings.APP_NAME)
