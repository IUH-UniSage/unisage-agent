import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import Any

import redis
from fastapi import APIRouter
from qdrant_client import QdrantClient
from sqlalchemy import text

from app.core.config import settings
from app.database.session import async_session_factory
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Health"])

logger = logging.getLogger(__name__)

# Short enough that a dead dependency can't make this endpoint hang - unisage-backend's
# AgentHealthIndicator (UNISAGE-62) has its own outer timeout, but this one is what actually
# bounds each individual check. Passed to each client's own timeout parameter AND enforced
# again via asyncio.wait_for() at the call site in health_check() - some client libraries
# (seen with qdrant-client's constructor doing its own compatibility handshake) don't fully
# honor a constructor-level timeout, so don't rely on the client's own setting alone.
_CHECK_TIMEOUT_SECONDS = 2


async def _check_database() -> dict[str, Any]:
    start = time.monotonic()
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "up", "response_time_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        logger.warning("Health check: database unreachable: %s", e)
        return {"status": "down", "error": str(e)}


def _check_redis_sync() -> dict[str, Any]:
    start = time.monotonic()
    try:
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=_CHECK_TIMEOUT_SECONDS,
            socket_timeout=_CHECK_TIMEOUT_SECONDS,
        )
        client.ping()
        client.close()
        return {"status": "up", "response_time_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        logger.warning("Health check: redis unreachable: %s", e)
        return {"status": "down", "error": str(e)}


def _check_qdrant_sync() -> dict[str, Any]:
    start = time.monotonic()
    try:
        client = QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=_CHECK_TIMEOUT_SECONDS
        )
        client.get_collections()
        return {"status": "up", "response_time_ms": round((time.monotonic() - start) * 1000)}
    except Exception as e:
        logger.warning("Health check: qdrant unreachable: %s", e)
        return {"status": "down", "error": str(e)}


@router.get("/health", response_model=ApiResponse[dict[str, Any]])
async def health_check() -> ApiResponse[dict[str, Any]]:
    """Aggregate liveness of this service AND everything it actually depends on to answer a
    chat request: its own Postgres DB, Redis (ingestion event pub/sub), and Qdrant (vector
    store for retrieval). A bare "I'm alive" ping was misleading - unisage-backend's
    AgentHealthIndicator (UNISAGE-62) only reads this endpoint's `data.status`, and a process
    that's up but can't reach its vector store can't actually serve a chat request, so it
    should not report "healthy".

    Each dependency check has its own short timeout, enforced twice (the client's own
    timeout parameter, plus an outer `asyncio.wait_for` here in case a library's internal
    timeout doesn't fully cover its own handshake) - a single dead dependency degrades
    `status` to "unhealthy" instead of hanging or failing this endpoint outright.
    """

    async def _bounded(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(coro, timeout=_CHECK_TIMEOUT_SECONDS + 1)
        except TimeoutError:
            return {"status": "down", "error": "timed out"}

    database_check, redis_check, qdrant_check = await asyncio.gather(
        _bounded(_check_database()),
        _bounded(asyncio.to_thread(_check_redis_sync)),
        _bounded(asyncio.to_thread(_check_qdrant_sync)),
    )
    components = {
        "database": database_check,
        "qdrant": qdrant_check,
        "redis": redis_check,
    }
    overall = "healthy" if all(c["status"] == "up" for c in components.values()) else "unhealthy"

    return ApiResponse.success(
        {
            "status": overall,
            "service": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "components": components,
        }
    )
