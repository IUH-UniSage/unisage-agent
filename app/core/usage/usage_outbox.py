"""Redis outbox for usage payloads. `enqueue()` only ever `LPUSH`es - it never calls
`POST /internal/usage-logs` itself, so a Java outage never adds latency (or a
failure) to a Chat response. The Celery drain worker that moves items from here to
Java (`drain_usage_outbox` task) is a separate piece, not implemented by this module.

Same "degrade, don't crash the caller" posture as `app/core/observability/alerting.py`, but the
failure here is logged at ERROR (not WARNING): a lost alert is just silence, a lost
usage payload is unrecoverable cost data for that request.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

OUTBOX_KEY = "usage:outbox"


class _RedisLike(Protocol):
    """Structural subset of `redis.asyncio.Redis` this module actually calls - same
    trick `model_router._RedisLike`/`alerting._RedisLike` use to let tests hand in a
    bare fake instead of a real Redis connection."""

    async def lpush(self, name: str, *values: Any) -> Any: ...

    async def aclose(self) -> Any: ...


async def enqueue_usage_payload(
    payload: dict[str, Any], *, redis_client: _RedisLike | None = None
) -> None:
    """`LPUSH`es `payload` (already a plain JSON-serializable dict) onto the outbox.

    Never raises - a Redis outage must not take Chat down with it. The payload is
    lost if this fails (no in-memory retry queue here; that would just move the
    same "process might die before flushing it" risk somewhere else) - logged at
    ERROR so it's visible. Redis AOF persistence, not application-level retries,
    is what actually prevents loss here.
    """

    body = json.dumps(payload)

    if redis_client is not None:
        try:
            await redis_client.lpush(OUTBOX_KEY, body)
        except Exception:
            logger.error(
                "usage_outbox: failed to enqueue payload for requestId=%s - usage record LOST",
                payload.get("requestId"),
                exc_info=True,
            )
        return

    try:
        conn = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    except Exception:
        logger.error(
            "usage_outbox: could not connect to Redis for requestId=%s - usage record LOST",
            payload.get("requestId"),
            exc_info=True,
        )
        return

    try:
        await conn.lpush(OUTBOX_KEY, body)
    except Exception:
        logger.error(
            "usage_outbox: failed to enqueue payload for requestId=%s - usage record LOST",
            payload.get("requestId"),
            exc_info=True,
        )
    finally:
        await conn.aclose()
