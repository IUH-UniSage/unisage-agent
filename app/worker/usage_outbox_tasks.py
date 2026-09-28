"""Celery task draining `app.core.usage_outbox`'s Redis outbox to backend-java -
Cost Tracking plan.md Task 7.

Runs on Beat every `USAGE_OUTBOX_DRAIN_INTERVAL_SECONDS` (see `celery_app.py`'s
`beat_schedule`). Uses a `SET NX EX` lock (same idiom as
`verification_tasks.try_acquire_verification_lock`), but degrades differently on
Redis failure: verification's lock degrades to "proceed anyway" because
correctness there rests on Java's own fencing, not the lock; this lock degrades
to "skip this run" instead, because a lock-less run's reclaim step (moving
everything stuck in `usage:outbox:processing` back onto the main outbox) could
race two drainers into each reclaiming a different half inconsistently - `LMOVE`
itself is atomic and safe under concurrent drainers, reclaim is not.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis

from app.core.config import settings
from app.core.usage_outbox import OUTBOX_KEY
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
)

logger = logging.getLogger(__name__)

PROCESSING_KEY = "usage:outbox:processing"
DEAD_KEY = "usage:outbox:dead"
LOCK_KEY = "usage:outbox:lock"
LOCK_TTL_SECONDS = 60
DEFAULT_MAX_ITEMS_PER_RUN = 500


def _try_acquire_lock(conn: redis.Redis) -> bool:
    try:
        return bool(conn.set(LOCK_KEY, "1", nx=True, ex=LOCK_TTL_SECONDS))
    except Exception:
        logger.warning(
            "usage_outbox drain: Redis lock unavailable, skipping this run", exc_info=True
        )
        return False


def _reclaim_stuck_items(conn: redis.Redis) -> None:
    """Moves everything left in `usage:outbox:processing` (a worker died mid-send
    on a prior run) back onto the main outbox, before this run claims anything new."""

    while conn.rpoplpush(PROCESSING_KEY, OUTBOX_KEY) is not None:
        pass


def drain_usage_outbox_once(
    *, max_items: int = DEFAULT_MAX_ITEMS_PER_RUN, redis_client: redis.Redis | None = None
) -> dict[str, int]:
    """Drains up to `max_items` from the outbox. A plain function, not the Celery
    task itself, so tests can call it directly without going through Celery's task
    machinery. Returns `{"sent": N, "dead": N}` for the health/metric endpoint.

    `redis_client`, when given (tests only - production always builds its own from
    `settings.REDIS_URL`), must support the same subset of `redis.Redis`'s sync API
    this function calls: `set`, `rpoplpush`, `lmove`, `lrem`, `lpush`, `rpush`,
    `close`."""

    owns_connection = redis_client is None
    if redis_client is not None:
        conn = redis_client
    else:
        try:
            conn = redis.Redis.from_url(settings.REDIS_URL)
        except Exception:
            logger.warning(
                "usage_outbox drain: Redis unavailable, skipping this run", exc_info=True
            )
            return {"sent": 0, "dead": 0}

    sent = 0
    dead = 0
    try:
        if not _try_acquire_lock(conn):
            return {"sent": 0, "dead": 0}

        _reclaim_stuck_items(conn)

        client = BackendJavaClient()
        for _ in range(max_items):
            item = conn.lmove(OUTBOX_KEY, PROCESSING_KEY, "RIGHT", "LEFT")
            if item is None:
                break

            payload: dict[str, Any] = json.loads(item)
            try:
                asyncio.run(client.ingest_usage_log(payload))
                conn.lrem(PROCESSING_KEY, 1, item)
                sent += 1
            except BackendJavaHTTPError as exc:
                if 400 <= exc.status_code < 500:
                    conn.lrem(PROCESSING_KEY, 1, item)
                    conn.lpush(DEAD_KEY, item)
                    dead += 1
                    logger.error(
                        "usage_outbox drain: requestId=%s rejected by Java (HTTP %s) - "
                        "moved to dead-letter",
                        payload.get("requestId"),
                        exc.status_code,
                    )
                else:
                    # 5xx - Java itself is unhealthy, not this one payload. Return it to the
                    # outbox immediately (plan.md: "lỗi mạng/5xx -> trả lại outbox") and stop
                    # this run rather than hammering a struggling Java with the rest of the batch.
                    conn.lrem(PROCESSING_KEY, 1, item)
                    conn.rpush(OUTBOX_KEY, item)
                    logger.warning(
                        "usage_outbox drain: requestId=%s got HTTP %s from Java - "
                        "returned to outbox, retrying next run",
                        payload.get("requestId"),
                        exc.status_code,
                    )
                    break
            except BackendJavaConnectionError:
                conn.lrem(PROCESSING_KEY, 1, item)
                conn.rpush(OUTBOX_KEY, item)
                logger.warning(
                    "usage_outbox drain: Java unreachable - returned requestId=%s to outbox, "
                    "stopping this run",
                    payload.get("requestId"),
                    exc_info=True,
                )
                break
    finally:
        if owns_connection:
            conn.close()

    return {"sent": sent, "dead": dead}


def outbox_health() -> dict[str, int]:
    """`{"pending": N, "dead": N}` for `GET /api/v1/health` (Task 11b, not yet wired there -
    this function is the piece that endpoint will call)."""

    try:
        conn: redis.Redis = redis.Redis.from_url(settings.REDIS_URL)
    except Exception:
        logger.warning("usage_outbox health: Redis unavailable", exc_info=True)
        return {"pending": 0, "dead": 0}

    try:
        pending = conn.llen(OUTBOX_KEY) + conn.llen(PROCESSING_KEY)
        dead = conn.llen(DEAD_KEY)
        return {"pending": pending, "dead": dead}
    except Exception:
        logger.warning("usage_outbox health: Redis error reading queue lengths", exc_info=True)
        return {"pending": 0, "dead": 0}
    finally:
        conn.close()
