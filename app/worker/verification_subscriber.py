"""Wakes the verify-before-active claim loop (`app.worker.verification_tasks`) immediately
when backend-java publishes to the verification-requested channel, instead of always waiting
for Celery Beat's next tick (Java publishes right after committing a new/superseded job,
Python reacts immediately on that event).

Structurally mirrors `app.core.registry.registry_subscriber`'s reconnect-loop shape, but
simpler: there is no version to compare here - any message on this channel just means "there
may be a job to claim now", so it enqueues the same Celery task Beat already runs on a
schedule. The distributed lock inside `verify_pending_credentials` (see
`app.worker.celery_app`) is what keeps a Beat tick and this wake-up from both running the
claim loop at once, not this module.

Only wired from the Celery worker process (`app.worker.celery_app`'s `worker_process_init`) -
the FastAPI process has no business running verification at all.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.worker.tasks.periodic import verify_pending_credentials

logger = logging.getLogger(__name__)

# Same short, unconfigurable reconnect delay as `app.core.registry.registry_subscriber` - only
# affects how quickly the pub/sub socket recovers, never correctness (Beat's schedule is the
# backstop).
_RECONNECT_DELAY_SECONDS = 2.0


async def _listen_forever(*, channel: str, redis_url: str) -> None:
    while True:
        conn = None
        pubsub = None
        try:
            conn = redis_asyncio.Redis.from_url(redis_url)
            pubsub = conn.pubsub()
            await pubsub.subscribe(channel)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                logger.info(
                    "verification_subscriber: verification-requested received, waking the "
                    "claim loop"
                )
                verify_pending_credentials.delay()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "verification_subscriber: pub/sub connection to %s lost/failed, retrying in %ss",
                channel,
                _RECONNECT_DELAY_SECONDS,
                exc_info=True,
            )
        finally:
            if pubsub is not None:
                try:
                    await pubsub.aclose()
                except Exception:  # pragma: no cover - best-effort cleanup only
                    pass
            if conn is not None:
                try:
                    await conn.aclose()
                except Exception:  # pragma: no cover - best-effort cleanup only
                    pass
        await asyncio.sleep(_RECONNECT_DELAY_SECONDS)


def start_thread_verification_subscriber() -> threading.Thread | None:
    """Call once from Celery's `worker_process_init` handler, alongside
    `app.core.registry.registry_subscriber.start_thread_registry_subscriber`.

    Returns `None` (and starts nothing) when `MODEL_REGISTRY_ENABLED` is false. Otherwise
    starts a **daemon** thread running its own asyncio event loop - same rationale as the
    registry subscriber's thread variant: Celery's prefork worker process is sync, and daemon
    means nothing needs to join or cancel it on shutdown.
    """

    if not settings.MODEL_REGISTRY_ENABLED:
        return None

    channel = settings.MODEL_REGISTRY_VERIFICATION_CHANNEL
    redis_url = settings.REDIS_URL

    def _run() -> None:
        asyncio.run(_listen_forever(channel=channel, redis_url=redis_url))

    thread = threading.Thread(target=_run, name="verification-subscriber", daemon=True)
    thread.start()
    return thread
