"""Redis pub/sub bridge for ingestion progress/completion events.

The Celery worker (`embed_chunks`) publishes frames synchronously; the
`WS /ingestion/events` relay consumes them asynchronously and fans them out
to browser clients, filtered per connection to the caller's departments.

Delivery is at-most-once - a dropped frame is never resent. The client's
reconciliation sweep (`GET /ingestion/jobs/{document_id}`) is the
authoritative recovery path and does not depend on any frame arriving.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis
import redis.asyncio as redis_asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

INGESTION_EVENTS_CHANNEL = "ingestion-events"


def publish_ingestion_event(frame: dict[str, Any]) -> None:
    """Best-effort synchronous publish, called from the Celery worker.

    A publish failure (Redis down, network blip) is logged and swallowed -
    it must never abort the embedding task, and the client sweep recovers
    the missed state anyway.
    """

    try:
        client = redis.Redis.from_url(settings.REDIS_URL)
        client.publish(INGESTION_EVENTS_CHANNEL, json.dumps(frame))
        client.close()
    except Exception:
        logger.exception("Failed to publish ingestion event %s", frame.get("type"))


async def ingestion_event_stream() -> AsyncIterator[dict[str, Any]]:
    """Yield each frame published to the ingestion-events channel.

    Used by the `WS /ingestion/events` relay. Isolated in its own function
    so tests can substitute a canned stream without a live Redis.
    """

    client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    pubsub = client.pubsub()
    await pubsub.subscribe(INGESTION_EVENTS_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            yield json.loads(message["data"])
    finally:
        await pubsub.aclose()
        await client.aclose()
