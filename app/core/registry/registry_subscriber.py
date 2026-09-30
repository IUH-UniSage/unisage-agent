"""Hot-reload for `app.core.registry.model_registry`'s cached snapshot (plan.md "Hot-reload
consistency", todo.md Task 8).

Two independent signals feed the same swap logic:

- **Subscribe**: a Redis pub/sub listener on `settings.MODEL_REGISTRY_CHANNEL` — the channel
  `ModelRegistryEventPublisher` (backend-java) publishes to `AFTER_COMMIT`. The message body is
  just the new version as a bare string (`String.valueOf(event.version())`); a `{"version": N}`
  JSON body is tolerated too since that's the shape plan.md's contract text describes, but the
  real Java publisher sends the bare form.
- **Poll**: every `settings.MODEL_REGISTRY_POLL_INTERVAL_SECONDS`, ask Java directly via
  `GET /internal/model-registry/version`. Runs unconditionally, independent of whether the
  pub/sub connection is even up — this is what makes a dropped/never-sent Redis message
  self-heal instead of wedging a worker on a stale snapshot forever.

Both signals funnel into `_reload_if_newer()`, which is the only thing allowed to call
`model_registry.set_current_snapshot()`: it ignores anything `<=` the currently cached
version (stale/out-of-order), and on a fetch/parse failure logs a warning and leaves the old
snapshot in place — never a partial/broken swap. An `asyncio.Lock` shared by both loops
serializes concurrent reload attempts so two rapid triggers don't fetch twice or step on each
other's swap.

Redis itself is only ever a *signal* here — Java's DB + version counter is the source of
truth (plan.md "Redis pub/sub chỉ là tín hiệu"), which is why a lost pub/sub message is a
non-issue, not a bug: the poll loop notices the version gap on its own within one interval.

Wired from two places, matching the two process types this service runs as:

- `app.main`'s lifespan: `start_asyncio_registry_subscriber()` schedules both loops as asyncio
  tasks on the running event loop, returns a handle whose `.stop()` cancels them cleanly on
  shutdown.
- `app.worker.celery_app`'s `worker_process_init` handler: `start_thread_registry_subscriber()`
  spins up a **daemon thread** running its own event loop (Celery's prefork workers are sync
  processes with no asyncio loop of their own) — nothing needs to join it, it dies with the
  worker process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.core.registry.model_registry import (
    get_current_snapshot,
    parse_snapshot,
    set_current_snapshot,
)

logger = logging.getLogger(__name__)

# How long to wait before reconnecting a dropped pub/sub connection - deliberately short and
# unconfigurable: this only affects how quickly the *subscribe* path recovers its socket, never
# correctness (the poll loop is what actually guarantees convergence).
_PUBSUB_RECONNECT_DELAY_SECONDS = 2.0


class RegistryClient(Protocol):
    """Structural type for the two `BackendJavaClient` methods this module calls — lets tests
    hand in a bare stub instead of a real `BackendJavaClient`/`httpx.MockTransport`."""

    async def get_model_registry_version(self) -> int: ...

    async def get_model_registry_snapshot(self) -> dict[str, Any]: ...


def _parse_version_message(raw: bytes | str) -> int | None:
    """Best-effort parse of one pub/sub message body into a version int.

    Returns `None` (never raises) for anything that isn't recognizably a version - an
    unparseable message is logged and dropped by the caller, exactly like a stale version:
    the poll loop is the backstop either way.
    """

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if isinstance(payload, dict) and "version" in payload:
        try:
            return int(payload["version"])
        except (TypeError, ValueError):
            return None
    return None


async def _reload_if_newer(
    new_version: int,
    *,
    client: RegistryClient,
    lock: asyncio.Lock,
    source: str,
) -> bool:
    """Fetch + parse + swap iff `new_version` is newer than the cached snapshot's version.

    Returns whether a swap happened. Never raises - a fetch/parse failure is logged as a
    warning and the previous snapshot (same object, same identity) is left in place.
    """

    current = get_current_snapshot()
    current_version = current.version if current is not None else -1
    if new_version <= current_version:
        logger.debug(
            "registry_subscriber: ignoring stale/out-of-order version=%s (current=%s, source=%s)",
            new_version,
            current_version,
            source,
        )
        return False

    async with lock:
        # Re-check inside the lock: another trigger (the other loop, or an earlier message)
        # may have already swapped us past new_version while we were waiting for it.
        current = get_current_snapshot()
        current_version = current.version if current is not None else -1
        if new_version <= current_version:
            logger.debug(
                "registry_subscriber: version=%s already applied by the time the lock was "
                "acquired (current=%s, source=%s)",
                new_version,
                current_version,
                source,
            )
            return False

        try:
            payload = await client.get_model_registry_snapshot()
            snapshot = parse_snapshot(payload)
        except Exception:
            logger.warning(
                "registry_subscriber: failed to fetch/parse snapshot for version=%s "
                "(source=%s) - keeping current snapshot (version=%s)",
                new_version,
                source,
                current_version,
                exc_info=True,
            )
            return False

        set_current_snapshot(snapshot)
        logger.info(
            "registry_subscriber: swapped model registry snapshot version=%s -> %s (source=%s)",
            current_version,
            snapshot.version,
            source,
        )
        return True


async def _consume_pubsub_messages(
    pubsub: Any, *, client: RegistryClient, lock: asyncio.Lock
) -> None:
    """Drains `pubsub.listen()` for as long as the connection lives - one `_reload_if_newer`
    call per `type == "message"` frame, everything else (subscribe confirmations, etc.) is
    skipped."""

    async for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        version = _parse_version_message(message["data"])
        if version is None:
            logger.warning(
                "registry_subscriber: unparseable message on %s: %r",
                settings.MODEL_REGISTRY_CHANNEL,
                message.get("data"),
            )
            continue
        await _reload_if_newer(version, client=client, lock=lock, source="pubsub")


async def _pubsub_listen_loop(
    *,
    client: RegistryClient,
    lock: asyncio.Lock,
    channel: str,
    redis_factory: Callable[[], Any],
) -> None:
    """Connect, subscribe, consume forever; reconnect (after a short delay) on any failure -
    a dropped connection here is never fatal to the process, and never a correctness problem
    either (the poll loop keeps converging independently)."""

    while True:
        conn = None
        pubsub = None
        try:
            conn = redis_factory()
            pubsub = conn.pubsub()
            await pubsub.subscribe(channel)
            await _consume_pubsub_messages(pubsub, client=client, lock=lock)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "registry_subscriber: pub/sub connection to %s lost/failed, retrying in %ss",
                channel,
                _PUBSUB_RECONNECT_DELAY_SECONDS,
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
        await asyncio.sleep(_PUBSUB_RECONNECT_DELAY_SECONDS)


async def _poll_once(*, client: RegistryClient, lock: asyncio.Lock) -> None:
    """One poll iteration - fetch the version, and if it's newer than what's cached, swap
    (and log loudly: this path firing at all means the subscribe path should have already
    caught this version, so it's a diagnostic signal that pub/sub messages are being missed,
    not a routine occurrence)."""

    try:
        new_version = await client.get_model_registry_version()
    except Exception:
        logger.warning("registry_subscriber: poll failed to fetch current version", exc_info=True)
        return

    current = get_current_snapshot()
    current_version = current.version if current is not None else -1
    if new_version <= current_version:
        return

    logger.warning(
        "registry_subscriber: poll detected version=%s newer than cached=%s - the pub/sub "
        "subscriber should have already caught this; a message may have been missed",
        new_version,
        current_version,
    )
    await _reload_if_newer(new_version, client=client, lock=lock, source="poll")


async def _poll_loop(*, client: RegistryClient, lock: asyncio.Lock, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        await _poll_once(client=client, lock=lock)


class RegistrySubscriberHandle:
    """Handle returned to the FastAPI lifespan - `.stop()` cancels both asyncio tasks and
    waits for them to actually finish, so shutdown never leaves a dangling task warning."""

    def __init__(self, tasks: list[asyncio.Task[None]]) -> None:
        self._tasks = tasks

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


def start_asyncio_registry_subscriber(
    *,
    client: RegistryClient | None = None,
    poll_interval: float | None = None,
) -> RegistrySubscriberHandle:
    """Call once from `app.main`'s lifespan, after `init_model_registry()`. No-op (returns a
    handle with no tasks) when `MODEL_REGISTRY_ENABLED` is false, matching
    `init_model_registry()`'s own no-op behavior for the same flag."""

    if not settings.MODEL_REGISTRY_ENABLED:
        return RegistrySubscriberHandle([])

    from app.integrations.backend_java_client import BackendJavaClient

    resolved_client: RegistryClient = client if client is not None else BackendJavaClient()
    lock = asyncio.Lock()
    interval = (
        poll_interval
        if poll_interval is not None
        else settings.MODEL_REGISTRY_POLL_INTERVAL_SECONDS
    )
    channel = settings.MODEL_REGISTRY_CHANNEL
    redis_url = settings.REDIS_URL

    tasks = [
        asyncio.create_task(
            _pubsub_listen_loop(
                client=resolved_client,
                lock=lock,
                channel=channel,
                redis_factory=lambda: redis_asyncio.Redis.from_url(redis_url),
            ),
            name="registry-subscriber-pubsub",
        ),
        asyncio.create_task(
            _poll_loop(client=resolved_client, lock=lock, interval=interval),
            name="registry-subscriber-poll",
        ),
    ]
    return RegistrySubscriberHandle(tasks)


async def _run_subscriber_forever(*, client: RegistryClient, poll_interval: float) -> None:
    lock = asyncio.Lock()
    await asyncio.gather(
        _pubsub_listen_loop(
            client=client,
            lock=lock,
            channel=settings.MODEL_REGISTRY_CHANNEL,
            redis_factory=lambda: redis_asyncio.Redis.from_url(settings.REDIS_URL),
        ),
        _poll_loop(client=client, lock=lock, interval=poll_interval),
    )


def start_thread_registry_subscriber(
    *,
    client: RegistryClient | None = None,
    poll_interval: float | None = None,
) -> threading.Thread | None:
    """Call once from Celery's `worker_process_init` handler, after `init_model_registry()`.

    Returns `None` (and starts nothing) when `MODEL_REGISTRY_ENABLED` is false. Otherwise
    starts a **daemon** thread running its own asyncio event loop with both loops on it -
    Celery's prefork worker process is sync and has no event loop of its own, and daemon means
    the harness/production process doesn't need to explicitly join or cancel it on shutdown;
    it dies with the worker process like every other Celery worker resource.
    """

    if not settings.MODEL_REGISTRY_ENABLED:
        return None

    from app.integrations.backend_java_client import BackendJavaClient

    resolved_client: RegistryClient = client if client is not None else BackendJavaClient()
    interval = (
        poll_interval
        if poll_interval is not None
        else settings.MODEL_REGISTRY_POLL_INTERVAL_SECONDS
    )

    def _run() -> None:
        asyncio.run(_run_subscriber_forever(client=resolved_client, poll_interval=interval))

    thread = threading.Thread(target=_run, name="registry-subscriber", daemon=True)
    thread.start()
    return thread
