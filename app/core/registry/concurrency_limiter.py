"""Per-credential cap on in-flight provider calls - each credential's `max_concurrency` (set on
the Java "Cấu hình AI" page) enforced on this side. Z.ai's free GLM models are limited by how
many requests are in flight at once rather than per minute, which `max_rpm` can't express.

Each in-flight call holds a lease in a Redis sorted set per `(credential_id,
credential_revision)` (`REDIS_URL`, DB 0, key prefix `mr:conc:`), scored by the lease's expiry so
a worker that dies mid-call can only pin a slot until the lease runs out. Acquire runs as one Lua
script, so two workers can never both take the last slot. When Redis is unreachable it degrades
to per-process, in-memory leases - same "degrade, don't crash" tradeoff as
`app.core.registry.rpm_limiter`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig

logger = logging.getLogger(__name__)

_KEY_PREFIX = "mr:conc"
# Longer than any single provider call should take (a long streamed answer included); only
# matters when a worker dies without releasing.
_LEASE_SECONDS = 600.0

# KEYS[1] = lease key; ARGV = now_ms, lease_ms, limit, member. Returns 1 when the lease was taken.
_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZCARD', key) < tonumber(ARGV[3]) then
  redis.call('ZADD', key, now + lease, ARGV[4])
  redis.call('PEXPIRE', key, lease)
  return 1
end
return 0
"""


class _RedisLike(Protocol):
    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...

    async def zrem(self, name: str, *values: Any) -> Any: ...

    async def aclose(self) -> Any: ...


def _lease_key(credential: CredentialConfig) -> str:
    return f"{_KEY_PREFIX}:{credential.id}:{credential.revision}"


class _InMemoryLeases:
    """Per-process fallback used only when Redis is unreachable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, float]] = {}

    def acquire(self, key: str, member: str, now: float, lease: float, limit: int) -> bool:
        with self._lock:
            leases = self._leases.setdefault(key, {})
            for stale in [m for m, expiry in leases.items() if expiry <= now]:
                del leases[stale]
            if len(leases) >= limit:
                return False
            leases[member] = now + lease
            return True

    def release(self, key: str, member: str) -> None:
        with self._lock:
            self._leases.get(key, {}).pop(member, None)


@dataclass
class ConcurrencyLease:
    """One held slot; `release()` is idempotent and never raises."""

    limiter: ConcurrencyLimiter | None
    key: str
    member: str
    released: bool = False

    async def release(self) -> None:
        if self.released or self.limiter is None:
            return
        self.released = True
        await self.limiter._release(self.key, self.member)


class ConcurrencyLimiter:
    """Test doubles: pass `redis_client` (anything with `eval`/`zrem`/`aclose`) and/or `clock`
    (seconds, like `time.time`)."""

    def __init__(
        self,
        *,
        redis_client: _RedisLike | None = None,
        clock: Callable[[], float] = time.time,
        lease_seconds: float = _LEASE_SECONDS,
    ) -> None:
        self._injected_redis_client = redis_client
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._in_memory = _InMemoryLeases()

    async def acquire(self, credential: CredentialConfig) -> ConcurrencyLease | None:
        """Takes one in-flight slot for `credential`. Returns the lease to release once the call
        is done, or `None` when every slot is taken. A credential without a positive
        `max_concurrency` always gets a no-op lease."""

        limit = credential.max_concurrency
        key = _lease_key(credential)
        if limit is None or limit <= 0:
            return ConcurrencyLease(limiter=None, key=key, member="")
        member = uuid.uuid4().hex
        now = self._clock()
        try:
            taken = await self._acquire_in_redis(key, member, now, limit)
        except Exception:
            logger.warning(
                "concurrency_limiter: Redis unavailable for %s - degrading to in-memory leases",
                key,
                exc_info=True,
            )
            taken = self._in_memory.acquire(key, member, now, self._lease_seconds, limit)
        return ConcurrencyLease(limiter=self, key=key, member=member) if taken else None

    def _connection(self) -> _RedisLike:
        return self._injected_redis_client or redis_asyncio.Redis.from_url(settings.REDIS_URL)

    async def _close(self, conn: _RedisLike) -> None:
        if self._injected_redis_client is None:
            try:
                await conn.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup only
                pass

    async def _acquire_in_redis(self, key: str, member: str, now: float, limit: int) -> bool:
        conn = self._connection()
        try:
            result = await conn.eval(
                _ACQUIRE_SCRIPT,
                1,
                key,
                int(now * 1000),
                int(self._lease_seconds * 1000),
                limit,
                member,
            )
        finally:
            await self._close(conn)
        return int(result) == 1

    async def _release(self, key: str, member: str) -> None:
        # Always clear the in-memory copy too: the lease may have been taken there during an
        # outage that has since ended.
        self._in_memory.release(key, member)
        conn = self._connection()
        try:
            await conn.zrem(key, member)
        except Exception:
            logger.warning(
                "concurrency_limiter: failed to release %s in Redis - its lease expires on its own",
                key,
                exc_info=True,
            )
        finally:
            await self._close(conn)


_default_limiter: ConcurrencyLimiter | None = None
_default_limiter_lock = threading.Lock()


def get_default_concurrency_limiter() -> ConcurrencyLimiter:
    """Lazily-constructed, process-wide `ConcurrencyLimiter`."""

    global _default_limiter
    if _default_limiter is None:
        with _default_limiter_lock:
            if _default_limiter is None:
                _default_limiter = ConcurrencyLimiter()
    return _default_limiter
