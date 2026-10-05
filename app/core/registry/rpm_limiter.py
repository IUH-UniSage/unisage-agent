"""Proactive per-credential requests-per-minute limit - each credential's `max_rpm` (set on the
Java "Cấu hình AI" page) enforced on this side, so a free-tier key at its provider limit is
skipped before the provider answers 429.

A sliding 60-second window per `(credential_id, credential_revision)`, kept in Redis (`REDIS_URL`,
DB 0, key prefix `mr:rpm:`) as a sorted set of call timestamps so every worker process shares one
count. The check-and-record runs as one Lua script, so two workers can never both take the last
slot. When Redis is unreachable it degrades to a per-process, in-memory window - same "degrade,
don't crash" tradeoff as `app.core.registry.model_router`'s circuit-breaker state: each process
then only sees its own calls.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig

logger = logging.getLogger(__name__)

_KEY_PREFIX = "mr:rpm"
_WINDOW_SECONDS = 60.0

# KEYS[1] = window key; ARGV = now_ms, window_ms, limit, member.
# Returns -1 when the call was recorded, otherwise the milliseconds until the oldest call in the
# window ages out (when the next slot frees up).
_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now, ARGV[4])
  redis.call('PEXPIRE', key, window)
  return -1
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
return math.max(0, tonumber(oldest[2]) + window - now)
"""


class _RedisLike(Protocol):
    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...

    async def aclose(self) -> Any: ...


def _window_key(credential: CredentialConfig) -> str:
    return f"{_KEY_PREFIX}:{credential.id}:{credential.revision}"


class _InMemoryWindows:
    """Per-process fallback used only when Redis is unreachable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, deque[float]] = {}

    def acquire(self, key: str, now: float, window: float, limit: int) -> float | None:
        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and calls[0] <= now - window:
                calls.popleft()
            if len(calls) < limit:
                calls.append(now)
                return None
            return max(0.0, calls[0] + window - now)


class RpmLimiter:
    """Test doubles: pass `redis_client` (anything with `eval`/`aclose`) and/or `clock`
    (seconds, like `time.time`)."""

    def __init__(
        self,
        *,
        redis_client: _RedisLike | None = None,
        clock: Callable[[], float] = time.time,
        window_seconds: float = _WINDOW_SECONDS,
    ) -> None:
        self._injected_redis_client = redis_client
        self._clock = clock
        self._window_seconds = window_seconds
        self._in_memory = _InMemoryWindows()

    async def acquire(self, credential: CredentialConfig) -> float | None:
        """Records one call against `credential`'s window. Returns `None` when the call may go
        ahead, otherwise how many seconds until a slot frees up (nothing is recorded then). A
        credential without a positive `max_rpm` is never limited."""

        limit = credential.max_rpm
        if limit is None or limit <= 0:
            return None
        key = _window_key(credential)
        now = self._clock()
        try:
            wait_ms = await self._acquire_in_redis(key, now, limit)
        except Exception:
            logger.warning(
                "rpm_limiter: Redis unavailable for %s - degrading to in-memory window",
                key,
                exc_info=True,
            )
            return self._in_memory.acquire(key, now, self._window_seconds, limit)
        return None if wait_ms < 0 else wait_ms / 1000.0

    async def _acquire_in_redis(self, key: str, now: float, limit: int) -> int:
        conn = self._injected_redis_client or redis_asyncio.Redis.from_url(settings.REDIS_URL)
        now_ms = int(now * 1000)
        try:
            result = await conn.eval(
                _ACQUIRE_SCRIPT,
                1,
                key,
                now_ms,
                int(self._window_seconds * 1000),
                limit,
                f"{now_ms}-{uuid.uuid4().hex}",
            )
        finally:
            if self._injected_redis_client is None:
                try:
                    await conn.aclose()
                except Exception:  # pragma: no cover - best-effort cleanup only
                    pass
        return int(result)


_default_limiter: RpmLimiter | None = None
_default_limiter_lock = threading.Lock()


def get_default_limiter() -> RpmLimiter:
    """Lazily-constructed, process-wide `RpmLimiter`."""

    global _default_limiter
    if _default_limiter is None:
        with _default_limiter_lock:
            if _default_limiter is None:
                _default_limiter = RpmLimiter()
    return _default_limiter
