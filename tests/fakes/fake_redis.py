"""A tiny in-memory stand-in for `redis.asyncio.Redis` - just the commands the retrieval
cache uses (this repo has no `fakeredis`; same spirit as the hand-rolled fakes in
`tests/core/test_registry_subscriber.py`). Expiry is driven by `clock`, a callable
returning seconds, so a test can move time forward instead of sleeping."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from typing import Any


class FakeRedisStore:
    """Shared state: one store, many `FakeRedis` clients (the cache opens one per call)."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.values: dict[str, bytes] = {}
        self.sets: dict[str, set[bytes]] = {}
        self.expires_at: dict[str, float] = {}
        self.clock = clock or time.monotonic
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = False

    def client(self) -> FakeRedis:
        return FakeRedis(self)

    def expire_sweep(self) -> None:
        now = self.clock()
        for key in [k for k, at in self.expires_at.items() if at <= now]:
            self.values.pop(key, None)
            self.sets.pop(key, None)
            self.expires_at.pop(key, None)

    def ttl(self, key: str) -> int:
        self.expire_sweep()
        if key not in self.values and key not in self.sets:
            return -2
        if key not in self.expires_at:
            return -1
        return max(0, int(self.expires_at[key] - self.clock()))


def _b(value: Any) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


class FakeRedis:
    def __init__(self, store: FakeRedisStore) -> None:
        self._store = store
        self.closed = False

    def _check(self, name: str, *args: Any) -> None:
        self._store.calls.append((name, args))
        if self._store.fail:
            raise ConnectionError("fake redis down")
        self._store.expire_sweep()

    async def get(self, key: str) -> bytes | None:
        self._check("get", key)
        return self._store.values.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._check("set", key, ex, px, nx)
        if nx and (key in self._store.values or key in self._store.sets):
            return None
        self._store.values[key] = _b(value)
        self._store.expires_at.pop(key, None)
        if ex is not None:
            self._store.expires_at[key] = self._store.clock() + ex
        elif px is not None:
            self._store.expires_at[key] = self._store.clock() + px / 1000
        return True

    async def delete(self, *keys: str) -> int:
        self._check("delete", *keys)
        removed = 0
        for key in keys:
            if self._store.values.pop(key, None) is not None:
                removed += 1
            if self._store.sets.pop(key, None) is not None:
                removed += 1
            self._store.expires_at.pop(key, None)
        return removed

    async def exists(self, *keys: str) -> int:
        self._check("exists", *keys)
        return sum(1 for key in keys if key in self._store.values or key in self._store.sets)

    async def incr(self, key: str) -> int:
        self._check("incr", key)
        current = int(self._store.values.get(key, b"0"))
        self._store.values[key] = str(current + 1).encode()
        return current + 1

    async def mget(self, keys: Iterable[str]) -> list[bytes | None]:
        keys = list(keys)
        self._check("mget", *keys)
        return [self._store.values.get(key) for key in keys]

    async def sadd(self, key: str, *members: Any) -> int:
        self._check("sadd", key, *members)
        target = self._store.sets.setdefault(key, set())
        before = len(target)
        target.update(_b(m) for m in members)
        return len(target) - before

    async def smembers(self, key: str) -> AbstractSet[bytes]:
        self._check("smembers", key)
        return set(self._store.sets.get(key, set()))

    async def expire(self, key: str, seconds: int) -> bool:
        self._check("expire", key, seconds)
        if key not in self._store.values and key not in self._store.sets:
            return False
        self._store.expires_at[key] = self._store.clock() + seconds
        return True

    async def ttl(self, key: str) -> int:
        self._check("ttl", key)
        return self._store.ttl(key)

    async def scan_iter(self, match: str = "*", count: int = 100):  # type: ignore[no-untyped-def]
        self._check("scan", match)
        prefix = match.rstrip("*")
        for key in list(self._store.values) + list(self._store.sets):
            if key.startswith(prefix):
                yield key.encode("utf-8")

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def aclose(self) -> None:
        self.closed = True


class FakePipeline:
    """Queues calls, runs them in order on `execute()` - enough for the cache's writes."""

    def __init__(self, client: FakeRedis) -> None:
        self._client = client
        self._queue: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., FakePipeline]:
        def enqueue(*args: Any, **kwargs: Any) -> FakePipeline:
            self._queue.append((name, args, kwargs))
            return self

        return enqueue

    async def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self._queue:
            results.append(await getattr(self._client, name)(*args, **kwargs))
        self._queue.clear()
        return results

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None
