"""Tests for app/worker/usage_outbox_tasks.py (drain side).
Uses a hand-rolled in-memory fake (same spirit as test_chat_stream_errors.py's
_FakeRedis) rather than a real Redis connection - only the small subset of the
list API this module actually calls needs faithful semantics."""

import json
from collections import defaultdict
from typing import Any

import httpx
import pytest

from app.integrations.backend_java_client import BackendJavaClient
from app.worker import usage_outbox_tasks
from app.worker.usage_outbox_tasks import (
    DEAD_KEY,
    OUTBOX_KEY,
    PROCESSING_KEY,
    drain_usage_outbox_once,
    outbox_health,
)


class _FakeRedisLists:
    """In-memory stand-in for the exact list/lock operations
    `drain_usage_outbox_once` calls - list semantics match Redis: LPUSH prepends,
    RPUSH appends, a plain list index 0 is the "left" end."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = defaultdict(list)
        self._locks: set[str] = set()

    def set(self, name: str, _value: Any, *, nx: bool = False, ex: int | None = None) -> bool:
        del ex
        if nx and name in self._locks:
            return False
        self._locks.add(name)
        return True

    def rpoplpush(self, src: str, dst: str) -> str | None:
        if not self._lists[src]:
            return None
        item = self._lists[src].pop()
        self._lists[dst].insert(0, item)
        return item

    def lmove(self, src: str, dst: str, src_end: str, dst_end: str) -> str | None:
        source = self._lists[src]
        if not source:
            return None
        item = source.pop() if src_end == "RIGHT" else source.pop(0)
        if dst_end == "LEFT":
            self._lists[dst].insert(0, item)
        else:
            self._lists[dst].append(item)
        return item

    def lrem(self, key: str, _count: int, value: str) -> int:
        lst = self._lists[key]
        if value in lst:
            lst.remove(value)
            return 1
        return 0

    def lpush(self, key: str, *values: str) -> int:
        for value in values:
            self._lists[key].insert(0, value)
        return len(self._lists[key])

    def rpush(self, key: str, *values: str) -> int:
        self._lists[key].extend(values)
        return len(self._lists[key])

    def llen(self, key: str) -> int:
        return len(self._lists[key])

    def close(self) -> None:
        return None

    # Test helper, not part of the Redis API.
    def contents(self, key: str) -> list[str]:
        return list(self._lists[key])


def _payload(request_id: str) -> dict[str, Any]:
    return {"requestId": request_id, "purpose": "CHAT", "status": "SUCCESS", "lines": []}


def _java_client(handler: Any) -> BackendJavaClient:
    return BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))


def test_drain_sends_every_item_and_removes_them_from_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRedisLists()
    fake.lpush(OUTBOX_KEY, json.dumps(_payload("r1")), json.dumps(_payload("r2")))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "log-1", "duplicate": False})

    monkeypatch.setattr(usage_outbox_tasks, "BackendJavaClient", lambda: _java_client(handler))

    result = drain_usage_outbox_once(redis_client=fake)

    assert result == {"sent": 2, "dead": 0}
    assert fake.contents(OUTBOX_KEY) == []
    assert fake.contents(PROCESSING_KEY) == []


def test_4xx_response_moves_item_to_dead_letter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedisLists()
    fake.lpush(OUTBOX_KEY, json.dumps(_payload("bad-payload")))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "USAGE_LOG_INVALID_PAYLOAD"})

    monkeypatch.setattr(usage_outbox_tasks, "BackendJavaClient", lambda: _java_client(handler))

    result = drain_usage_outbox_once(redis_client=fake)

    assert result == {"sent": 0, "dead": 1}
    assert fake.contents(PROCESSING_KEY) == []
    assert len(fake.contents(DEAD_KEY)) == 1


def test_5xx_response_returns_item_to_outbox_and_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRedisLists()
    fake.lpush(OUTBOX_KEY, json.dumps(_payload("r1")), json.dumps(_payload("r2")))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"code": "SYS_UNCATEGORIZED"})

    monkeypatch.setattr(usage_outbox_tasks, "BackendJavaClient", lambda: _java_client(handler))

    result = drain_usage_outbox_once(redis_client=fake)

    assert result == {"sent": 0, "dead": 0}
    # The failed item goes back to the outbox, the run stops - the second item was
    # never even attempted, still sitting untouched in the outbox too.
    assert fake.contents(PROCESSING_KEY) == []
    assert fake.contents(DEAD_KEY) == []
    assert len(fake.contents(OUTBOX_KEY)) == 2


def test_reclaims_items_stuck_in_processing_from_a_previous_crashed_run() -> None:
    fake = _FakeRedisLists()
    fake.lpush(PROCESSING_KEY, json.dumps(_payload("stuck")))

    # Acquire the lock first so we can observe the reclaim in isolation, then
    # release it by calling the drain with an already-full outbox check.
    assert fake.set(usage_outbox_tasks.LOCK_KEY, "1", nx=True, ex=60)
    usage_outbox_tasks._reclaim_stuck_items(fake)  # type: ignore[arg-type]

    assert fake.contents(PROCESSING_KEY) == []
    assert len(fake.contents(OUTBOX_KEY)) == 1


def test_second_concurrent_drain_skips_when_lock_is_held() -> None:
    fake = _FakeRedisLists()
    fake.lpush(OUTBOX_KEY, json.dumps(_payload("r1")))
    assert fake.set(
        usage_outbox_tasks.LOCK_KEY, "1", nx=True, ex=60
    )  # simulate a run already in progress

    result = drain_usage_outbox_once(redis_client=fake)

    assert result == {"sent": 0, "dead": 0}
    assert len(fake.contents(OUTBOX_KEY)) == 1  # untouched


def test_outbox_health_reports_pending_and_dead_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedisLists()
    fake.lpush(OUTBOX_KEY, "a", "b")
    fake.lpush(PROCESSING_KEY, "c")
    fake.lpush(DEAD_KEY, "d")

    class _FakeRedisClass:
        @staticmethod
        def from_url(_url: str) -> _FakeRedisLists:
            return fake

    class _FakeRedisModule:
        Redis = _FakeRedisClass

    monkeypatch.setattr(usage_outbox_tasks, "redis", _FakeRedisModule())

    result = outbox_health()

    assert result == {"pending": 3, "dead": 1}
