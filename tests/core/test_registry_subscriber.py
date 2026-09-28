"""`app.core.registry.registry_subscriber` — Task 8's hot-reload: a pub/sub listener + independent
poll, both funneling into the same atomic swap of `app.core.registry.model_registry`'s cached
snapshot.

No live Redis anywhere here — a hand-rolled fake pub/sub connection (this repo has no
`fakeredis` dependency) plus a bare structural stub satisfying `RegistryClient`'s two async
methods, same "no live backend-java" spirit as `test_model_registry.py`'s
`httpx.MockTransport` use.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

import app.core.registry.model_registry as model_registry
import app.core.registry.registry_subscriber as registry_subscriber
from app.core.registry.model_registry import get_current_snapshot, parse_snapshot
from app.core.registry.registry_subscriber import (
    RegistrySubscriberHandle,
    _consume_pubsub_messages,
    _parse_version_message,
    _poll_once,
    _reload_if_newer,
    start_asyncio_registry_subscriber,
    start_thread_registry_subscriber,
)


def _snapshot_payload(version: int) -> dict[str, Any]:
    return {
        "version": version,
        "generatedAt": "2026-09-25T03:00:00Z",
        "purposes": {"CHAT": [], "EMBEDDING": [], "EXTRACTION": []},
        "embeddingIndexIdentity": None,
    }


class FakeRegistryClient:
    """Structural stand-in for `BackendJavaClient` — only the two methods
    `RegistryClient` (the Protocol `registry_subscriber` type-hints against) needs."""

    def __init__(self) -> None:
        self.fetch_calls: list[int] = []
        self.current_payload: dict[str, Any] | None = None
        self.fetch_delay: float = 0.0
        self.fail_next_fetch = False
        self.version_to_return = 0
        self.version_calls = 0

    async def get_model_registry_snapshot(self) -> dict[str, Any]:
        if self.fetch_delay:
            await asyncio.sleep(self.fetch_delay)
        if self.fail_next_fetch:
            self.fail_next_fetch = False
            raise RuntimeError("simulated backend-java failure")
        assert self.current_payload is not None, "test forgot to set current_payload"
        self.fetch_calls.append(int(self.current_payload["version"]))
        return self.current_payload

    async def get_model_registry_version(self) -> int:
        self.version_calls += 1
        return self.version_to_return


class FakePubSub:
    """Just enough of `redis.asyncio`'s pubsub object for `_consume_pubsub_messages`:
    `listen()` as an async generator over a canned message list, `subscribe`/`aclose` as
    no-ops."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    async def subscribe(self, channel: str) -> None:
        del channel

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        for message in self._messages:
            yield message

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


# ── _parse_version_message ──────────────────────────────────────────────────


def test_parse_version_message_bare_int_string() -> None:
    assert _parse_version_message("42") == 42


def test_parse_version_message_bare_int_bytes() -> None:
    assert _parse_version_message(b"42") == 42


def test_parse_version_message_json_dict_shape() -> None:
    assert _parse_version_message('{"version": 7}') == 7


def test_parse_version_message_garbage_returns_none() -> None:
    assert _parse_version_message("not-a-version") is None
    assert _parse_version_message('{"unrelated": true}') is None
    assert _parse_version_message("") is None


# ── _reload_if_newer: the core swap decision ────────────────────────────────


@pytest.mark.asyncio
async def test_reload_if_newer_swaps_and_fetches_exactly_once() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    client.current_payload = _snapshot_payload(2)

    swapped = await _reload_if_newer(2, client=client, lock=asyncio.Lock(), source="test")

    assert swapped is True
    assert client.fetch_calls == [2]
    current = get_current_snapshot()
    assert current is not None
    assert current.version == 2


@pytest.mark.asyncio
async def test_reload_if_newer_ignores_equal_version_no_fetch() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(5))
    client = FakeRegistryClient()

    swapped = await _reload_if_newer(5, client=client, lock=asyncio.Lock(), source="test")

    assert swapped is False
    assert client.fetch_calls == []


@pytest.mark.asyncio
async def test_reload_if_newer_ignores_stale_out_of_order_version_no_fetch() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(5))
    client = FakeRegistryClient()

    swapped = await _reload_if_newer(3, client=client, lock=asyncio.Lock(), source="test")

    assert swapped is False
    assert client.fetch_calls == []
    current = get_current_snapshot()
    assert current is not None
    assert current.version == 5


@pytest.mark.asyncio
async def test_reload_if_newer_fetch_failure_keeps_old_snapshot_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    old_snapshot = parse_snapshot(_snapshot_payload(1))
    model_registry._current_snapshot = old_snapshot
    client = FakeRegistryClient()
    client.fail_next_fetch = True

    with caplog.at_level(logging.WARNING):
        swapped = await _reload_if_newer(2, client=client, lock=asyncio.Lock(), source="test")

    assert swapped is False
    # `is`, not `==` - proves the old object was never replaced, not just that an equal
    # object took its place.
    assert get_current_snapshot() is old_snapshot
    assert any("failed to fetch/parse snapshot" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_concurrent_reloads_do_not_overlap_fetches() -> None:
    """Two triggers racing for the same newer version share one lock - only one of them
    should actually fetch+swap; the other must see the swap already applied once it gets
    the lock and back off without fetching again."""

    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    client.current_payload = _snapshot_payload(2)
    client.fetch_delay = 0.05  # widen the race window so both coroutines are in flight together
    lock = asyncio.Lock()

    results = await asyncio.gather(
        _reload_if_newer(2, client=client, lock=lock, source="a"),
        _reload_if_newer(2, client=client, lock=lock, source="b"),
    )

    assert sorted(results) == [False, True]
    assert client.fetch_calls == [2]  # exactly one fetch happened, not two
    current = get_current_snapshot()
    assert current is not None
    assert current.version == 2


# ── The single most important invariant: an in-flight reference is unaffected ───


@pytest.mark.asyncio
async def test_in_flight_snapshot_reference_survives_a_swap_elsewhere() -> None:
    """Simulates a request that grabbed the snapshot before a reload happened - it must
    keep seeing the old (frozen) object, unaffected by the swap, proving the swap is a
    reference reassignment and never an in-place mutation."""

    old_snapshot = parse_snapshot(_snapshot_payload(1))
    model_registry._current_snapshot = old_snapshot
    in_flight_reference = get_current_snapshot()  # what an "in-flight request" holds on to

    client = FakeRegistryClient()
    client.current_payload = _snapshot_payload(2)
    swapped = await _reload_if_newer(2, client=client, lock=asyncio.Lock(), source="test")

    assert swapped is True
    new_snapshot = get_current_snapshot()
    assert new_snapshot is not None
    assert new_snapshot.version == 2

    # The reference grabbed earlier is untouched: same object, same (old) data.
    assert in_flight_reference is old_snapshot
    assert in_flight_reference.version == 1
    assert in_flight_reference is not new_snapshot


# ── _poll_once: independent self-heal, and its diagnostic log ───────────────


@pytest.mark.asyncio
async def test_poll_once_catches_version_bump_with_no_pubsub_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    client.version_to_return = 3
    client.current_payload = _snapshot_payload(3)

    with caplog.at_level(logging.WARNING):
        await _poll_once(client=client, lock=asyncio.Lock())

    current = get_current_snapshot()
    assert current is not None
    assert current.version == 3
    assert any("should have already caught" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_poll_once_noop_when_version_not_newer() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(4))
    client = FakeRegistryClient()
    client.version_to_return = 4

    await _poll_once(client=client, lock=asyncio.Lock())

    assert client.fetch_calls == []
    current = get_current_snapshot()
    assert current is not None
    assert current.version == 4


@pytest.mark.asyncio
async def test_poll_once_swallows_version_fetch_failure() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))

    class _BrokenClient:
        async def get_model_registry_version(self) -> int:
            raise RuntimeError("network down")

        async def get_model_registry_snapshot(self) -> dict[str, Any]:
            raise AssertionError("must not be called")

    await _poll_once(client=_BrokenClient(), lock=asyncio.Lock())  # must not raise

    current = get_current_snapshot()
    assert current is not None
    assert current.version == 1


# ── _consume_pubsub_messages: message -> reload wiring ──────────────────────


@pytest.mark.asyncio
async def test_consume_pubsub_messages_triggers_reload_on_message_type() -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    client.current_payload = _snapshot_payload(2)
    messages = [
        {"type": "subscribe", "data": 1},  # subscribe confirmation - must be skipped, not parsed
        {"type": "message", "data": b"2"},
    ]

    await _consume_pubsub_messages(FakePubSub(messages), client=client, lock=asyncio.Lock())

    assert client.fetch_calls == [2]
    current = get_current_snapshot()
    assert current is not None
    assert current.version == 2


@pytest.mark.asyncio
async def test_consume_pubsub_messages_ignores_unparseable_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    messages = [{"type": "message", "data": b"not-a-version"}]

    with caplog.at_level(logging.WARNING):
        await _consume_pubsub_messages(FakePubSub(messages), client=client, lock=asyncio.Lock())

    assert client.fetch_calls == []
    assert any("unparseable message" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_consume_pubsub_messages_two_rapid_messages_same_version_fetch_once() -> None:
    """Two pub/sub frames for the same version arriving back to back (e.g. Java retried the
    publish) must not cause two fetches - the second is stale by the time it's handled."""

    model_registry._current_snapshot = parse_snapshot(_snapshot_payload(1))
    client = FakeRegistryClient()
    client.current_payload = _snapshot_payload(2)
    messages = [
        {"type": "message", "data": b"2"},
        {"type": "message", "data": b"2"},
    ]

    await _consume_pubsub_messages(FakePubSub(messages), client=client, lock=asyncio.Lock())

    assert client.fetch_calls == [2]


# ── Wiring: no-op when the feature flag is off, clean task cancellation ─────


@pytest.mark.asyncio
async def test_start_asyncio_registry_subscriber_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_subscriber.settings, "MODEL_REGISTRY_ENABLED", False)

    handle = start_asyncio_registry_subscriber()

    assert handle._tasks == []
    await handle.stop()  # must not raise on an empty handle


@pytest.mark.asyncio
async def test_start_asyncio_registry_subscriber_schedules_two_tasks_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_subscriber.settings, "MODEL_REGISTRY_ENABLED", True)
    client = FakeRegistryClient()
    client.version_to_return = 0

    handle = start_asyncio_registry_subscriber(client=client, poll_interval=1000)

    try:
        assert len(handle._tasks) == 2
        assert all(not task.done() for task in handle._tasks)
    finally:
        await handle.stop()

    assert all(task.done() for task in handle._tasks)


@pytest.mark.asyncio
async def test_registry_subscriber_handle_stop_cancels_running_tasks() -> None:
    async def _forever() -> None:
        await asyncio.sleep(1000)

    tasks = [asyncio.create_task(_forever()), asyncio.create_task(_forever())]
    handle = RegistrySubscriberHandle(tasks)

    await handle.stop()

    assert all(task.done() for task in tasks)


def test_start_thread_registry_subscriber_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_subscriber.settings, "MODEL_REGISTRY_ENABLED", False)

    assert start_thread_registry_subscriber() is None
