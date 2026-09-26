"""Tests for `app.core.model_router` — todo.md Task 10.

No live Redis, no live `backend-java` anywhere in this file: Redis is a hand-rolled
fake async client (this repo has no `fakeredis` dependency, matching
`test_registry_subscriber.py`'s own hand-rolled pub/sub fake), and `backend-java` is a
bare structural stub recording `report_health()` calls, same spirit as
`test_backend_java_client.py`'s `httpx.MockTransport` use elsewhere.
"""

from __future__ import annotations

from typing import Any

import httpx
import httpx2
import openai
import pytest

import app.core.model_registry as model_registry
from app.core.model_registry import CredentialConfig, ModelRegistrySnapshot, parse_snapshot
from app.core.model_router import ModelRouter, NoAvailableCredentialError, _state_key


# ── fixtures / test doubles ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


def _credential(
    *, id: str, revision: int = 1, priority: int | None = 1, api_key: str = "sk-secret"
) -> CredentialConfig:
    return CredentialConfig(
        id=id,
        revision=revision,
        source_type="CLOUD_API",
        provider="openai",
        model_name="gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        priority=priority,
        max_rpm=None,
        api_key=api_key,
    )


def _set_snapshot(*, version: int, chat: tuple[CredentialConfig, ...]) -> ModelRegistrySnapshot:
    snapshot = ModelRegistrySnapshot(
        version=version,
        generated_at=parse_snapshot(
            {"version": version, "generatedAt": "2026-09-26T00:00:00Z", "purposes": {}}
        ).generated_at,
        purposes={"CHAT": chat},
        embedding_index_identity=None,
    )
    model_registry._current_snapshot = snapshot
    return snapshot


class _Clock:
    """Injectable fake clock — lets cooldown-TTL tests advance time without a real
    `sleep`, and stays deterministic regardless of how slowly the test happens to run."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAsyncRedis:
    """Minimal stand-in for `redis.asyncio.Redis` — only `set`/`exists`/`aclose`, the
    three methods `ModelRouter` calls. TTL is evaluated against an injected `_Clock`
    instead of wall-clock time."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._expiry: dict[str, float] = {}

    async def set(self, name: str, value: Any, *, ex: int | None = None) -> Any:
        self._expiry[name] = self._clock.now + float(ex) if ex is not None else float("inf")
        return True

    async def exists(self, name: str) -> int:
        expiry = self._expiry.get(name)
        if expiry is None:
            return 0
        if self._clock.now >= expiry:
            del self._expiry[name]
            return 0
        return 1

    async def aclose(self) -> None:
        pass


class _RedisDownError(ConnectionError):
    pass


class DownRedis:
    """Simulates Redis being completely unreachable — every call raises."""

    async def set(self, name: str, value: Any, *, ex: int | None = None) -> Any:
        raise _RedisDownError("simulated Redis outage")

    async def exists(self, name: str) -> int:
        raise _RedisDownError("simulated Redis outage")

    async def aclose(self) -> None:
        pass


class FakeBackendClient:
    """Structural stand-in for `BackendJavaClient` — only `report_health()`, the one
    method `ModelRouter.record_failure()` calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_on_report: Exception | None = None

    async def report_health(self, **kwargs: Any) -> dict[str, Any]:
        if self.raise_on_report is not None:
            raise self.raise_on_report
        self.calls.append(kwargs)
        return {"applied": True}


def _connection_error() -> openai.APIConnectionError:
    """A TRANSIENT failure with no `Retry-After` header at all — exercises the default
    cooldown backoff."""

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _rate_limit_error_with_retry_after(seconds: str) -> openai.RateLimitError:
    """A TRANSIENT failure that carries a `Retry-After` header — exercises the
    provider-supplied cooldown."""

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": "rate limited", "type": "rate_limit_error", "code": None}
    response = httpx2.Response(
        429, request=request, json={"error": body}, headers={"Retry-After": seconds}
    )
    return openai.RateLimitError("rate limited", response=response, body=body)


def _permanent_error() -> openai.AuthenticationError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}
    response = httpx2.Response(401, request=request, json={"error": body})
    return openai.AuthenticationError("bad key", response=response, body=body)


def _router(*, clock: _Clock | None = None, redis_client: Any = None) -> tuple[ModelRouter, FakeBackendClient]:
    backend = FakeBackendClient()
    if redis_client is None and clock is not None:
        redis_client = FakeAsyncRedis(clock)
    router = ModelRouter(redis_client=redis_client, backend_client=backend)
    return router, backend


# ── priority selection + TRANSIENT cooldown chain ───────────────────────────


@pytest.mark.asyncio
async def test_picks_highest_priority_credential_first() -> None:
    cred_a = _credential(id="a", priority=2)
    cred_b = _credential(id="b", priority=1)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    router, _backend = _router(clock=_Clock())

    selected = await router.get_next_credential("CHAT")

    assert selected.id == "b"


@pytest.mark.asyncio
async def test_transient_failure_chain_moves_through_credentials_in_priority_order() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    cred_c = _credential(id="c", priority=3)
    _set_snapshot(version=1, chat=(cred_a, cred_b, cred_c))
    clock = _Clock()
    router, backend = _router(clock=clock)

    first = await router.get_next_credential("CHAT")
    assert first.id == "a"
    await router.record_failure(first, _connection_error(), snapshot_version=1)

    second = await router.get_next_credential("CHAT")
    assert second.id == "b"
    await router.record_failure(second, _connection_error(), snapshot_version=1)

    third = await router.get_next_credential("CHAT")
    assert third.id == "c"

    # All three failures reported to Java, none of them PERMANENT.
    assert [call["error_type"] for call in backend.calls] == ["TRANSIENT", "TRANSIENT"]


@pytest.mark.asyncio
async def test_transient_credential_recovers_once_cooldown_ttl_expires() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    clock = _Clock()
    router, _backend = _router(clock=clock)

    await router.record_failure(cred_a, _connection_error(), snapshot_version=1)
    assert (await router.get_next_credential("CHAT")).id == "b"

    # Not enough time has passed yet - "a" still cooling down.
    clock.advance(1.0)
    assert (await router.get_next_credential("CHAT")).id == "b"

    # Advance well past the default cooldown - "a" is usable again, and since it's
    # higher priority than "b" it's picked first again.
    clock.advance(60.0)
    assert (await router.get_next_credential("CHAT")).id == "a"


@pytest.mark.asyncio
async def test_retry_after_header_sets_the_cooldown_ttl() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    clock = _Clock()
    router, _backend = _router(clock=clock)

    await router.record_failure(cred_a, _rate_limit_error_with_retry_after("5"), snapshot_version=1)
    assert (await router.get_next_credential("CHAT")).id == "b"

    clock.advance(4.0)
    assert (await router.get_next_credential("CHAT")).id == "b"  # still within the 5s window

    clock.advance(2.0)
    assert (await router.get_next_credential("CHAT")).id == "a"  # past it now


# ── PERMANENT failure: immediate exclusion, no waiting for a TTL ───────────


@pytest.mark.asyncio
async def test_permanent_failure_excludes_credential_immediately() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    router, backend = _router(clock=_Clock())

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=1)

    selected = await router.get_next_credential("CHAT")
    assert selected.id == "b"  # "a" excluded in this same test run, no TTL wait

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["error_type"] == "PERMANENT"
    assert call["credential_id"] == "a"


@pytest.mark.asyncio
async def test_permanent_failure_health_report_body_shape() -> None:
    cred_a = _credential(id="cred-perm", revision=4, priority=1, api_key="sk-topsecretvalue")
    _set_snapshot(version=7, chat=(cred_a,))
    router, backend = _router(clock=_Clock())

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=7)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["credential_id"] == "cred-perm"
    assert call["credential_revision"] == 4
    assert call["snapshot_version"] == 7
    assert call["error_type"] == "PERMANENT"
    assert call["error_code"]
    assert "sk-topsecretvalue" not in call["message"]
    assert call["occurred_at"]


# ── exhaustion ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_credentials_at_all_raises_no_available_credential_error() -> None:
    _set_snapshot(version=1, chat=())
    router, _backend = _router(clock=_Clock())

    with pytest.raises(NoAvailableCredentialError):
        await router.get_next_credential("CHAT")


@pytest.mark.asyncio
async def test_exhausting_every_credential_raises_no_available_credential_error() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    router, _backend = _router(clock=_Clock())

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=1)
    await router.record_failure(cred_b, _permanent_error(), snapshot_version=1)

    with pytest.raises(NoAvailableCredentialError) as exc_info:
        await router.get_next_credential("CHAT")
    assert exc_info.value.purpose == "CHAT"


@pytest.mark.asyncio
async def test_no_available_credential_error_does_not_loop_or_retry() -> None:
    """A single call either returns a credential or raises - nothing in
    `get_next_credential` retries/sleeps/loops internally."""

    _set_snapshot(version=1, chat=())
    router, _backend = _router(clock=_Clock())

    with pytest.raises(NoAvailableCredentialError):
        await router.get_next_credential("EXTRACTION")


# ── Redis down: degrade to in-memory, never crash ───────────────────────────


@pytest.mark.asyncio
async def test_redis_down_still_selects_a_usable_credential() -> None:
    cred_a = _credential(id="a", priority=1)
    _set_snapshot(version=1, chat=(cred_a,))
    router, _backend = _router(redis_client=DownRedis())

    selected = await router.get_next_credential("CHAT")

    assert selected.id == "a"


@pytest.mark.asyncio
async def test_redis_down_permanent_failure_still_excludes_in_memory() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    router, backend = _router(redis_client=DownRedis())

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=1)
    selected = await router.get_next_credential("CHAT")

    assert selected.id == "b"
    # The health report to Java must still be attempted even though Redis is down -
    # these are independent failure modes.
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_redis_down_transient_failure_degrades_without_raising() -> None:
    cred_a = _credential(id="a", priority=1)
    cred_b = _credential(id="b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))
    router, _backend = _router(redis_client=DownRedis())

    await router.record_failure(cred_a, _connection_error(), snapshot_version=1)  # must not raise
    selected = await router.get_next_credential("CHAT")

    assert selected.id == "b"


@pytest.mark.asyncio
async def test_health_report_failure_is_swallowed_not_raised() -> None:
    """Java being unreachable for the health report itself must never surface as an
    exception from `record_failure` - the caller already has a bigger problem (the
    provider call that just failed)."""

    cred_a = _credential(id="a", priority=1)
    _set_snapshot(version=1, chat=(cred_a,))
    router, backend = _router(clock=_Clock())
    backend.raise_on_report = RuntimeError("backend-java unreachable")

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=1)  # must not raise


# ── health report reflects the moment of failure, not a later-refreshed snapshot ─


@pytest.mark.asyncio
async def test_health_report_uses_snapshot_version_captured_at_failure_not_later_refresh() -> None:
    cred_a = _credential(id="a", revision=1, priority=1)
    _set_snapshot(version=5, chat=(cred_a,))
    router, backend = _router(clock=_Clock())

    # Capture what was active at the moment of failure, exactly like a future caller
    # (Task 11/12) would: read the credential + the snapshot version together, before
    # anything else happens.
    snapshot_version_at_failure = model_registry.get_current_snapshot().version
    assert snapshot_version_at_failure == 5

    # The snapshot drifts forward (e.g. a hot-reload triggered by an unrelated change)
    # *before* the failure is actually reported.
    _set_snapshot(version=99, chat=(cred_a,))

    await router.record_failure(cred_a, _permanent_error(), snapshot_version=snapshot_version_at_failure)

    assert len(backend.calls) == 1
    assert backend.calls[0]["snapshot_version"] == 5
    assert backend.calls[0]["credential_revision"] == 1


@pytest.mark.asyncio
async def test_health_report_uses_credential_revision_from_the_failing_call_not_a_new_one() -> None:
    """Same idea for `credentialRevision`: a rotation could bump the credential's own
    revision between the failure and the report being sent - the report must still
    carry the revision that was actually active when the call failed."""

    cred_a_old_revision = _credential(id="a", revision=1, priority=1)
    _set_snapshot(version=1, chat=(cred_a_old_revision,))
    router, backend = _router(clock=_Clock())

    # Simulate a rotation bumping this credential's revision after the failure
    # happened but before the report is sent.
    cred_a_new_revision = _credential(id="a", revision=2, priority=1)
    _set_snapshot(version=2, chat=(cred_a_new_revision,))

    await router.record_failure(cred_a_old_revision, _permanent_error(), snapshot_version=1)

    assert backend.calls[0]["credential_revision"] == 1
    assert backend.calls[0]["snapshot_version"] == 1


# ── the exclusion/cooldown key is scoped to (credential id, revision) ───────


def test_state_key_includes_credential_id_and_revision() -> None:
    assert _state_key("cred-1", 3) == "mr:cb:cred-1:3"


@pytest.mark.asyncio
async def test_new_revision_after_rotation_starts_with_clean_state() -> None:
    """A credential whose revision just bumped (rotation promoted a new candidate) is
    never blocked by a PERMANENT/cooldown marker written against its *old* revision -
    the key is scoped to (id, revision), so a new revision is always clean."""

    old_revision = _credential(id="a", revision=1, priority=1)
    _set_snapshot(version=1, chat=(old_revision,))
    router, _backend = _router(clock=_Clock())
    await router.record_failure(old_revision, _permanent_error(), snapshot_version=1)

    # Rotation promotes a new revision of the same credential id.
    new_revision = _credential(id="a", revision=2, priority=1)
    _set_snapshot(version=2, chat=(new_revision,))

    selected = await router.get_next_credential("CHAT")
    assert selected.revision == 2
