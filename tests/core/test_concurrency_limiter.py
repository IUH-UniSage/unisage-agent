"""Tests for `app.core.registry.concurrency_limiter` and its use in
`app.core.llm.rate_limited_model`.

No live Redis: `FakeLeaseRedis` replays the acquire script's sorted-set steps in Python."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.llm.rate_limited_model import RateLimitedModel
from app.core.registry.concurrency_limiter import ConcurrencyLimiter
from app.core.registry.errors import (
    CredentialConcurrencySaturatedError,
    CredentialRpmSaturatedError,
)
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.rpm_limiter import RpmLimiter


def _credential(
    *,
    id: str = "cred-1",
    revision: int = 1,
    max_concurrency: int | None = 1,
    max_rpm: int | None = None,
) -> CredentialConfig:
    return CredentialConfig(
        id=id,
        revision=revision,
        source_type="CLOUD_API",
        provider="zai",
        model_name="glm-4.7-flash",
        api_base_url="https://api.z.ai/api/paas/v4",
        priority=1,
        max_rpm=max_rpm,
        api_key="zai-secret",
        max_concurrency=max_concurrency,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeLeaseRedis:
    """Stand-in for `redis.asyncio.Redis` running the acquire script and `zrem`."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, int]] = {}
        self.eval_calls = 0

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        self.eval_calls += 1
        key, now, lease, limit, member = keys_and_args
        zset = self.zsets.setdefault(key, {})
        for stale in [m for m, expiry in zset.items() if expiry <= now]:
            del zset[stale]
        if len(zset) < limit:
            zset[member] = now + lease
            return 1
        return 0

    async def zrem(self, name: str, *values: Any) -> Any:
        zset = self.zsets.get(name, {})
        return sum(1 for value in values if zset.pop(value, None) is not None)

    async def aclose(self) -> None:
        pass


class DownRedis:
    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        raise ConnectionError("simulated Redis outage")

    async def zrem(self, name: str, *values: Any) -> Any:
        raise ConnectionError("simulated Redis outage")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_refuses_once_every_slot_is_held_and_frees_on_release() -> None:
    limiter = ConcurrencyLimiter(redis_client=FakeLeaseRedis(), clock=_Clock())
    credential = _credential(max_concurrency=2)

    first = await limiter.acquire(credential)
    second = await limiter.acquire(credential)
    assert first is not None and second is not None
    assert await limiter.acquire(credential) is None

    await first.release()
    assert await limiter.acquire(credential) is not None


@pytest.mark.asyncio
async def test_release_is_idempotent() -> None:
    limiter = ConcurrencyLimiter(redis_client=FakeLeaseRedis(), clock=_Clock())
    credential = _credential(max_concurrency=1)

    lease = await limiter.acquire(credential)
    assert lease is not None
    await lease.release()
    await lease.release()

    other = await limiter.acquire(credential)
    assert other is not None
    await lease.release()  # must not free `other`'s slot
    assert await limiter.acquire(credential) is None


@pytest.mark.asyncio
async def test_a_lease_left_by_a_dead_worker_expires() -> None:
    clock = _Clock()
    limiter = ConcurrencyLimiter(redis_client=FakeLeaseRedis(), clock=clock, lease_seconds=600)
    credential = _credential(max_concurrency=1)

    assert await limiter.acquire(credential) is not None  # never released
    clock.advance(599)
    assert await limiter.acquire(credential) is None
    clock.advance(2)
    assert await limiter.acquire(credential) is not None


@pytest.mark.asyncio
async def test_slots_are_per_credential_and_revision() -> None:
    limiter = ConcurrencyLimiter(redis_client=FakeLeaseRedis(), clock=_Clock())

    assert await limiter.acquire(_credential(id="a")) is not None
    assert await limiter.acquire(_credential(id="b")) is not None
    assert await limiter.acquire(_credential(id="a", revision=2)) is not None
    assert await limiter.acquire(_credential(id="a")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrency", [None, 0])
async def test_credential_without_positive_limit_is_never_limited(
    max_concurrency: int | None,
) -> None:
    redis_client = FakeLeaseRedis()
    limiter = ConcurrencyLimiter(redis_client=redis_client, clock=_Clock())

    for _ in range(5):
        lease = await limiter.acquire(_credential(max_concurrency=max_concurrency))
        assert lease is not None
        await lease.release()
    assert redis_client.eval_calls == 0


@pytest.mark.asyncio
async def test_redis_down_degrades_to_in_memory_leases() -> None:
    limiter = ConcurrencyLimiter(redis_client=DownRedis(), clock=_Clock())
    credential = _credential(max_concurrency=1)

    lease = await limiter.acquire(credential)
    assert lease is not None
    assert await limiter.acquire(credential) is None
    await lease.release()
    assert await limiter.acquire(credential) is not None


# ── RateLimitedModel ────────────────────────────────────────────────────────


def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("ok")])


async def _echo_stream(messages: list[ModelMessage], info: AgentInfo) -> Any:
    yield "ok"


def _agent(
    credential: CredentialConfig, *, rpm_limiter: RpmLimiter | None = None
) -> tuple[Agent[None, str], FakeLeaseRedis]:
    redis_client = FakeLeaseRedis()
    model = RateLimitedModel(
        FunctionModel(_echo, stream_function=_echo_stream),
        credential,
        limiter=rpm_limiter,
        concurrency_limiter=ConcurrencyLimiter(redis_client=redis_client, clock=_Clock()),
    )
    return Agent(model), redis_client


@pytest.mark.asyncio
async def test_open_stream_holds_the_slot_until_it_closes() -> None:
    agent, redis_client = _agent(_credential(max_concurrency=1))

    async with agent.run_stream("hi") as result:
        with pytest.raises(CredentialConcurrencySaturatedError) as exc_info:
            await agent.run("second caller")
        assert await result.get_output() == "ok"

    assert exc_info.value.max_concurrency == 1
    assert all(not zset for zset in redis_client.zsets.values())
    assert (await agent.run("after the stream")).output == "ok"


@pytest.mark.asyncio
async def test_sequential_requests_each_release_their_slot() -> None:
    agent, redis_client = _agent(_credential(max_concurrency=1))

    for _ in range(3):
        assert (await agent.run("hi")).output == "ok"
    assert all(not zset for zset in redis_client.zsets.values())


@pytest.mark.asyncio
async def test_rpm_refusal_gives_the_concurrency_slot_back() -> None:
    class FullRpmLimiter(RpmLimiter):
        async def acquire(self, credential: CredentialConfig) -> float | None:
            return 12.0

    agent, redis_client = _agent(
        _credential(max_concurrency=1, max_rpm=1), rpm_limiter=FullRpmLimiter()
    )

    with pytest.raises(CredentialRpmSaturatedError):
        await agent.run("hi")
    assert all(not zset for zset in redis_client.zsets.values())
