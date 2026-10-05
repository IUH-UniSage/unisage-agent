"""Tests for `app.core.registry.rpm_limiter` and `app.core.llm.rate_limited_model`.

No live Redis: `FakeZsetRedis` replays the acquire script's sorted-set steps in Python, so the
window arithmetic is exercised end to end; the Lua itself is the same few commands."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.llm.rate_limited_model import RateLimitedModel
from app.core.registry.errors import CredentialRpmSaturatedError
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.rpm_limiter import RpmLimiter


def _credential(
    *, id: str = "cred-1", revision: int = 1, max_rpm: int | None = 2
) -> CredentialConfig:
    return CredentialConfig(
        id=id,
        revision=revision,
        source_type="CLOUD_API",
        provider="google",
        model_name="gemini-3.1-flash-lite-preview",
        api_base_url="https://generativelanguage.googleapis.com",
        priority=1,
        max_rpm=max_rpm,
        api_key="sk-secret",
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeZsetRedis:
    """Stand-in for `redis.asyncio.Redis.eval` running the acquire script."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, int]] = {}
        self.eval_calls = 0

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        self.eval_calls += 1
        key, now, window, limit, member = keys_and_args
        zset = self.zsets.setdefault(key, {})
        for stale in [m for m, score in zset.items() if score <= now - window]:
            del zset[stale]
        if len(zset) < limit:
            zset[member] = now
            return -1
        return max(0, min(zset.values()) + window - now)

    async def aclose(self) -> None:
        pass


class DownRedis:
    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        raise ConnectionError("simulated Redis outage")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_allows_max_rpm_calls_then_refuses_with_time_until_a_slot_frees() -> None:
    clock = _Clock()
    limiter = RpmLimiter(redis_client=FakeZsetRedis(), clock=clock)
    credential = _credential(max_rpm=2)

    assert await limiter.acquire(credential) is None
    clock.advance(10.0)
    assert await limiter.acquire(credential) is None
    clock.advance(5.0)

    assert await limiter.acquire(credential) == pytest.approx(45.0)


@pytest.mark.asyncio
async def test_slot_frees_once_the_oldest_call_leaves_the_window() -> None:
    clock = _Clock()
    limiter = RpmLimiter(redis_client=FakeZsetRedis(), clock=clock)
    credential = _credential(max_rpm=1)

    assert await limiter.acquire(credential) is None
    clock.advance(59.0)
    assert await limiter.acquire(credential) is not None
    clock.advance(1.5)
    assert await limiter.acquire(credential) is None


@pytest.mark.asyncio
async def test_refused_call_does_not_take_a_slot() -> None:
    clock = _Clock()
    redis_client = FakeZsetRedis()
    limiter = RpmLimiter(redis_client=redis_client, clock=clock)
    credential = _credential(max_rpm=1)

    await limiter.acquire(credential)
    await limiter.acquire(credential)
    await limiter.acquire(credential)

    assert len(redis_client.zsets["mr:rpm:cred-1:1"]) == 1


@pytest.mark.asyncio
async def test_windows_are_per_credential_and_revision() -> None:
    limiter = RpmLimiter(redis_client=FakeZsetRedis(), clock=_Clock())

    assert await limiter.acquire(_credential(id="a", max_rpm=1)) is None
    assert await limiter.acquire(_credential(id="b", max_rpm=1)) is None
    assert await limiter.acquire(_credential(id="a", revision=2, max_rpm=1)) is None
    assert await limiter.acquire(_credential(id="a", max_rpm=1)) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("max_rpm", [None, 0])
async def test_credential_without_positive_max_rpm_is_never_limited(max_rpm: int | None) -> None:
    redis_client = FakeZsetRedis()
    limiter = RpmLimiter(redis_client=redis_client, clock=_Clock())

    for _ in range(5):
        assert await limiter.acquire(_credential(max_rpm=max_rpm)) is None
    assert redis_client.eval_calls == 0


@pytest.mark.asyncio
async def test_redis_down_degrades_to_in_memory_window() -> None:
    clock = _Clock()
    limiter = RpmLimiter(redis_client=DownRedis(), clock=clock)
    credential = _credential(max_rpm=2)

    assert await limiter.acquire(credential) is None
    assert await limiter.acquire(credential) is None
    assert await limiter.acquire(credential) == pytest.approx(60.0)
    clock.advance(61.0)
    assert await limiter.acquire(credential) is None


# ── RateLimitedModel ────────────────────────────────────────────────────────


def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("ok")])


async def _echo_stream(messages: list[ModelMessage], info: AgentInfo) -> Any:
    yield "ok"


def _rate_limited_agent(max_rpm: int) -> tuple[Agent[None, str], FakeZsetRedis]:
    redis_client = FakeZsetRedis()
    model = RateLimitedModel(
        FunctionModel(_echo, stream_function=_echo_stream),
        _credential(max_rpm=max_rpm),
        limiter=RpmLimiter(redis_client=redis_client, clock=_Clock()),
    )
    return Agent(model), redis_client


@pytest.mark.asyncio
async def test_rate_limited_model_counts_each_request_and_refuses_past_max_rpm() -> None:
    agent, redis_client = _rate_limited_agent(max_rpm=2)

    assert (await agent.run("hi")).output == "ok"
    assert (await agent.run("hi")).output == "ok"
    with pytest.raises(CredentialRpmSaturatedError) as exc_info:
        await agent.run("hi")

    assert exc_info.value.credential_id == "cred-1"
    assert exc_info.value.max_rpm == 2
    assert redis_client.eval_calls == 3


@pytest.mark.asyncio
async def test_rate_limited_model_refuses_a_stream_before_anything_is_streamed() -> None:
    agent, _redis_client = _rate_limited_agent(max_rpm=1)

    async with agent.run_stream("hi") as result:
        assert await result.get_output() == "ok"

    streamed: list[str] = []
    with pytest.raises(CredentialRpmSaturatedError):
        async with agent.run_stream("hi") as result:
            async for chunk in result.stream_text(delta=True):
                streamed.append(chunk)
    assert streamed == []
