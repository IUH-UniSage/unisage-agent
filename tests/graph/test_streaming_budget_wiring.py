"""PROVIDER-scope budget wiring in `stream_agent_text`/`run_agent_text_with_failover`
(via `run_agent_text_with_failover`, the simpler of the two to drive directly).
Uses a real `BudgetTracker` against local Redis - acquire/release run actual Lua,
which a hand-rolled fake can't faithfully stand in for."""

from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
import redis.asyncio as redis_asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.budget.snapshot import set_current_budget_snapshot
from app.core.budget.tracker import BudgetTracker
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter, NoBudgetAvailableError
from app.graph.streaming import BudgetContext, run_agent_text_with_failover


class _FakeCircuitBreakerRedis:
    """Never blocks any credential - only PROVIDER budget denies in these tests."""

    async def set(self, name: str, _value: Any, *, ex: int | None = None) -> Any:
        del name, ex
        return True

    async def exists(self, _name: str) -> int:
        return 0

    async def aclose(self) -> Any:
        return None


class _FakeBackendClient:
    async def report_health(self, **kwargs: Any) -> None:
        del kwargs


def _credential(credential_id: str, provider: str, priority: int) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider=provider,
        model_name="gemini-2.5-flash",
        api_base_url="https://example.invalid",
        priority=priority,
        max_rpm=60,
        api_key="key",
    )


@pytest_asyncio.fixture
async def redis_client():
    client = redis_asyncio.Redis.from_url("redis://localhost:6379/0")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
def tracker(redis_client):
    return BudgetTracker(redis_client)


def _agent_factory(model: Any) -> Agent[None, str]:
    return Agent(model=model)


def _success_function(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="ok")])


@pytest.mark.asyncio
async def test_provider_budget_denial_falls_over_to_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch, redis_client, tracker
) -> None:
    from app.core.budget.snapshot import BudgetEntry, BudgetSnapshot

    primary = _credential("cred-primary", provider="openai", priority=1)
    fallback = _credential("cred-fallback", provider="anthropic", priority=2)
    monkeypatch.setattr(
        "app.core.registry.model_router.active_credentials_for",
        lambda purpose: [primary, fallback],
    )
    monkeypatch.setattr(
        "app.graph.streaming.build_model", lambda credential: FunctionModel(_success_function)
    )
    router = ModelRouter(
        redis_client=_FakeCircuitBreakerRedis(), backend_client=_FakeBackendClient()
    )

    # openai is BLOCK-exhausted; anthropic has no configured budget at all.
    set_current_budget_snapshot(
        BudgetSnapshot(
            version=1,
            entries=(
                BudgetEntry(
                    scope="PROVIDER",
                    scope_provider="openai",
                    scope_purpose=None,
                    period="DAILY",
                    limit_usd=Decimal("0"),
                    action="BLOCK",
                    throttle_max_concurrency=None,
                ),
            ),
        )
    )

    agent = _agent_factory(FunctionModel(_success_function))
    budget = BudgetContext(
        tracker=tracker,
        request_id="req-failover",
        reserve_seq=iter(range(10)).__next__,
        per_attempt_estimate_usd=Decimal("0.001"),
    )

    output = await run_agent_text_with_failover(
        agent,
        "hello",
        purpose="CHAT",
        credential=primary,
        snapshot_version=1,
        agent_factory=_agent_factory,
        router=router,
        budget=budget,
    )

    assert output == "ok"
    committed = await redis_client.get("budget:committed:PROVIDER:anthropic:DAILY:" + _today())
    assert committed is not None


@pytest.mark.asyncio
async def test_provider_budget_denial_on_every_candidate_raises_no_budget_available(
    monkeypatch: pytest.MonkeyPatch, redis_client, tracker
) -> None:
    from app.core.budget.snapshot import BudgetEntry, BudgetSnapshot

    primary = _credential("cred-primary", provider="openai", priority=1)
    monkeypatch.setattr(
        "app.core.registry.model_router.active_credentials_for", lambda purpose: [primary]
    )
    router = ModelRouter(
        redis_client=_FakeCircuitBreakerRedis(), backend_client=_FakeBackendClient()
    )

    set_current_budget_snapshot(
        BudgetSnapshot(
            version=1,
            entries=(
                BudgetEntry(
                    scope="PROVIDER",
                    scope_provider="openai",
                    scope_purpose=None,
                    period="DAILY",
                    limit_usd=Decimal("0"),
                    action="BLOCK",
                    throttle_max_concurrency=None,
                ),
            ),
        )
    )

    agent = _agent_factory(FunctionModel(_success_function))
    budget = BudgetContext(
        tracker=tracker,
        request_id="req-exhausted",
        reserve_seq=iter(range(10)).__next__,
        per_attempt_estimate_usd=Decimal("0.001"),
    )

    with pytest.raises(NoBudgetAvailableError):
        await run_agent_text_with_failover(
            agent,
            "hello",
            purpose="CHAT",
            credential=primary,
            snapshot_version=1,
            agent_factory=_agent_factory,
            router=router,
            budget=budget,
        )


@pytest.mark.asyncio
async def test_successful_call_releases_reservation_leaving_nothing_outstanding(
    monkeypatch: pytest.MonkeyPatch, redis_client, tracker
) -> None:
    from app.core.budget.snapshot import BudgetSnapshot

    primary = _credential("cred-primary", provider="openai", priority=1)
    monkeypatch.setattr(
        "app.core.registry.model_router.active_credentials_for", lambda purpose: [primary]
    )
    router = ModelRouter(
        redis_client=_FakeCircuitBreakerRedis(), backend_client=_FakeBackendClient()
    )
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=()))

    agent = _agent_factory(FunctionModel(_success_function))

    def on_attempt(outcome: Any) -> Decimal:
        del outcome
        return Decimal("0.00042")

    budget = BudgetContext(
        tracker=tracker,
        request_id="req-release",
        reserve_seq=iter(range(10)).__next__,
        per_attempt_estimate_usd=Decimal("0.001"),
    )

    await run_agent_text_with_failover(
        agent,
        "hello",
        purpose="CHAT",
        credential=primary,
        snapshot_version=1,
        agent_factory=_agent_factory,
        router=router,
        on_attempt=on_attempt,
        budget=budget,
    )

    reserved = await redis_client.get("budget:reserved:PROVIDER:openai:DAILY:" + _today())
    inflight = await redis_client.get("budget:inflight:PROVIDER:openai")
    committed = await redis_client.get("budget:committed:PROVIDER:openai:DAILY:" + _today())
    assert reserved is None or int(reserved) == 0
    assert inflight is None or int(inflight) == 0
    assert committed == b"420"


def _today() -> str:
    from app.core.budget.period import current_period_key

    return current_period_key("DAILY")
