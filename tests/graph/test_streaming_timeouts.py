"""Per-attempt time limits in `stream_agent_text` (first visible token) and
`run_agent_text_with_failover` (whole call): running out fails over to the next
credential like any other provider failure, and a stream that has started is never cut."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import app.core.registry.model_router as model_router_module
from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.graph.streaming import run_agent_text_with_failover, stream_agent_text


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def set(self, name: str, value: Any, *, ex: int | None = None) -> Any:
        self.values[name] = value
        return True

    async def exists(self, name: str) -> int:
        return 1 if name in self.values else 0

    async def get(self, name: str) -> Any:
        return self.values.get(name)

    async def aclose(self) -> Any:
        return None


class _FakeBackendClient:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    async def report_health(self, **kwargs: Any) -> None:
        self.reports.append(kwargs)


def _credential(credential_id: str, priority: int) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider="google",
        model_name=f"model-{credential_id}",
        api_base_url="https://example.invalid",
        priority=priority,
        max_rpm=None,
        api_key="key",
    )


PRIMARY = _credential("primary", 1)
FALLBACK = _credential("fallback", 2)


async def _hanging_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    await asyncio.sleep(10)
    yield "never"


async def _ok_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield "ok"


async def _slow_after_first_token(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str]:
    yield "first "
    await asyncio.sleep(0.5)
    yield "second"


async def _hanging_run(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    await asyncio.sleep(10)
    return ModelResponse(parts=[TextPart(content="never")])


def _ok_run(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="ok")])


def _agent_factory(model: Any) -> Agent[None, str]:
    return Agent(model=model)


async def _no_alert(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _FakeBackendClient:
    monkeypatch.setattr(
        model_router_module, "active_credentials_for", lambda purpose: (PRIMARY, FALLBACK)
    )
    monkeypatch.setattr(model_router_module, "alert_credential_failure", _no_alert)
    return _FakeBackendClient()


def _router(backend: _FakeBackendClient) -> ModelRouter:
    return ModelRouter(redis_client=_FakeRedis(), backend_client=backend)


@pytest.mark.asyncio
async def test_no_first_token_in_time_fails_over_to_the_next_credential(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackendClient
) -> None:
    monkeypatch.setattr(settings, "CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        "app.graph.streaming.build_model",
        lambda credential: FunctionModel(stream_function=_ok_stream),
    )
    tokens: list[str] = []

    async def sink(token: str) -> None:
        tokens.append(token)

    output = await stream_agent_text(
        _agent_factory(FunctionModel(stream_function=_hanging_stream)),
        "hello",
        sink,
        purpose="CHAT",
        credential=PRIMARY,
        snapshot_version=1,
        agent_factory=_agent_factory,
        router=_router(backend),
    )

    assert output == "ok"
    assert [report["credential_id"] for report in backend.reports] == ["primary"]
    assert backend.reports[0]["error_code"] == "TimeoutError"


@pytest.mark.asyncio
async def test_a_started_stream_is_not_cut_by_the_first_token_limit(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackendClient
) -> None:
    # Above `stream_text`'s 0.1s debounce, so the first token can arrive in time.
    monkeypatch.setattr(settings, "CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 0.25)

    async def sink(_token: str) -> None:
        return None

    output = await stream_agent_text(
        _agent_factory(FunctionModel(stream_function=_slow_after_first_token)),
        "hello",
        sink,
        purpose="CHAT",
        credential=PRIMARY,
        snapshot_version=1,
        agent_factory=_agent_factory,
        router=_router(backend),
    )

    assert output == "first second"
    assert backend.reports == []


@pytest.mark.asyncio
async def test_aux_call_over_its_time_limit_fails_over(
    backend: _FakeBackendClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.graph.streaming.build_model", lambda credential: FunctionModel(_ok_run)
    )

    output = await run_agent_text_with_failover(
        _agent_factory(FunctionModel(_hanging_run)),
        "hello",
        purpose="CHAT",
        credential=PRIMARY,
        snapshot_version=1,
        agent_factory=_agent_factory,
        router=_router(backend),
        timeout_seconds=0.05,
    )

    assert output == "ok"
    assert [report["credential_id"] for report in backend.reports] == ["primary"]
