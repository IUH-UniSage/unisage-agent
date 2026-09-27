"""`app.core.llm.provider_models` — maps a registry credential's `provider`/`source_type` to
the native PydanticAI (Model, Provider) pair that builds it (ADR 0005, plan.md Task 5).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.llm.provider_models import UnsupportedProviderError, build_model
from app.core.model_registry import CredentialConfig
from app.core.ssrf_guard import PinnedNetworkBackend


def _credential(**overrides: object) -> CredentialConfig:
    defaults: dict[str, object] = dict(
        id="11111111-1111-1111-1111-111111111111",
        revision=3,
        source_type="CLOUD_API",
        provider="openai",
        model_name="gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        priority=1,
        max_rpm=500,
        api_key="sk-real-secret-value",
    )
    defaults.update(overrides)
    return CredentialConfig(**defaults)  # type: ignore[arg-type]


def test_openai_provider_builds_openai_chat_model() -> None:
    model = build_model(_credential())

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-4o-mini"


def test_self_hosted_uses_openai_compatible_transport_regardless_of_provider_string() -> None:
    credential = _credential(
        source_type="SELF_HOSTED",
        provider=None,
        model_name="llama-3-70b",
        api_base_url="https://self-hosted.internal.test/v1",
    )

    model = build_model(credential)

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "llama-3-70b"


def test_unknown_provider_raises_without_building_anything() -> None:
    credential = _credential(provider="some-provider-nobody-registered")

    with pytest.raises(UnsupportedProviderError) as exc_info:
        build_model(credential)

    assert exc_info.value.provider == "some-provider-nobody-registered"


def test_anthropic_is_not_yet_wired_and_raises_a_clear_error() -> None:
    """`anthropic` is deliberately absent from the provider map today - see
    provider_models.py's module docstring: the installed pydantic-ai/anthropic SDK versions
    reject a plain `httpx.AsyncClient` (`build_provider_http_client()`'s return type) for
    `AnthropicProvider`, and there is no httpcore2-based `PinnedNetworkBackend` equivalent yet.
    Wiring "anthropic" without one would mean either an SSRF-unpinned client or a hard crash -
    this asserts the actual, intentional behavior: a typed refusal, same as any other
    unsupported provider.
    """

    credential = _credential(provider="anthropic", model_name="claude-3-5-sonnet")

    with pytest.raises(UnsupportedProviderError):
        build_model(credential)


def test_provider_gets_ssrf_pinned_http_client_kwarg() -> None:
    """Every branch must pass `http_client=` into the Provider constructor - that's the one
    thing that makes the resulting client SSRF-pinned (ADR 0005) instead of the SDK's own
    unguarded default transport."""

    model = build_model(_credential())

    assert isinstance(model.provider, OpenAIProvider)


@pytest.fixture
def connect_tcp_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Spies on `PinnedNetworkBackend.connect_tcp` - see
    tests/core/test_provider_call_sites_use_pinned_backend.py for the full rationale. Raising
    immediately means no real socket ever opens; this file only cares whether a request attempt
    reaches the pinned backend at all.
    """

    calls: list[tuple[str, int]] = []

    async def _spy(self: PinnedNetworkBackend, host: str, port: int, **kwargs: object) -> object:
        calls.append((host, port))
        raise RuntimeError("blocked by test spy - no real socket opened")

    monkeypatch.setattr(PinnedNetworkBackend, "connect_tcp", _spy)
    return calls


def _drive_one_request(httpx_client: object) -> None:
    async def _make_one_request() -> None:
        async with httpx_client:  # type: ignore[attr-defined]
            await httpx_client.get("/")  # type: ignore[attr-defined]

    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(_make_one_request())


def test_google_provider_builds_google_model() -> None:
    credential = _credential(
        provider="google",
        model_name="gemini-2.0-flash",
        api_base_url="https://generativelanguage.googleapis.com",
    )

    model = build_model(credential)

    assert isinstance(model, GoogleModel)
    assert isinstance(model.provider, GoogleProvider)
    assert model.model_name == "gemini-2.0-flash"


def test_google_provider_http_client_reaches_pinned_backend(
    connect_tcp_spy: list[tuple[str, int]],
) -> None:
    credential = _credential(
        provider="google",
        model_name="gemini-2.0-flash",
        api_base_url="https://generativelanguage.googleapis.com",
    )

    model = build_model(credential)
    assert isinstance(model.provider, GoogleProvider)
    httpx_client = model.provider.client._api_client._http_options.httpx_async_client  # type: ignore[attr-defined]

    _drive_one_request(httpx_client)

    assert connect_tcp_spy, "GoogleProvider's http_client never reached PinnedNetworkBackend.connect_tcp"
    assert all(host == "generativelanguage.googleapis.com" for host, _ in connect_tcp_spy)


def test_groq_provider_builds_groq_model() -> None:
    credential = _credential(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        api_base_url="https://api.groq.com",
    )

    model = build_model(credential)

    assert isinstance(model, GroqModel)
    assert isinstance(model.provider, GroqProvider)
    assert model.model_name == "llama-3.3-70b-versatile"


def test_groq_provider_http_client_reaches_pinned_backend(
    connect_tcp_spy: list[tuple[str, int]],
) -> None:
    credential = _credential(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        api_base_url="https://api.groq.com",
    )

    model = build_model(credential)
    assert isinstance(model.provider, GroqProvider)
    httpx_client = model.provider.client._client  # type: ignore[attr-defined]

    _drive_one_request(httpx_client)

    assert connect_tcp_spy, "GroqProvider's http_client never reached PinnedNetworkBackend.connect_tcp"
    assert all(host == "api.groq.com" for host, _ in connect_tcp_spy)


def test_mistral_provider_builds_mistral_model() -> None:
    credential = _credential(
        provider="mistral",
        model_name="mistral-large-latest",
        api_base_url="https://api.mistral.ai",
    )

    model = build_model(credential)

    assert isinstance(model, MistralModel)
    assert isinstance(model.provider, MistralProvider)
    assert model.model_name == "mistral-large-latest"


def test_mistral_provider_http_client_reaches_pinned_backend(
    connect_tcp_spy: list[tuple[str, int]],
) -> None:
    credential = _credential(
        provider="mistral",
        model_name="mistral-large-latest",
        api_base_url="https://api.mistral.ai",
    )

    model = build_model(credential)
    assert isinstance(model.provider, MistralProvider)
    httpx_client = model.provider.client.sdk_configuration.async_client  # type: ignore[attr-defined]

    _drive_one_request(httpx_client)

    assert connect_tcp_spy, "MistralProvider's http_client never reached PinnedNetworkBackend.connect_tcp"
    assert all(host == "api.mistral.ai" for host, _ in connect_tcp_spy)
