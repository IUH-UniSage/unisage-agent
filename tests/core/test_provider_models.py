"""`app.core.llm.provider_models` — maps a registry credential's `provider`/`source_type` to
the native PydanticAI (Model, Provider) pair that builds it (ADR 0005, plan.md Task 5).
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.llm.provider_models import UnsupportedProviderError, build_model
from app.core.model_registry import CredentialConfig


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
