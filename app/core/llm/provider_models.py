"""Maps a registry credential's `provider` string to the native PydanticAI (`Model`, `Provider`)
pair that builds it — ADR 0005 (`unisage-backend/docs/adr/0005-dynamic-model-registry.md`):
LiteLLM's SDK can't accept `http_client=` injection, so every provider PydanticAI builds here
must accept `http_client=` on its own `Provider` constructor so `build_provider_http_client()`
(the SSRF-pinned factory, Task 0.6) can be threaded through.

`get_graph_models()` (app/api/deps.py) is the only caller today; Task 6/9's verifier builds a
candidate credential's `Model` the same way.
"""

from __future__ import annotations

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.zai import ZaiProvider

from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client
from app.core.llm.rate_limited_model import RateLimitedModel
from app.core.registry.model_registry import CredentialConfig

_SELF_HOSTED_SOURCE_TYPE = "SELF_HOSTED"

# credential.provider -> (Model class, Provider class). Every entry here was confirmed
# empirically (not from docs) to accept `http_client=<httpx.AsyncClient>` on its own `Provider`
# constructor and to actually store that exact object as the transport its SDK client dispatches
# requests through — the same test the Task 0.2/Task 5 spikes ran for OpenAI/Anthropic, rerun here
# for each provider below against the pydantic-ai/SDK versions actually resolved on this branch
# (pydantic-ai 2.49.0):
#   - "google" (`GoogleProvider`, backed by `google-genai`'s `Client`): accepts a plain
#     `httpx.AsyncClient` and stores it at `client._api_client._http_options.httpx_async_client`.
#     It emits a `PydanticAIDeprecationWarning` ("use `httpx2.AsyncClient` instead") but does not
#     raise - same non-fatal deprecation OpenAI's own path already carries.
#   - "zai" (`ZaiProvider`, an OpenAI-compatible provider): accepts a plain `httpx.AsyncClient`
#     like "openai", but its constructor has no `base_url` parameter (the URL is a fixed
#     property) - `BaseUrlZaiProvider` below adds one so it fits `build_model()`'s uniform call.
#     `ZaiModel` (an `OpenAIChatModel`) is what understands GLM's thinking mode and
#     `reasoning_content`, which the plain OpenAI model would ignore.
#
# "anthropic" is deliberately NOT here, even though ADR 0005 assumed it would use the same
# http_client= mechanism as OpenAI ("... AnthropicProvider, ... dùng cùng cơ chế http_client=").
# It doesn't, in the pydantic-ai/anthropic versions actually resolved on this branch
# (pydantic-ai 2.49.0): `AnthropicProvider` only accepts `httpx2.AsyncClient` — a separate forked
# package with its own transport/connection-pool stack (`httpcore2`, not `httpcore`) — and raises
# `TypeError` at construction if handed a plain `httpx.AsyncClient` (confirmed empirically, not a
# type-hint-only mismatch: `AnthropicProvider(api_key=..., http_client=httpx.AsyncClient())` ->
# "Invalid `http_client` argument; `httpx.AsyncClient` is from the `httpx` package, but this SDK
# uses `httpx2`."). `build_provider_http_client()` (app/core/llm/http_client.py) only ever builds
# a plain `httpx.AsyncClient` backed by httpcore's `PinnedNetworkBackend`
# (app/core/security/ssrf_guard.py) — there is no SSRF-pinned client this module can hand to
# `AnthropicProvider` today. Wiring "anthropic" in here anyway would mean either constructing an
# *unpinned* httpx2 client (an SSRF hole — never acceptable) or building a second,
# httpcore2-based `PinnedNetworkBackend` equivalent (a change to `app/core/security/ssrf_guard.py`/
# `http_client.py`, with its own redirect/TLS/rebinding/proxy test suite — out of this task's
# scope). Until that exists, a credential with provider="anthropic" raises
# `UnsupportedProviderError` rather than silently skipping the guard.
#
# Two more providers with a native pydantic-ai `Model`/`Provider` pair were checked and rejected,
# without adding a second HTTP stack (also out of scope):
#   - "xai" (`XaiProvider`, `pydantic_ai.providers.xai`): its native SDK (`xai-sdk`) is gRPC, not
#     HTTP at all - the constructor has no `http_client=` parameter of any kind (plain httpx,
#     httpx2, or otherwise) to inject the pinned transport into.
#   - "deepseek" (`DeepSeekProvider`, `pydantic_ai.providers.deepseek`): its underlying transport
#     *is* the OpenAI SDK and does accept a plain `httpx.AsyncClient` (same as "openai" above), but
#     its constructor has no `base_url` parameter at all (`base_url` is a hardcoded property
#     pointing at DeepSeek's own API) - it doesn't fit `build_model()`'s uniform
#     `provider_cls(base_url=..., api_key=..., http_client=...)` call without a SELF_HOSTED-style
#     special case. Left out as a scoping decision, not an SSRF one; picking it back up just needs
#     that one branch added to `build_model()`.


class BaseUrlZaiProvider(ZaiProvider):
    """`ZaiProvider` that uses the credential's own base URL - pydantic-ai's reads a fixed
    `base_url` property, which `ZaiProvider.__init__` uses to build its client."""

    def __init__(
        self, *, base_url: str | None, api_key: str, http_client: httpx.AsyncClient
    ) -> None:
        self._configured_base_url = base_url
        super().__init__(api_key=api_key, http_client=http_client)

    @property
    def base_url(self) -> str:
        return self._configured_base_url or super().base_url


_PROVIDER_MAP: dict[str, tuple[type[Model], type[Provider]]] = {
    "openai": (OpenAIChatModel, OpenAIProvider),
    "google": (GoogleModel, GoogleProvider),
    "zai": (ZaiModel, BaseUrlZaiProvider),
}


class UnsupportedProviderError(Exception):
    """Raised when a credential's provider has no native PydanticAI transport proven to accept
    the SSRF-pinned `http_client=` injection (ADR 0005).

    Mirrors plan.md's `CHAT_MODEL_PROVIDER_UNSUPPORTED` / health code
    `PROVIDER_TRANSPORT_UNSUPPORTED` — there is no live health-report call site wired to this yet
    (that's Task 6/9), so this is just a clear, typed exception for now. There is never a fallback
    to a default SDK client for an unsupported provider.
    """

    def __init__(self, provider: str | None) -> None:
        self.provider = provider
        super().__init__(
            f"CHAT_MODEL_PROVIDER_UNSUPPORTED: no SSRF-safe native PydanticAI transport for "
            f"provider {provider!r} (health code PROVIDER_TRANSPORT_UNSUPPORTED)"
        )


def build_model(credential: CredentialConfig) -> Model:
    """Builds the native PydanticAI `Model` for one registry credential, wired through the
    SSRF-pinned `build_provider_http_client()` factory — never a default SDK client.

    `SELF_HOSTED` credentials always use the OpenAI-compatible transport
    (`OpenAIChatModel` + `OpenAIProvider(base_url=credential.api_base_url, ...)`) regardless of
    `credential.provider`, since a self-hosted server speaks the OpenAI wire format. Every other
    `source_type` is looked up by `credential.provider` in `_PROVIDER_MAP`; no entry there raises
    `UnsupportedProviderError` — this function never builds anything in that case.

    A credential with a positive `max_rpm` or `max_concurrency` comes back wrapped in
    `RateLimitedModel`, which enforces those limits before every request (the native model is
    its `.wrapped`).
    """

    http_client = build_provider_http_client(
        ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
    )

    if credential.source_type == _SELF_HOSTED_SOURCE_TYPE:
        model_cls, provider_cls = OpenAIChatModel, OpenAIProvider
    else:
        entry = _PROVIDER_MAP.get(credential.provider or "")
        if entry is None:
            raise UnsupportedProviderError(credential.provider)
        model_cls, provider_cls = entry

    provider = provider_cls(
        base_url=credential.api_base_url,
        api_key=credential.api_key,
        http_client=http_client,
    )
    return _with_limits(model_cls(credential.model_name or "", provider=provider), credential)


def _with_limits(model: Model, credential: CredentialConfig) -> Model:
    limits = (credential.max_rpm, credential.max_concurrency)
    if any(limit is not None and limit > 0 for limit in limits):
        return RateLimitedModel(model, credential)
    return model
