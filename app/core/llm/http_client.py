"""The one place a provider HTTP client is built — plan.md "SSRF policy".

Every outbound call to a URL that came from the model registry (embedder,
multi-representation, graph model, verifier) must go through
`build_provider_http_client`. No other code may construct `httpx.Client`/
`httpx.AsyncClient` for a provider call — enforced by
`tests/core/test_no_raw_provider_clients.py`.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass

import certifi
import httpcore
import httpx

from app.core.ssrf_guard import PinnedNetworkBackend, PinnedNetworkBackendSync


class ProviderRedirectRejectedError(Exception):
    """A provider (or something impersonating it) answered with a 3xx — never followed."""

    def __init__(self, url: str, status_code: int) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(f"provider {url} -> unexpected redirect (HTTP {status_code})")


@dataclass(frozen=True)
class ProviderConnectionInfo:
    """The minimum a client needs to be built — not the full registry credential DTO (Task 4)."""

    api_base_url: str
    allowlisted_hosts: frozenset[str] = frozenset()


def _build_ssl_context() -> ssl.SSLContext:
    # Explicit, from certifi — never trust_env-derived (SSL_CERT_FILE etc are ignored below via
    # trust_env=False on the client itself; this context construction never reads env either).
    context = ssl.create_default_context(cafile=certifi.where())
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


async def _reject_redirects(response: httpx.Response) -> None:
    if response.is_redirect:
        raise ProviderRedirectRejectedError(str(response.request.url), response.status_code)


def _reject_redirects_sync(response: httpx.Response) -> None:
    if response.is_redirect:
        raise ProviderRedirectRejectedError(str(response.request.url), response.status_code)


def build_provider_http_client(credential: ProviderConnectionInfo) -> httpx.AsyncClient:
    """Builds the one kind of client every provider call must use.

    `trust_env=False` so `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`/`SSL_CERT_FILE` are all ignored.
    `follow_redirects=False` plus an event hook: every 3xx becomes `ProviderRedirectRejectedError`
    rather than something a caller could accidentally read the body of. The transport's
    `network_backend` is `PinnedNetworkBackend`, so every connection this client makes resolves
    once and connects to that verified IP — see `app/core/ssrf_guard.py`.
    """

    network_backend = PinnedNetworkBackend(allowlist=credential.allowlisted_hosts)
    pool = httpcore.AsyncConnectionPool(
        ssl_context=_build_ssl_context(),
        network_backend=network_backend,
    )
    transport = httpx.AsyncHTTPTransport()
    transport._pool = pool  # noqa: SLF001 - httpx doesn't expose network_backend injection publicly

    return httpx.AsyncClient(
        base_url=credential.api_base_url,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        timeout=httpx.Timeout(30.0, connect=10.0),
        event_hooks={"response": [_reject_redirects]},
    )


def build_provider_http_client_sync(credential: ProviderConnectionInfo) -> httpx.Client:
    """Sync counterpart of `build_provider_http_client`, for provider call sites that run
    outside an event loop (Celery tasks). Same guarantees, same knobs — `httpcore.ConnectionPool`
    + `PinnedNetworkBackendSync` instead of the async pair.
    """

    network_backend = PinnedNetworkBackendSync(allowlist=credential.allowlisted_hosts)
    pool = httpcore.ConnectionPool(
        ssl_context=_build_ssl_context(),
        network_backend=network_backend,
    )
    transport = httpx.HTTPTransport()
    transport._pool = pool  # noqa: SLF001 - httpx doesn't expose network_backend injection publicly

    return httpx.Client(
        base_url=credential.api_base_url,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        timeout=httpx.Timeout(30.0, connect=10.0),
        event_hooks={"response": [_reject_redirects_sync]},
    )
