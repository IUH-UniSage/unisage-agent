"""Redirect-to-blocked-target - todo.md Task 0.6's "Test redirect" item.

A real (loopback, plain HTTP - TLS isn't the point here) server answers every
request with `302 Location: http://169.254.169.254/` - the canonical cloud
metadata SSRF target. The factory client must raise
`ProviderRedirectRejectedError` (see `app/core/llm/http_client.py`) rather
than following it, and - the part a purely code-level "follow_redirects is
False" assertion can't prove - the blocked target must never actually be
dialed. `169.254.169.254` isn't bindable on a dev/CI box to prove that
directly, so this spies on `PinnedNetworkBackendSync.connect_tcp` (the one
chokepoint every provider-call socket goes through) instead: zero calls
naming that host proves it never even entered the resolve step, let alone
opened a socket.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest

from app.core.llm.http_client import (
    ProviderConnectionInfo,
    ProviderRedirectRejectedError,
    build_provider_http_client_sync,
)
from app.core.security.ssrf_guard import PinnedNetworkBackendSync

BLOCKED_TARGET = "169.254.169.254"


class _RedirectingHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(302)
        self.send_header("Location", f"http://{BLOCKED_TARGET}/")
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def redirecting_server() -> Iterator[tuple[str, int]]:
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", port
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def connect_tcp_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    original = PinnedNetworkBackendSync.connect_tcp

    def _spying_connect_tcp(
        self: PinnedNetworkBackendSync, host: str, port: int, **kwargs: object
    ) -> object:
        calls.append((host, port))
        return original(self, host, port, **kwargs)

    monkeypatch.setattr(PinnedNetworkBackendSync, "connect_tcp", _spying_connect_tcp)
    return calls


def test_redirect_to_metadata_ip_is_rejected_and_never_dialed(
    redirecting_server: tuple[str, int],
    connect_tcp_spy: list[tuple[str, int]],
) -> None:
    host, port = redirecting_server
    client = build_provider_http_client_sync(
        ProviderConnectionInfo(
            api_base_url=f"http://{host}:{port}", allowlisted_hosts=frozenset({host})
        )
    )
    with client:
        with pytest.raises(ProviderRedirectRejectedError) as exc_info:
            client.get("/")

    assert exc_info.value.status_code == 302
    # The initial request to the redirecting server is the only connection
    # ever attempted - the blocked target must not appear at all.
    assert connect_tcp_spy == [(host, port)]
    assert all(h != BLOCKED_TARGET for h, _ in connect_tcp_spy)


@pytest.mark.parametrize("status_code", [301, 307, 308])
def test_every_redirect_status_is_rejected(
    status_code: int,
    connect_tcp_spy: list[tuple[str, int]],
) -> None:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(status_code)
            self.send_header("Location", f"http://{BLOCKED_TARGET}/")
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = build_provider_http_client_sync(
            ProviderConnectionInfo(
                api_base_url=f"http://127.0.0.1:{port}",
                allowlisted_hosts=frozenset({"127.0.0.1"}),
            )
        )
        with client:
            with pytest.raises(ProviderRedirectRejectedError) as exc_info:
                client.get("/")
        assert exc_info.value.status_code == status_code
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert all(h != BLOCKED_TARGET for h, _ in connect_tcp_spy)
