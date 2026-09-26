"""`trust_env` - todo.md Task 0.6's "Test trust_env" item.

`HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` point at a small counting HTTP proxy
this test runs itself, and `SSL_CERT_FILE` points at an unrelated CA bundle.
`build_provider_http_client`/`_sync` pass `trust_env=False`, so none of that
should matter: the client must make zero connections through the proxy, and
its SSL context must still come from `certifi` - not from `SSL_CERT_FILE` -
per `app/core/llm/http_client.py::_build_ssl_context`.

The "does it still use certifi, not SSL_CERT_FILE" half is checked
structurally (spy on `certifi.where`, since that's the one function
`_build_ssl_context` calls to decide its trust store) rather than by trying
to prove a negative over the network - `test_ssrf_host_sni_cert.py` already
proves `_build_ssl_context`'s cafile is what actually governs trust.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from collections.abc import Iterator

import pytest

from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.ssrf_guard import PinnedNetworkBackendSync


class _CountingProxyHandler(http.server.BaseHTTPRequestHandler):
    """Bare minimum "proxy": counts every request/CONNECT it receives and
    refuses to actually forward anything - if the client under test ever
    reaches this at all, `trust_env=False` has failed regardless of what
    this handler does next."""

    def log_message(self, *args: object) -> None:
        pass

    def _count_and_reject(self) -> None:
        self.server.request_count += 1  # type: ignore[attr-defined]
        self.send_response(502)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._count_and_reject()

    def do_CONNECT(self) -> None:
        self._count_and_reject()


class _CountingProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    request_count: int = 0
    daemon_threads = True


@pytest.fixture
def counting_proxy() -> Iterator[tuple[str, _CountingProxyServer]]:
    httpd = _CountingProxyServer(("127.0.0.1", 0), _CountingProxyHandler)
    httpd.request_count = 0
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def target_server() -> Iterator[tuple[str, int]]:
    class _OkHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", port
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_client_ignores_proxy_env_and_ssl_cert_file_env(
    counting_proxy: tuple[str, _CountingProxyServer],
    target_server: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    proxy_url, proxy_server = counting_proxy
    target_host, target_port = target_server

    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("ALL_PROXY", proxy_url)

    unrelated_ca_bundle = tmp_path / "unrelated-ca-bundle.pem"  # type: ignore[operator]
    unrelated_ca_bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nnot a real cert\n-----END CERTIFICATE-----\n"
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(unrelated_ca_bundle))

    import app.core.llm.http_client as http_client_module

    certifi_where_calls: list[str] = []
    real_where = http_client_module.certifi.where

    def _spying_where() -> str:
        result = real_where()
        certifi_where_calls.append(result)
        return result

    monkeypatch.setattr(http_client_module.certifi, "where", _spying_where)

    client = build_provider_http_client_sync(
        ProviderConnectionInfo(
            api_base_url=f"http://{target_host}:{target_port}",
            allowlisted_hosts=frozenset({target_host}),
        )
    )
    assert client.trust_env is False

    with client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == "ok"
    # Zero requests through the proxy - the client never dialed it, not even
    # to have the CONNECT rejected.
    assert proxy_server.request_count == 0
    # The SSL context was built from certifi's real bundle path, never from
    # $SSL_CERT_FILE.
    assert certifi_where_calls, "expected _build_ssl_context to call certifi.where()"
    assert all(path != str(unrelated_ca_bundle) for path in certifi_where_calls)


def test_client_never_connects_to_proxy_host_even_via_pinned_backend(
    counting_proxy: tuple[str, _CountingProxyServer],
    target_server: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-suspenders: spy at the socket chokepoint itself
    (`PinnedNetworkBackendSync.connect_tcp`), not just at the proxy's own
    request counter, in case something in a future httpx/httpcore version
    changes how the proxy CONNECT itself is issued."""

    proxy_url, proxy_server = counting_proxy
    target_host, target_port = target_server
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("ALL_PROXY", proxy_url)

    connect_calls: list[tuple[str, int]] = []
    original = PinnedNetworkBackendSync.connect_tcp

    def _spying_connect_tcp(
        self: PinnedNetworkBackendSync, host: str, port: int, **kwargs: object
    ) -> object:
        connect_calls.append((host, port))
        return original(self, host, port, **kwargs)

    monkeypatch.setattr(PinnedNetworkBackendSync, "connect_tcp", _spying_connect_tcp)

    client = build_provider_http_client_sync(
        ProviderConnectionInfo(
            api_base_url=f"http://{target_host}:{target_port}",
            allowlisted_hosts=frozenset({target_host}),
        )
    )
    with client:
        client.get("/")

    assert connect_calls == [(target_host, target_port)]
    assert proxy_server.request_count == 0
