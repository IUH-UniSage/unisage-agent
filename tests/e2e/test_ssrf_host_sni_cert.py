"""Host header / TLS SNI / certificate validation, end to end through the real
factory client - todo.md Task 0.6's "Test Host/SNI/cert" item.

Runs entirely on loopback, no Docker: a real `ssl.SSLContext`-wrapped
`http.server.HTTPServer` on `127.0.0.1` stands in for the fake LLM provider,
and `app.core.ssrf_guard.resolve_all` is monkeypatched to resolve
`fake-provider.test` to `127.0.0.1` (there is no real DNS entry for it here -
the real-DNS case is `test_ssrf_rebinding.py`'s job; this file is about what
happens *after* resolution: does the client keep the original hostname for
`Host`/SNI/cert-hostname-checking the way `PinnedNetworkBackend`'s docstring
promises, and does `httpx`'s TLS verification actually reject a bad cert).

The client's own SSL context (`app.core.llm.http_client._build_ssl_context`)
only ever trusts `certifi`'s real bundle - never a test CA - by design (no
`verify=False` escape hatch exists to inject one). To exercise the *positive*
path this test does what the real integration harness's build step would do
outside the code under test: build a temporary CA bundle file that is
"certifi's real bundle, plus this test's throwaway CA appended", and
monkeypatch `certifi.where` (only in the `http_client` module, only for the
duration of the test) to return that file - i.e. it swaps which bundle
`ssl.create_default_context(cafile=...)` reads from, not the validation
logic itself.
"""

from __future__ import annotations

import http.server
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path

import certifi
import httpx
import pytest

from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from tests.e2e.tls import cert_variants

HOSTNAME = "fake-provider.test"
WRONG_HOSTNAME = "wrong-host.test"


class _CapturingHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        captured = self.server.captured  # type: ignore[attr-defined]
        captured["host_header"] = self.headers.get("Host")
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _CapturingHTTPServer(http.server.HTTPServer):
    captured: dict[str, str | None]


def _start_tls_server(
    cert_pem: bytes, key_pem: bytes, tmp_path: Path
) -> tuple[_CapturingHTTPServer, int, dict]:
    cert_path = tmp_path / "leaf.cert.pem"
    key_path = tmp_path / "leaf.key.pem"
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)

    captured: dict[str, str | None] = {"sni": None, "host_header": None}

    def _sni_callback(sock: ssl.SSLSocket, server_name: str | None, ctx: ssl.SSLContext) -> None:
        captured["sni"] = server_name

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    context.sni_callback = _sni_callback

    httpd = _CapturingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
    httpd.captured = captured
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    port = httpd.socket.getsockname()[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port, captured


def _stop(httpd: http.server.HTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def trusted_ca() -> cert_variants.GeneratedCa:
    return cert_variants.make_ca("unisage-agent test-ssrf-host-sni-cert CA")


@pytest.fixture
def rogue_ca() -> cert_variants.GeneratedCa:
    return cert_variants.make_ca("unisage-agent test-ssrf-host-sni-cert ROGUE CA")


@pytest.fixture
def trusted_bundle(trusted_ca: cert_variants.GeneratedCa, tmp_path: Path) -> Path:
    """certifi's real bundle + `trusted_ca` appended - see module docstring."""

    bundle_path = tmp_path / "bundle.pem"
    real_bundle = Path(certifi.where()).read_bytes()
    bundle_path.write_bytes(real_bundle + b"\n" + trusted_ca.cert_pem)
    return bundle_path


@pytest.fixture(autouse=True)
def _resolve_fake_provider_test_to_loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`fake-provider.test`/`wrong-host.test` have no real DNS entry here - point
    `resolve_all` at loopback for exactly those two names, real resolution for
    anything else (nothing else is looked up in this file, but fail-open-to-real
    is safer than silently redirecting unrelated hostnames)."""

    import app.core.ssrf_guard as ssrf_guard

    real_resolve_all = ssrf_guard.resolve_all

    def _fake_resolve_all(host: str, port: int) -> list[str]:
        if host in (HOSTNAME, WRONG_HOSTNAME):
            return ["127.0.0.1"]
        return real_resolve_all(host, port)

    monkeypatch.setattr(ssrf_guard, "resolve_all", _fake_resolve_all)
    yield


def _client_for(
    port: int, hostname: str, bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> httpx.Client:
    import app.core.llm.http_client as http_client_module

    monkeypatch.setattr(http_client_module.certifi, "where", lambda: str(bundle_path))
    return build_provider_http_client_sync(
        ProviderConnectionInfo(
            api_base_url=f"https://{hostname}:{port}",
            allowlisted_hosts=frozenset({hostname.lower()}),
        )
    )


def test_valid_cert_reaches_server_with_correct_host_and_sni(
    trusted_ca: cert_variants.GeneratedCa,
    trusted_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = cert_variants.make_leaf(trusted_ca, HOSTNAME)
    httpd, port, captured = _start_tls_server(leaf.cert_pem, leaf.key_pem, tmp_path)
    try:
        client = _client_for(port, HOSTNAME, trusted_bundle, monkeypatch)
        with client:
            response = client.get("/")
        assert response.status_code == 200
        assert response.text == "ok"
    finally:
        _stop(httpd)

    assert captured["sni"] == HOSTNAME
    assert captured["host_header"] == f"{HOSTNAME}:{port}"


def test_hostname_mismatch_cert_is_rejected(
    trusted_ca: cert_variants.GeneratedCa,
    trusted_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Trusted CA, but the leaf is for a DIFFERENT hostname than the one the
    # client is connecting to - check_hostname must catch this even though the
    # cert chain itself verifies fine.
    leaf = cert_variants.make_leaf(trusted_ca, WRONG_HOSTNAME)
    httpd, port, captured = _start_tls_server(leaf.cert_pem, leaf.key_pem, tmp_path)
    try:
        client = _client_for(port, HOSTNAME, trusted_bundle, monkeypatch)
        with client, pytest.raises(httpx.ConnectError):
            client.get("/")
    finally:
        _stop(httpd)

    # The handshake must fail before any HTTP response is readable - the
    # server's handler (which sets host_header) must never have run.
    assert captured["host_header"] is None


def test_expired_cert_is_rejected(
    trusted_ca: cert_variants.GeneratedCa,
    trusted_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    leaf = cert_variants.make_leaf(
        trusted_ca,
        HOSTNAME,
        not_before=now - datetime.timedelta(days=30),
        not_after=now - datetime.timedelta(days=1),
    )
    httpd, port, captured = _start_tls_server(leaf.cert_pem, leaf.key_pem, tmp_path)
    try:
        client = _client_for(port, HOSTNAME, trusted_bundle, monkeypatch)
        with client, pytest.raises(httpx.ConnectError):
            client.get("/")
    finally:
        _stop(httpd)

    assert captured["host_header"] is None


def test_wrong_ca_cert_is_rejected(
    rogue_ca: cert_variants.GeneratedCa,
    trusted_bundle: Path,  # trusted_bundle does NOT contain rogue_ca
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = cert_variants.make_leaf(rogue_ca, HOSTNAME)
    httpd, port, captured = _start_tls_server(leaf.cert_pem, leaf.key_pem, tmp_path)
    try:
        client = _client_for(port, HOSTNAME, trusted_bundle, monkeypatch)
        with client, pytest.raises(httpx.ConnectError):
            client.get("/")
    finally:
        _stop(httpd)

    assert captured["host_header"] is None
