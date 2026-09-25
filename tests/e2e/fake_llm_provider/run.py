"""Entrypoint for the fake provider container.

Two ports, one process: plain HTTP on `FAKE_PROVIDER_HTTP_PORT` (default
8000) and, when `FAKE_PROVIDER_TLS_CERT_DIR` is set, TLS on
`FAKE_PROVIDER_TLS_PORT` (default 8443) using the CA-signed cert from
`tests/e2e/tls/generate_certs.py` for `fake-provider.test`. Registry-lifecycle
tests use the plain port; SSRF/DNS-rebinding tests that need a real TLS
handshake against a hostname use the TLS one.

uvicorn can only bind one (host, port, ssl) combination per `run()` call, so
TLS runs in a background thread while the main thread serves plain HTTP -
simplest thing that works for a single test-only process, no need for a
second container.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import uvicorn

from tests.e2e.fake_llm_provider.app import app


def _run_http(port: int) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _run_tls(port: int, cert_dir: Path) -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        ssl_certfile=str(cert_dir / "server.cert.pem"),
        ssl_keyfile=str(cert_dir / "server.key.pem"),
        log_level="info",
    )


def main() -> None:
    http_port = int(os.environ.get("FAKE_PROVIDER_HTTP_PORT", "8000"))
    tls_port = int(os.environ.get("FAKE_PROVIDER_TLS_PORT", "8443"))
    tls_cert_dir = os.environ.get("FAKE_PROVIDER_TLS_CERT_DIR")

    if tls_cert_dir:
        cert_dir = Path(tls_cert_dir)
        from tests.e2e.tls.generate_certs import generate

        generate(cert_dir)  # idempotent - no-op if certs already exist
        threading.Thread(target=_run_tls, args=(tls_port, cert_dir), daemon=True).start()

    _run_http(http_port)


if __name__ == "__main__":
    main()
