"""Runs inside the `ssrf-probe` image only (see ../Dockerfile) - not imported
by anything else.

Makes exactly one real outbound GET through `build_provider_http_client_sync`
(the same factory every provider call site uses) against a URL whose host is
resolved by the container's own DNS resolver (set via `docker run --dns` by
`test_ssrf_rebinding.py`) - a real OS-level resolution, not a monkeypatched
one. Prints one JSON line to stdout describing the outcome so the test
(driving this via the `docker` SDK, reading container logs) doesn't have to
parse anything more fragile than that.
"""

from __future__ import annotations

import json
import sys
from urllib.parse import urlsplit

sys.path.insert(0, "/srv")

from app.core.llm.http_client import (
    ProviderConnectionInfo,
    build_provider_http_client_sync,
)
from app.core.ssrf_guard import SsrfBlockedError


def main() -> None:
    url = sys.argv[1]
    allowlist_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    allowlist = frozenset(h for h in allowlist_arg.split(",") if h)

    parts = urlsplit(url)
    base_url = f"{parts.scheme}://{parts.hostname}:{parts.port}"
    client = build_provider_http_client_sync(
        ProviderConnectionInfo(api_base_url=base_url, allowlisted_hosts=allowlist)
    )
    try:
        with client:
            response = client.get(parts.path or "/")
        result = {"outcome": "reached", "status_code": response.status_code}
    except SsrfBlockedError as exc:
        result = {"outcome": "blocked", "reason": exc.reason}
    except Exception as exc:
        result = {"outcome": "error", "type": type(exc).__name__, "detail": str(exc)}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
