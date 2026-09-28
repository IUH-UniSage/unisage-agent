"""SSRF guard for provider URLs — plan.md "SSRF policy".

Two checks, always run in this order: syntax (no network) then resolved-IP
range (needs DNS). `PinnedNetworkBackend` is the actual defense against DNS
rebinding — it resolves once and connects to that exact IP, at the socket
layer, never re-resolving for the real connection.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import anyio.to_thread
import httpcore

ALLOWED_SCHEMES = {"http", "https"}
MAX_URL_LENGTH = 2048
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x1f\x7f\s]")

# Same set as backend-java's SsrfGuard.BLOCKED_RANGES — keep in sync (see
# contracts/ssrf-url-vectors.json for the shared test vectors).
_BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "fe80::/10",
        "100.64.0.0/10",
        "fd00::/8",
        "224.0.0.0/4",
        "ff00::/8",
        "0.0.0.0/8",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "240.0.0.0/4",
    )
]


class SsrfSyntaxError(ValueError):
    """URL failed the syntax pass — carries a machine-readable `reason`."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class SsrfBlockedError(Exception):
    """Resolved IP (or the connection actually made) falls in a blocked range — always PERMANENT."""

    def __init__(self, reason: str, host: str | None = None) -> None:
        self.reason = reason
        self.host = host
        super().__init__(f"{reason} host={host}")


def validate_syntax(url: str) -> str:
    """Syntax-only pass. Returns the validated (IDNA A-label, trailing-dot-stripped) host."""

    if not url:
        raise SsrfSyntaxError("URL_EMPTY")
    if len(url) > MAX_URL_LENGTH:
        raise SsrfSyntaxError("URL_TOO_LONG")
    if _CONTROL_OR_SPACE.search(url) or "\\" in url:
        raise SsrfSyntaxError("CONTROL_CHAR_OR_WHITESPACE")

    parts = urlsplit(url)

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise SsrfSyntaxError("SCHEME_NOT_ALLOWED")
    if "@" in parts.netloc:
        raise SsrfSyntaxError("USERINFO_NOT_ALLOWED")
    if parts.query:
        raise SsrfSyntaxError("QUERY_NOT_ALLOWED")
    if parts.fragment:
        raise SsrfSyntaxError("FRAGMENT_NOT_ALLOWED")

    host = parts.hostname
    if not host:
        raise SsrfSyntaxError("HOST_EMPTY")
    if "%" in host:
        raise SsrfSyntaxError("URL_SYNTAX_INVALID")

    try:
        port = parts.port
    except ValueError as exc:
        raise SsrfSyntaxError("PORT_NOT_NUMERIC") from exc
    if port is not None and not (1 <= port <= 65535):
        raise SsrfSyntaxError("PORT_OUT_OF_RANGE")

    normalized_host = host[:-1] if host.endswith(".") else host
    try:
        a_label_host = normalized_host.encode("idna").decode("ascii")
        # Python's idna codec passes an already-ASCII label through unvalidated (e.g. a bare
        # "xn--" with no encoded content) — round-trip decode to actually check punycode shape.
        a_label_host.encode("ascii").decode("idna")
    except UnicodeError as exc:
        raise SsrfSyntaxError("HOST_IDNA_INVALID") from exc
    return a_label_host


def _unwrap_ipv4_mapped(addr: ipaddress.IPv6Address | ipaddress.IPv4Address):
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def is_blocked(ip_str: str) -> bool:
    addr = _unwrap_ipv4_mapped(ipaddress.ip_address(ip_str))
    return any(addr in network for network in _BLOCKED_NETWORKS if addr.version == network.version)


def resolve_all(host: str, port: int) -> list[str]:
    """All A/AAAA IPs for host — blocking `getaddrinfo`, called from a thread by callers."""

    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    seen: list[str] = []
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in seen:
            seen.append(ip)
    return seen


def validate_resolved(host: str, port: int, allowlist: frozenset[str]) -> str:
    """Resolves `host`, verifies every address, and returns the one IP to actually connect to.

    Fails closed: any unresolvable host, or any resolved address in a blocked range while the
    host isn't explicitly allowlisted, is rejected — never "pick the first safe-looking one and
    ignore the rest".
    """

    try:
        addresses = resolve_all(host, port)
    except OSError as exc:
        raise SsrfBlockedError("DNS_RESOLUTION_FAILED", host) from exc
    if not addresses:
        raise SsrfBlockedError("DNS_RESOLUTION_FAILED", host)

    explicitly_allowed = host.lower() in allowlist
    for ip in addresses:
        if is_blocked(ip) and not explicitly_allowed:
            raise SsrfBlockedError("RESOLVED_IP_BLOCKED", host)

    return addresses[0]


@dataclass
class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Wraps a real `httpcore.AsyncNetworkBackend`, pinning `connect_tcp` to a verified IP.

    Resolves `host` once here, checks every address against the same blocked ranges as
    `validate_resolved`, then connects the real backend to that literal IP — never lets the
    underlying connect do its own (second, re-race-able) DNS lookup. The request layer above
    this still uses the original hostname for the `Host` header, TLS SNI, and certificate
    hostname checking — this backend never rewrites those, only the socket target.
    """

    allowlist: frozenset[str] = field(default_factory=frozenset)
    _delegate: httpcore.AsyncNetworkBackend = field(default_factory=httpcore.AnyIOBackend)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        # getaddrinfo is blocking — never call it directly on the event loop thread.
        verified_ip = await anyio.to_thread.run_sync(validate_resolved, host, port, self.allowlist)
        return await self._delegate.connect_tcp(
            verified_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError("Unix sockets are never used for provider calls")


@dataclass
class PinnedNetworkBackendSync(httpcore.NetworkBackend):
    """Sync counterpart of `PinnedNetworkBackend`, for provider call sites that are
    themselves sync (Celery tasks — see
    `app/core/llm/http_client.py::build_provider_http_client_sync`).

    `socket.getaddrinfo` already blocks the calling thread either way, so unlike the
    async version there's no event loop to protect by offloading to a thread — the
    resolve happens inline.
    """

    allowlist: frozenset[str] = field(default_factory=frozenset)
    _delegate: httpcore.NetworkBackend = field(default_factory=httpcore.SyncBackend)

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.NetworkStream:
        verified_ip = validate_resolved(host, port, self.allowlist)
        return self._delegate.connect_tcp(
            verified_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError("Unix sockets are never used for provider calls")
