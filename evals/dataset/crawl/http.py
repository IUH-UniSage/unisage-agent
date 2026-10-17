"""HTTP client that enforces the crawl rules from the evaluation spec.

Every request goes through `PoliteClient.get`, so the rules cannot be skipped by
a caller: host allowlist (`*.iuh.edu.vn`), `robots.txt`, and at most one request
per `min_interval` seconds per host.

Several IUH hosts serve an incomplete TLS chain (missing intermediate), which
fails verification even though the site is genuine. For those hosts only, the
client retries without verification and records the host in `insecure_hosts`
so the manifest can say which files were fetched that way.
"""

import asyncio
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

USER_AGENT = "UniSageEvalBot/0.1 (academic RAG evaluation; polite crawl, 1 req/s)"
ROOT_DOMAIN = "iuh.edu.vn"


def is_allowed_host(host: str) -> bool:
    host = host.lower().split(":")[0]
    return host == ROOT_DOMAIN or host.endswith("." + ROOT_DOMAIN)


def parse_robots(
    status_code: int, content_type: str, body: str
) -> urllib.robotparser.RobotFileParser:
    """Build a parser from a robots.txt response.

    Missing robots.txt means "allow all". IUH hosts often answer `/robots.txt`
    with a redirect to an HTML page (status 200, text/html) - that is not a
    robots file either, so it is treated as missing rather than parsed as rules.
    """

    parser = urllib.robotparser.RobotFileParser()
    if status_code == 200 and content_type.startswith("text/plain"):
        parser.parse(body.splitlines())
    else:
        parser.parse([])
    return parser


class DisallowedUrlError(Exception):
    """The URL is outside the allowlist or blocked by robots.txt."""


@dataclass
class PoliteClient:
    min_interval: float = 1.0
    timeout: float = 30.0
    transport: httpx.AsyncBaseTransport | None = None
    insecure_hosts: set[str] = field(default_factory=set)
    _robots: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _last_request: dict[str, float] = field(default_factory=dict)
    _clients: dict[bool, httpx.AsyncClient] = field(default_factory=dict)

    def _client(self, verify: bool) -> httpx.AsyncClient:
        if verify not in self._clients:
            self._clients[verify] = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
                follow_redirects=True,
                verify=verify,
                transport=self.transport,
            )
        return self._clients[verify]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _wait_turn(self, host: str) -> None:
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request[host] = time.monotonic()

    async def _send(self, host: str, request: httpx.Request, *, stream: bool) -> httpx.Response:
        verify = host not in self.insecure_hosts
        client = self._client(verify)
        try:
            return await client.send(
                client.build_request(request.method, request.url), stream=stream
            )
        except httpx.ConnectError as exc:
            if not verify or "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                raise
            self.insecure_hosts.add(host)
            client = self._client(False)
            return await client.send(
                client.build_request(request.method, request.url), stream=stream
            )

    async def _robots_for(self, scheme: str, host: str) -> urllib.robotparser.RobotFileParser:
        if host not in self._robots:
            try:
                await self._wait_turn(host)
                request = httpx.Request("GET", f"{scheme}://{host}/robots.txt")
                response = await self._send(host, request, stream=False)
                self._robots[host] = parse_robots(
                    response.status_code,
                    response.headers.get("content-type", ""),
                    response.text,
                )
            except httpx.HTTPError:
                self._robots[host] = parse_robots(404, "", "")
        return self._robots[host]

    async def get(self, url: str, *, stream: bool = False) -> httpx.Response:
        """GET `url` under the crawl rules. With `stream=True` the caller must close it."""

        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.scheme not in ("http", "https") or not is_allowed_host(host):
            raise DisallowedUrlError(url)

        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            robots = await self._robots_for(parts.scheme, host)
            if not robots.can_fetch(USER_AGENT, url):
                raise DisallowedUrlError(url)
            await self._wait_turn(host)
            return await self._send(host, httpx.Request("GET", url), stream=stream)
