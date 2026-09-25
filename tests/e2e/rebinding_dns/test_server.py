"""Local (no docker) tests for the rebinding DNS resolver - binds a real UDP
socket on an ephemeral port and queries it with `dnslib`, but stays inside a
single test process. Not marked `integration`: nothing here needs the compose
stack.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
from dnslib import DNSRecord
from dnslib.server import DNSServer

from tests.e2e.rebinding_dns.server import RebindingResolver

SAFE_IP = "10.10.10.10"
DECOY_IP = "10.10.10.20"


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server() -> Iterator[tuple[DNSServer, RebindingResolver, int]]:
    port = _free_udp_port()
    resolver = RebindingResolver(safe_ip=SAFE_IP, decoy_ip=DECOY_IP)
    server = DNSServer(resolver, port=port, address="127.0.0.1", tcp=False)
    server.start_thread()
    # start_thread() returns immediately after spawning the thread; give the
    # UDP socket a moment to actually be listening before the first query.
    time.sleep(0.1)
    try:
        yield server, resolver, port
    finally:
        server.stop()


def _query_a(port: int) -> str:
    q = DNSRecord.question("rebind.test", "A")
    resp_bytes = q.send("127.0.0.1", port, timeout=2)
    reply = DNSRecord.parse(resp_bytes)
    assert reply.rr, "expected at least one answer record"
    return str(reply.rr[0].rdata)


def test_first_query_returns_safe_ip(
    running_server: tuple[DNSServer, RebindingResolver, int],
) -> None:
    _server, _resolver, port = running_server
    assert _query_a(port) == SAFE_IP


def test_second_and_later_queries_return_decoy_ip(
    running_server: tuple[DNSServer, RebindingResolver, int],
) -> None:
    _server, _resolver, port = running_server
    assert _query_a(port) == SAFE_IP
    assert _query_a(port) == DECOY_IP
    assert _query_a(port) == DECOY_IP


def test_ttl_is_zero_on_every_answer(
    running_server: tuple[DNSServer, RebindingResolver, int],
) -> None:
    _server, _resolver, port = running_server
    q = DNSRecord.question("rebind.test", "A")
    reply = DNSRecord.parse(q.send("127.0.0.1", port, timeout=2))
    assert reply.rr[0].ttl == 0


def test_unknown_hostname_is_nxdomain(
    running_server: tuple[DNSServer, RebindingResolver, int],
) -> None:
    _server, _resolver, port = running_server
    q = DNSRecord.question("something-else.test", "A")
    reply = DNSRecord.parse(q.send("127.0.0.1", port, timeout=2))
    assert reply.header.rcode == 3


def test_reset_makes_next_query_safe_again(
    running_server: tuple[DNSServer, RebindingResolver, int],
) -> None:
    _server, resolver, port = running_server
    assert _query_a(port) == SAFE_IP
    assert _query_a(port) == DECOY_IP
    resolver.reset()
    assert _query_a(port) == SAFE_IP


def test_resolve_is_thread_safe_under_concurrent_queries() -> None:
    # Not through the socket - directly hammers `resolve()` from many threads
    # to check the counter itself doesn't race (the fixture above already
    # covers the "real UDP round trip" side).
    resolver = RebindingResolver(safe_ip=SAFE_IP, decoy_ip=DECOY_IP)
    q = DNSRecord.question("rebind.test", "A")

    results: list[str] = []
    results_lock = threading.Lock()

    def _hit() -> None:
        reply = resolver.resolve(q, None)
        with results_lock:
            results.append(str(reply.rr[0].rdata))

    threads = [threading.Thread(target=_hit) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(SAFE_IP) == 1
    assert results.count(DECOY_IP) == 49
