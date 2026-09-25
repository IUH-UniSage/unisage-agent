"""A tiny authoritative DNS server for the `rebind.test` DNS-rebinding test.

plan.md "SSRF policy": syntax/allowlist checks at save time are not enough - a
hostname can resolve to a safe IP when Java validates it and a different
(internal) IP by the time Python actually connects. The only real test for
that is a real DNS server that changes its answer between queries, with the
agent's resolver actually pointed at it (compose `dns:`), exercising the OS
resolver - not a mocked one.

Behavior for `rebind.test` (A record only):
- 1st query for a given client address: returns the fake LLM provider's IP.
- every query after that (same or different client): returns the decoy
  "internal" service's IP.
- TTL 0 on every answer, so nothing caches the first answer past that query.

Any other name gets NXDOMAIN. This is deliberately not a general resolver -
it only ever answers for the one hostname the SSRF rebinding test cares
about.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

from dnslib import QTYPE, RR, A, DNSRecord
from dnslib.server import DNSServer

logger = logging.getLogger("rebinding_dns")

REBIND_HOSTNAME = "rebind.test."


class RebindingResolver:
    """Answers `rebind.test`: fake-provider IP first, decoy IP on every later query.

    One counter for the whole server process (not per-client) - the point of
    the test is "does the agent's *second* connection attempt (from
    connection reuse, a retry, or a second worker) still land on the verified
    IP", so it must flip after the very first answer regardless of who asks.
    """

    def __init__(self, safe_ip: str, decoy_ip: str) -> None:
        self.safe_ip = safe_ip
        self.decoy_ip = decoy_ip
        self._lock = threading.Lock()
        self._query_count = 0

    def resolve(self, request: DNSRecord, handler: object) -> DNSRecord:
        reply = request.reply()
        qname = request.q.qname
        qtype = QTYPE[request.q.qtype]

        if str(qname).lower() != REBIND_HOSTNAME or qtype not in ("A", "ANY"):
            reply.header.rcode = 3  # NXDOMAIN
            return reply

        with self._lock:
            self._query_count += 1
            answer_ip = self.safe_ip if self._query_count == 1 else self.decoy_ip

        logger.info(
            "rebind.test query #%d -> %s", self._query_count, answer_ip
        )
        reply.add_answer(RR(qname, QTYPE.A, rdata=A(answer_ip), ttl=0))
        return reply

    def reset(self) -> None:
        with self._lock:
            self._query_count = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("DNS_PORT", "53")))
    parser.add_argument(
        "--safe-ip", default=os.environ.get("REBIND_SAFE_IP", "127.0.0.1"),
        help="IP returned on the first query - the fake LLM provider's address",
    )
    parser.add_argument(
        "--decoy-ip", default=os.environ.get("REBIND_DECOY_IP", "127.0.0.2"),
        help="IP returned on every subsequent query - the decoy 'internal' service",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    resolver = RebindingResolver(safe_ip=args.safe_ip, decoy_ip=args.decoy_ip)
    server = DNSServer(resolver, port=args.port, address="0.0.0.0", tcp=False)
    logger.info(
        "rebinding DNS server listening on :%d (safe=%s decoy=%s)",
        args.port, args.safe_ip, args.decoy_ip,
    )
    server.start()


if __name__ == "__main__":
    main()
