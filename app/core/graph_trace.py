"""Structured per-request tracing for the streaming graph (T6.1).

Two separate concerns, both requested to replace the raw httpx/openai/
sqlalchemy DEBUG firehose that used to be the only way to see what a
request did:

1. `GraphTrace.node(name)` - always-on (independent of `settings.DEBUG`),
   one INFO line per graph node a request actually passes through, carrying
   the identifiers needed to correlate it: conversation_id, the assistant
   message being generated, and who asked (user_id or "guest" + client IP).
2. `GraphTrace.prompt(node_name, text)` - only when `settings.DEBUG` is
   true, dumps the exact prompt sent to an LLM node after it's built.
"""

import logging

from app.core.config import settings

logger = logging.getLogger("unisage.graph")


class GraphTrace:
    """One instance per `/chat/stream` request, threaded into `run_graph`."""

    def __init__(
        self,
        *,
        conversation_id: str,
        message_id: str,
        user_id: str | None,
        client_ip: str | None,
    ) -> None:
        self._conversation_id = conversation_id
        self._message_id = message_id
        self._user_id = user_id or "guest"
        self._client_ip = client_ip or "-"

    def node(self, name: str) -> None:
        logger.info(
            "node=%s conversation_id=%s message_id=%s user_id=%s ip=%s",
            name,
            self._conversation_id,
            self._message_id,
            self._user_id,
            self._client_ip,
        )

    def prompt(self, node_name: str, text: str) -> None:
        if not settings.DEBUG:
            return
        logger.info(
            "prompt node=%s conversation_id=%s message_id=%s:\n%s",
            node_name,
            self._conversation_id,
            self._message_id,
            text,
        )
