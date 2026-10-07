"""Structured per-request tracing for the streaming graph.

Two separate concerns, both requested to replace the raw httpx/openai/
sqlalchemy DEBUG firehose that used to be the only way to see what a
request did:

1. `GraphTrace.node(name)` - always-on (independent of `settings.APP_DEBUG`),
   one INFO line per graph node a request actually passes through, carrying
   the identifiers needed to correlate it: conversation_id, the assistant
   message being generated, and who asked (user_id or "guest" + client IP).
2. `GraphTrace.prompt(node_name, text)` - only when `settings.APP_DEBUG` is
   true, dumps the per-request parts worth inspecting: the HyDE output
   used as the retrieval query, and for GenerationSynthesisNode only the
   `academic_metadata` and `prepared_context` blocks (not the static YAML
   around them).
"""

import logging
import time

from pydantic_ai.models import Model

from app.core.config import settings

logger = logging.getLogger("unisage.graph")

_NO_MODEL = "-"


def _model_name(model: Model | str | None) -> str:
    """`GraphModels`' fields are `Model | str` (a test double may pass a plain
    string), and most nodes don't call an LLM at all - normalizes all three
    cases to what's worth putting in a log line."""

    if model is None:
        return _NO_MODEL
    if isinstance(model, str):
        return model
    return model.model_name


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
        self._started_at = time.perf_counter()
        self._current: tuple[str, float] | None = None
        self._first_token_seen = False

    def node(self, name: str, *, model: Model | str | None = None) -> None:
        self._end_current_node()
        self._current = (name, time.perf_counter())
        logger.info(
            "node=%s model=%s conversation_id=%s message_id=%s user_id=%s ip=%s",
            name,
            _model_name(model),
            self._conversation_id,
            self._message_id,
            self._user_id,
            self._client_ip,
        )

    def first_token(self) -> None:
        if self._first_token_seen:
            return
        self._first_token_seen = True
        node_name, node_started_at = self._current or ("-", self._started_at)
        now = time.perf_counter()
        logger.info(
            "first_token node=%s ttft_ms=%.0f total_ms=%.0f conversation_id=%s message_id=%s",
            node_name,
            (now - node_started_at) * 1000,
            (now - self._started_at) * 1000,
            self._conversation_id,
            self._message_id,
        )

    def finish(self) -> None:
        self._end_current_node()
        logger.info(
            "graph_done total_ms=%.0f conversation_id=%s message_id=%s",
            (time.perf_counter() - self._started_at) * 1000,
            self._conversation_id,
            self._message_id,
        )

    def prompt(self, node_name: str, text: str) -> None:
        if not settings.APP_DEBUG:
            return
        logger.info(
            "prompt node=%s conversation_id=%s message_id=%s:\n%s",
            node_name,
            self._conversation_id,
            self._message_id,
            text,
        )

    def _end_current_node(self) -> None:
        if self._current is None:
            return
        name, started_at = self._current
        self._current = None
        logger.info(
            "node_done=%s elapsed_ms=%.0f conversation_id=%s message_id=%s",
            name,
            (time.perf_counter() - started_at) * 1000,
            self._conversation_id,
            self._message_id,
        )
