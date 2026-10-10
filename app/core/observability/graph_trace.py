"""Structured per-request tracing for the streaming graph.

Two separate concerns, both requested to replace the raw httpx/openai/
sqlalchemy DEBUG firehose that used to be the only way to see what a
request did:

1. `GraphTrace.node(name)` - always-on (independent of `settings.APP_DEBUG`),
   one INFO line per graph node a request actually passes through, carrying
   the identifiers needed to correlate it: conversation_id, the assistant
   message being generated, and who asked (user_id or "guest" + client IP).
   An LLM node's line also names the model, the thinking level asked for and
   the credential; a failover inside the node (`GraphTrace.model_switch`, reached
   from `app.graph.streaming` through `active_trace()`) logs one more `node=`
   line for the same node with the replacement.
2. `GraphTrace.prompt(node_name, text)` - only when `settings.APP_DEBUG` is
   true, dumps the per-request parts worth inspecting: the HyDE output
   used as the retrieval query, and for GenerationSynthesisNode only the
   `academic_metadata` and `prepared_context` blocks (not the static YAML
   around them).
"""

import logging
import time
from contextvars import ContextVar, Token

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig

logger = logging.getLogger("unisage.graph")

_NO_MODEL = "-"

_active_trace: ContextVar["GraphTrace | None"] = ContextVar("graph_trace", default=None)


def bind_trace(trace: "GraphTrace") -> Token["GraphTrace | None"]:
    """Makes `trace` what `active_trace()` returns for the rest of this request's task."""

    return _active_trace.set(trace)


def unbind_trace(token: Token["GraphTrace | None"]) -> None:
    _active_trace.reset(token)


def active_trace() -> "GraphTrace | None":
    """The trace of the graph request running in this task, if any - lets the shared LLM
    helpers log a failover without every node threading the trace through to them."""

    return _active_trace.get()


def _model_name(model: Model | str | None) -> str:
    """`GraphModels`' fields are `Model | str` (a test double may pass a plain
    string), and most nodes don't call an LLM at all - normalizes all three
    cases to what's worth putting in a log line."""

    if model is None:
        return _NO_MODEL
    if isinstance(model, str):
        return model
    return model.model_name


def _thinking_label(agent: Agent[None, str] | None) -> str:
    """The `thinking` setting the agent sends: `default` when it sends none (the model
    reasons at its own default level), `off` for False."""

    if agent is None:
        return _NO_MODEL
    model_settings = agent.model_settings
    if callable(model_settings):
        # Resolved per run from the RunContext - not knowable before the call.
        return "per-run"
    thinking = (model_settings or {}).get("thinking")
    if thinking is None:
        return "default"
    if thinking is False:
        return "off"
    if thinking is True:
        return "on"
    return str(thinking)


def _credential_label(credential: CredentialConfig | None) -> str:
    if credential is None:
        return _NO_MODEL
    return credential.display_name or credential.id


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

    def node(
        self,
        name: str,
        *,
        agent: Agent[None, str] | None = None,
        credential: CredentialConfig | None = None,
    ) -> None:
        self._end_current_node()
        self._current = (name, time.perf_counter())
        logger.info(
            "node=%s model=%s thinking=%s credential=%s conversation_id=%s message_id=%s "
            "user_id=%s ip=%s",
            name,
            _model_name(agent.model if agent is not None else None),
            _thinking_label(agent),
            _credential_label(credential),
            self._conversation_id,
            self._message_id,
            self._user_id,
            self._client_ip,
        )

    def model_switch(
        self,
        agent: Agent[None, str],
        credential: CredentialConfig,
        *,
        failed_credential: CredentialConfig | None,
        reason: str,
    ) -> None:
        """One more `node=` line for the node running now: it moved off `failed_credential`
        (for `reason`) onto `credential`/`agent`."""

        node_name = self._current[0] if self._current is not None else "-"
        logger.warning(
            "node=%s model=%s thinking=%s credential=%s failover_from=%s reason=%s "
            "conversation_id=%s message_id=%s",
            node_name,
            _model_name(agent.model),
            _thinking_label(agent),
            _credential_label(credential),
            _credential_label(failed_credential),
            reason,
            self._conversation_id,
            self._message_id,
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
