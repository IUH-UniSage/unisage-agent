"""Central logging setup — plan.md "Secret redaction".

Two independent protections live here:

  - Provider SDKs (httpx/httpcore/openai/anthropic) log request headers/bodies
    at DEBUG level by default — plan.md's "Không log request body/headers của
    lời gọi provider" rule. This module caps those loggers at WARNING so a
    stray `logging.getLogger("openai").setLevel(logging.DEBUG)` elsewhere, or
    a library's own default, can't leak an `Authorization` header into the log
    stream. `settings.APP_DEBUG` only controls this service's *own* verbose
    output (see `app/core/observability/graph_trace.py`'s prompt dump) — it must never raise
    these levels back up.
  - `SecretRedactionFilter`, attached to every handler on the root logger
    (not to the root *logger* object itself — see note below), runs
    `app.core.security.redaction.redact` over every record's formatted message *and*
    its exception traceback (`exc_info`) / `stack_info` before that handler
    writes it out — the last line of defense for anything that slips past the
    level cap above or a `safe_error_message()` call site a developer forgot.

Call `configure_logging()` once, as early as possible, in each process
entrypoint (`app/main.py` for the API/ASGI process, `app/worker/celery_app.py`
for Celery worker/beat).

Note on *why* the filter goes on handlers, not the root Logger object:
`logging.Logger.filter()` only runs a logger's own filters for records that
*originate* at that exact logger — a child logger's `logging.getLogger(__name__)`
calls never consult an ancestor's `addFilter()`. Records only universally pass
through a *handler's* `filter()` as they propagate up the hierarchy, so
attaching to root's handlers (which `basicConfig()` sets up, and where nearly
every module's records eventually land) is what actually redacts everything.
"""

from __future__ import annotations

import logging
import re
import traceback
from contextvars import ContextVar, Token

from app.core.config import settings
from app.core.security.redaction import redact

# Silenced because they can log request headers/bodies (may contain the
# provider API key) at DEBUG — plan.md "Secret redaction".
_NOISY_PROVIDER_LOGGERS = (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "openai",
    "openai._base_client",
    "anthropic",
    "anthropic._base_client",
)
# Silenced for readability only, not secrecy — kept in its own tuple so it's
# obvious which loggers above are there *because* of the redaction rule.
_OTHER_NOISY_LOGGERS = ("sqlalchemy.engine", "asyncio")

# `trace_id` is the chat turn's request id (`TraceIdFilter`), "-" outside a turn: grep one
# id to get every line of the turn that produced an answer (shown on its staff ticket).
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(trace_id)s]: %(message)s"
_NO_TRACE_ID = "-"

_trace_id: ContextVar[str] = ContextVar("log_trace_id", default=_NO_TRACE_ID)

_RESET = "\033[0m"
_BOLD_CYAN = "\033[1;36m"
_GREEN = "\033[32m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_MAGENTA = "\033[1;35m"
_BOLD_BLUE = "\033[1;34m"
_BRIGHT_BLUE = "\033[94m"
_DIM = "\033[2m"
# Yellow stays reserved for warnings; bold red marks errors and, by request, every retrieval
# cache line (hit, miss, miss_first, miss_second, stale, bypass) plus a graph node served from
# the cache (`cached=True`), so a turn's cache story stands out when scrolling the log.
_YELLOW = "\033[33m"
_BOLD_RED = "\033[1;31m"
_RETRIEVAL_CACHE_LOGGER = "app.graph.nodes.retrieval_cache"
_LEVEL_COLORS = {logging.WARNING: _YELLOW, logging.ERROR: _BOLD_RED, logging.CRITICAL: _BOLD_RED}
_SLOW_NODE_MS = 2000
# Green: the web search lines that say whether Tavily was answered from the cache - each
# lookup with its key (bold on a hit), each search's pages and the turn's summary.
_WEB_SEARCH_PREFIXES: dict[str, tuple[str, ...]] = {
    "app.graph.nodes.web_search": ("Tavily cache ", "Web search "),
    "app.graph.streaming_graph": ("Web search turn:",),
}
_ELAPSED_PATTERN = re.compile(r"elapsed_ms=(\d+)")


class SecretRedactionFilter(logging.Filter):
    """Redacts a record's formatted message and any exception/stack text.

    Mutates the record in place and always returns `True` (never drops a
    record) — attach to the root *logger* (`logging.getLogger().addFilter`),
    not to individual handlers, so it runs exactly once per record regardless
    of how many handlers are configured.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive, malformed % args
            message = str(record.msg)
        # truncate=False: same reasoning as the traceback branch below - a
        # local console log line isn't bounded storage. This specifically
        # matters for `graph_trace.py`'s debug-only prompt dump
        # (APP_DEBUG=true's whole reason to exist is inspecting the full
        # prepared_context/academic_metadata blocks), which the 500-char cap
        # was silently truncating mid-word.
        record.msg = redact(message, truncate=False)
        record.args = None

        if record.exc_info:
            raw_traceback = "".join(traceback.format_exception(*record.exc_info))
            # truncate=False: this is a local console log line, not bounded
            # storage (DB/Slack/HTTP response) - the 500-char cap `redact()`
            # applies by default for those exists so a request body can't
            # blow up storage, but on a traceback it just chops off the
            # exception type/message that's almost always at the very end.
            record.exc_text = redact(raw_traceback, truncate=False)
            # Clear exc_info so the handler's default Formatter doesn't also
            # render the raw (unredacted) traceback from it — exc_text above
            # is what actually gets appended to the output once set.
            record.exc_info = None

        stack_info = getattr(record, "stack_info", None)
        if stack_info:
            record.stack_info = redact(str(stack_info), truncate=False)

        return True


def bind_trace_id(trace_id: str) -> Token[str]:
    """Stamps `trace_id` on every record logged from this task (and the tasks it starts)."""

    return _trace_id.set(trace_id)


def unbind_trace_id(token: Token[str]) -> None:
    _trace_id.reset(token)


class TraceIdFilter(logging.Filter):
    """Sets `record.trace_id` for `_LOG_FORMAT`. On the handler like `SecretRedactionFilter`,
    so it also covers a record built without the record factory (`makeLogRecord`); a handler
    runs in the logging call's own context, so the contextvar is the caller's."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = _trace_id.get()
        return True


class ColorFormatter(logging.Formatter):
    """Highlights the graph trace (`unisage.graph`) so a turn reads at a glance:
    node starts cyan, node timings green (magenta when slow), debug dumps blue; the
    web search / Tavily cache lines green (a cache hit bold); and the retrieval cache
    lines (`retrieval_cache layer=... outcome=...`) and nodes skipped thanks to it
    (`node=... cached=True`) bold red."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        color = self._color_of(record)
        if color is None:
            return line
        if record.name == "unisage.graph" and record.getMessage().startswith("prompt "):
            header, _, body = line.partition("\n")
            return f"{color}{header}{_RESET}\n{body}" if body else f"{color}{line}{_RESET}"
        return f"{color}{line}{_RESET}"

    @staticmethod
    def _color_of(record: logging.LogRecord) -> str | None:
        if record.levelno in _LEVEL_COLORS:
            return _LEVEL_COLORS[record.levelno]
        message = record.getMessage()
        if record.name == "google_genai.models":
            return _DIM
        if record.name == _RETRIEVAL_CACHE_LOGGER and message.startswith("retrieval_cache "):
            return _BOLD_RED
        if message.startswith(_WEB_SEARCH_PREFIXES.get(record.name, ())):
            return _BOLD_GREEN if message.startswith("Tavily cache HIT") else _GREEN
        if record.name != "unisage.graph":
            return None
        if message.startswith("node_done="):
            elapsed = _ELAPSED_PATTERN.search(message)
            slow = elapsed is not None and int(elapsed.group(1)) >= _SLOW_NODE_MS
            return _BOLD_MAGENTA if slow else _GREEN
        if message.startswith("node="):
            return _BOLD_RED if " cached=True" in message else _BOLD_CYAN
        if message.startswith(("first_token", "graph_done")):
            return _BOLD_BLUE
        if message.startswith("prompt "):
            return _BRIGHT_BLUE
        return None


def configure_logging() -> None:
    """Safe to call repeatedly / from multiple entrypoints (e.g. a module
    imported by both the ASGI app and a Celery task): `basicConfig()` no-ops
    once root already has a handler, the noisy-logger level-setting is
    naturally idempotent, and the filter-attach loop below only adds a filter
    to a handler that doesn't already have one — so a *new* handler added to
    root between calls (as pytest's logging plugin does) still gets covered
    on the next call, unlike a one-shot "already configured" guard would."""

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    for handler in logging.getLogger().handlers:
        if not any(isinstance(existing, SecretRedactionFilter) for existing in handler.filters):
            handler.addFilter(SecretRedactionFilter())
        if not any(isinstance(existing, TraceIdFilter) for existing in handler.filters):
            handler.addFilter(TraceIdFilter())
        if settings.LOG_COLOR and _writes_to_terminal(handler):
            handler.setFormatter(ColorFormatter(_LOG_FORMAT))

    for name in (*_NOISY_PROVIDER_LOGGERS, *_OTHER_NOISY_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)


def _writes_to_terminal(handler: logging.Handler) -> bool:
    stream = getattr(handler, "stream", None)
    return isinstance(handler, logging.StreamHandler) and bool(
        stream is not None and hasattr(stream, "isatty") and stream.isatty()
    )
