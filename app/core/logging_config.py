"""Central logging setup — plan.md "Secret redaction".

Two independent protections live here:

  - Provider SDKs (httpx/httpcore/openai/anthropic) log request headers/bodies
    at DEBUG level by default — plan.md's "Không log request body/headers của
    lời gọi provider" rule. This module caps those loggers at WARNING so a
    stray `logging.getLogger("openai").setLevel(logging.DEBUG)` elsewhere, or
    a library's own default, can't leak an `Authorization` header into the log
    stream. `settings.APP_DEBUG` only controls this service's *own* verbose
    output (see `app/core/graph_trace.py`'s prompt dump) — it must never raise
    these levels back up.
  - `SecretRedactionFilter`, attached to every handler on the root logger
    (not to the root *logger* object itself — see note below), runs
    `app.core.redaction.redact` over every record's formatted message *and*
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
import traceback

from app.core.redaction import redact

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
        record.msg = redact(message)
        record.args = None

        if record.exc_info:
            raw_traceback = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact(raw_traceback)
            # Clear exc_info so the handler's default Formatter doesn't also
            # render the raw (unredacted) traceback from it — exc_text above
            # is what actually gets appended to the output once set.
            record.exc_info = None

        stack_info = getattr(record, "stack_info", None)
        if stack_info:
            record.stack_info = redact(str(stack_info))

        return True


def configure_logging() -> None:
    """Safe to call repeatedly / from multiple entrypoints (e.g. a module
    imported by both the ASGI app and a Celery task): `basicConfig()` no-ops
    once root already has a handler, the noisy-logger level-setting is
    naturally idempotent, and the filter-attach loop below only adds a filter
    to a handler that doesn't already have one — so a *new* handler added to
    root between calls (as pytest's logging plugin does) still gets covered
    on the next call, unlike a one-shot "already configured" guard would."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for handler in logging.getLogger().handlers:
        if not any(isinstance(existing, SecretRedactionFilter) for existing in handler.filters):
            handler.addFilter(SecretRedactionFilter())

    for name in (*_NOISY_PROVIDER_LOGGERS, *_OTHER_NOISY_LOGGERS):
        logging.getLogger(name).setLevel(logging.WARNING)
