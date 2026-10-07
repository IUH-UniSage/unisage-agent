"""SecretRedactionFilter tests — plan.md "Secret redaction": the redaction
filter must scrub both a plain message and exception/traceback text before
any handler writes it out, and `configure_logging()` must not raise the noisy
provider loggers' levels back up."""

import io
import logging
import sys

from app.core.observability.logging_config import SecretRedactionFilter, configure_logging


def _make_record(msg: str, *args: object, exc_info: tuple | None = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


def test_filter_redacts_plain_message() -> None:
    record = _make_record(
        "failed calling provider: Authorization: Bearer sk-live-abcdefghij1234567890"
    )
    assert SecretRedactionFilter().filter(record) is True
    assert record.getMessage() == "failed calling provider: [REDACTED]"


def test_filter_redacts_percent_style_args() -> None:
    record = _make_record("key %s rejected", "sk-proj-abcdefghijklmnopqrstuvwx")
    assert SecretRedactionFilter().filter(record) is True
    assert record.getMessage() == "key [REDACTED] rejected"


def test_filter_redacts_exception_traceback() -> None:
    try:
        raise ValueError("bad key sk-proj-abcdefghijklmnopqrstuvwx")
    except ValueError:
        record = _make_record("boom", exc_info=sys.exc_info())

    assert SecretRedactionFilter().filter(record) is True
    assert record.exc_info is None
    assert "[REDACTED]" in record.exc_text
    assert "sk-proj-abcdefghijklmnopqrstuvwx" not in record.exc_text


def test_configure_logging_attaches_filter_to_root_handlers_exactly_once() -> None:
    configure_logging()
    configure_logging()  # idempotent - must not double-attach

    for handler in logging.getLogger().handlers:
        matching = [f for f in handler.filters if isinstance(f, SecretRedactionFilter)]
        assert len(matching) == 1, f"expected exactly one filter on {handler!r}, got {matching}"


def test_configure_logging_caps_noisy_provider_loggers_at_warning() -> None:
    configure_logging()
    for name in ("httpx", "httpcore", "openai", "anthropic"):
        assert logging.getLogger(name).level == logging.WARNING


def test_end_to_end_redaction_through_a_real_handler() -> None:
    """Builds an isolated logger + StreamHandler (independent of pytest's own
    log capture plumbing) with the real filter attached the same way
    `configure_logging()` attaches it, and proves a secret logged through it
    never reaches the handler's output stream."""

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactionFilter())

    logger = logging.getLogger("test.redaction.e2e")
    logger.propagate = False
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)

    logger.error("leaked: Authorization: Bearer sk-live-abcdefghij1234567890")

    output = stream.getvalue()
    assert "[REDACTED]" in output
    assert "sk-live-abcdefghij1234567890" not in output


def _graph_record(
    msg: str, *, name: str = "unisage.graph", level: int = logging.INFO
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1, msg=msg, args=None, exc_info=None
    )


def test_color_formatter_highlights_graph_trace_lines() -> None:
    from app.core.observability.logging_config import ColorFormatter

    formatter = ColorFormatter("%(message)s")

    assert (
        formatter.format(_graph_record("node=03_X model=-")) == "\033[1;36mnode=03_X model=-\033[0m"
    )
    assert formatter.format(_graph_record("node_done=03_X elapsed_ms=120")).startswith("\033[32m")
    assert formatter.format(_graph_record("node_done=03_X elapsed_ms=3324")).startswith(
        "\033[1;31m"
    )
    assert formatter.format(_graph_record("graph_done total_ms=1")).startswith("\033[1;35m")
    assert formatter.format(_graph_record("plain", name="app.other")) == "plain"


def test_color_formatter_colors_only_the_header_of_a_prompt_dump() -> None:
    from app.core.observability.logging_config import ColorFormatter

    line = ColorFormatter("%(message)s").format(_graph_record('prompt node=03_X:\n{"tasks": []}'))

    assert line == '\033[33mprompt node=03_X:\033[0m\n{"tasks": []}'
