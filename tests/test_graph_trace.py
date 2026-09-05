import logging

import pytest

from app.core.config import settings
from app.core.graph_trace import GraphTrace


def _trace() -> GraphTrace:
    return GraphTrace(
        conversation_id="conv-1", message_id="msg-1", user_id="user-1", client_ip="1.2.3.4"
    )


def test_node_always_logs_regardless_of_debug_flag(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "DEBUG", False)
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        _trace().node("01_GreetingDetectionNode")

    assert "node=01_GreetingDetectionNode" in caplog.text
    assert "conversation_id=conv-1" in caplog.text
    assert "message_id=msg-1" in caplog.text
    assert "user_id=user-1" in caplog.text
    assert "ip=1.2.3.4" in caplog.text


def test_node_falls_back_to_guest_and_dash_when_unset() -> None:
    trace = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)
    with_caplog = logging.getLogger("unisage.graph")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    with_caplog.addHandler(handler)
    try:
        trace.node("01_GreetingDetectionNode")
    finally:
        with_caplog.removeHandler(handler)

    assert "user_id=guest" in records[0]
    assert "ip=-" in records[0]


def test_prompt_logs_only_when_debug_true(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        monkeypatch.setattr(settings, "DEBUG", False)
        _trace().prompt("12_GenerationSynthesisNode", "the full system prompt")
        assert "the full system prompt" not in caplog.text

        monkeypatch.setattr(settings, "DEBUG", True)
        _trace().prompt("12_GenerationSynthesisNode", "the full system prompt")
        assert "the full system prompt" in caplog.text
        assert "node=12_GenerationSynthesisNode" in caplog.text
