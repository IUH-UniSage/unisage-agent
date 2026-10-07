import logging

import pytest

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace


def _trace() -> GraphTrace:
    return GraphTrace(
        conversation_id="conv-1", message_id="msg-1", user_id="user-1", client_ip="1.2.3.4"
    )


def test_node_always_logs_regardless_of_debug_flag(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "APP_DEBUG", False)
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
        monkeypatch.setattr(settings, "APP_DEBUG", False)
        _trace().prompt("10_GenerationSynthesisNode", "the full system prompt")
        assert "the full system prompt" not in caplog.text

        monkeypatch.setattr(settings, "APP_DEBUG", True)
        _trace().prompt("10_GenerationSynthesisNode", "the full system prompt")
        assert "the full system prompt" in caplog.text
        assert "node=10_GenerationSynthesisNode" in caplog.text


def test_timing_logs_node_elapsed_first_token_and_total(caplog: pytest.LogCaptureFixture) -> None:
    trace = _trace()
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        trace.node("03_MessageClassificationNode")
        trace.node("10_GenerationSynthesisNode")
        trace.first_token()
        trace.first_token()
        trace.finish()

    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("node_done=03_MessageClassificationNode elapsed_ms=") for m in messages)
    assert any(m.startswith("node_done=10_GenerationSynthesisNode elapsed_ms=") for m in messages)
    assert (
        sum(m.startswith("first_token node=10_GenerationSynthesisNode ttft_ms=") for m in messages)
        == 1
    )
    assert messages[-1].startswith("graph_done total_ms=")
