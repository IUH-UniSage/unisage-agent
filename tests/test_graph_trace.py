import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from app.core.config import settings
from app.core.observability.graph_trace import (
    GraphTrace,
    active_trace,
    bind_trace,
    unbind_trace,
)
from app.core.registry.model_registry import CredentialConfig


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


def _agent(thinking: object = None) -> Agent[None, str]:
    settings_ = ModelSettings(thinking=thinking) if thinking is not None else None  # type: ignore[typeddict-item]
    return Agent(TestModel(model_name="gpt-5-mini"), model_settings=settings_)


def _credential(credential_id: str, display_name: str | None = None) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider="openai",
        model_name="gpt-5-mini",
        api_base_url="https://api.openai.com/v1",
        priority=1,
        max_rpm=None,
        api_key="sk-secret",
        display_name=display_name,
    )


@pytest.mark.parametrize(
    ("thinking", "label"),
    [(None, "default"), (False, "off"), (True, "on"), ("minimal", "minimal")],
)
def test_llm_node_line_names_model_thinking_and_credential(
    thinking: object, label: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        _trace().node(
            "03_MessageClassificationNode",
            agent=_agent(thinking),
            credential=_credential("id-1", "Chat-Fallback"),
        )

    assert (
        "node=03_MessageClassificationNode model=gpt-5-mini "
        f"thinking={label} credential=Chat-Fallback "
    ) in caplog.text


def test_non_llm_node_line_uses_dashes(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        _trace().node("08_RetrievalFilteringNode")

    assert "node=08_RetrievalFilteringNode model=- thinking=- credential=- " in caplog.text


def test_model_switch_logs_another_line_for_the_running_node(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trace = _trace()
    with caplog.at_level(logging.INFO, logger="unisage.graph"):
        trace.node("10_GenerationSynthesisNode", agent=_agent(), credential=_credential("a"))
        trace.model_switch(
            _agent("minimal"),
            _credential("b", "C-14"),
            failed_credential=_credential("a", "Chat-Fallback"),
            reason="TimeoutError",
        )

    switch = caplog.records[-1]
    assert switch.levelno == logging.WARNING
    assert switch.getMessage().startswith(
        "node=10_GenerationSynthesisNode model=gpt-5-mini thinking=minimal credential=C-14 "
        "failover_from=Chat-Fallback reason=TimeoutError "
    )


def test_active_trace_is_only_set_while_bound() -> None:
    trace = _trace()
    assert active_trace() is None
    token = bind_trace(trace)
    try:
        assert active_trace() is trace
    finally:
        unbind_trace(token)
    assert active_trace() is None
