"""Unit tests for UsageRecorder. Graph-level line-count/
failover/disconnect scenarios are covered separately by
tests/graph/test_usage_recorder_wiring.py; this file exercises the recorder in isolation
(no real graph run needed)."""

import json
from typing import Any

import pytest

import app.core.usage.usage_outbox as usage_outbox_module
from app.core.registry.model_registry import CredentialConfig
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.streaming import AttemptOutcome


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens


def _credential(**overrides: Any) -> CredentialConfig:
    defaults: dict[str, Any] = {
        "id": "cred-1",
        "revision": 1,
        "source_type": "CLOUD_API",
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_base_url": "https://api.openai.com/v1",
        "priority": 1,
        "max_rpm": None,
        "api_key": "sk-test",
    }
    defaults.update(overrides)
    return CredentialConfig(**defaults)


@pytest.fixture
def captured_outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def _fake_enqueue(payload: dict[str, Any], *, redis_client: Any = None) -> None:
        del redis_client
        captured.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(usage_outbox_module, "enqueue_usage_payload", _fake_enqueue)
    return captured


@pytest.mark.asyncio
async def test_priced_success_line(captured_outbox: list[dict[str, Any]]) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(100, 50),
            latency_ms=250,
        )
    )
    await recorder.close(status="SUCCESS")

    line = captured_outbox[0]["lines"][0]
    assert line["costStatus"] == "PRICED"
    assert line["costUsd"] is not None
    assert line["inputTokens"] == 100
    assert line["outputTokens"] == 50
    assert line["status"] == "SUCCESS"
    assert line["latencyMs"] == 250


@pytest.mark.asyncio
async def test_unpriced_model_line_has_null_cost_usd(captured_outbox: list[dict[str, Any]]) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(model_name="totally-unknown-model-xyz"),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(100, 50),
            latency_ms=250,
        )
    )
    await recorder.close(status="SUCCESS")

    line = captured_outbox[0]["lines"][0]
    assert line["costStatus"] == "UNPRICED"
    assert line["costUsd"] is None
    assert float(line["estimatedCostUsd"]) > 0


@pytest.mark.asyncio
async def test_self_hosted_credential_is_free_with_null_cost_usd(
    captured_outbox: list[dict[str, Any]],
) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(
                source_type="SELF_HOSTED", provider=None, model_name="local-llama"
            ),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(100, 50),
            latency_ms=250,
        )
    )
    await recorder.close(status="SUCCESS")

    line = captured_outbox[0]["lines"][0]
    assert line["costStatus"] == "FREE"
    assert line["costUsd"] is None  # DB CHECK requires NULL for any non-PRICED status


@pytest.mark.asyncio
async def test_error_line_has_null_cost_usd_and_error_code(
    captured_outbox: list[dict[str, Any]],
) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="ERROR",
            usage=None,
            latency_ms=100,
            error_code="RuntimeError",
        )
    )
    await recorder.close(status="ERROR")

    line = captured_outbox[0]["lines"][0]
    assert line["status"] == "ERROR"
    assert line["costUsd"] is None
    assert line["errorCode"] == "RuntimeError"
    assert captured_outbox[0]["status"] == "ERROR"


@pytest.mark.asyncio
async def test_no_lines_recorded_means_nothing_is_enqueued(
    captured_outbox: list[dict[str, Any]],
) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    await recorder.close(status="SUCCESS")

    assert captured_outbox == []


@pytest.mark.asyncio
async def test_close_is_idempotent(captured_outbox: list[dict[str, Any]]) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(10, 5),
            latency_ms=1,
        )
    )
    await recorder.close(status="SUCCESS")
    await recorder.close(status="SUCCESS")

    assert len(captured_outbox) == 1


@pytest.mark.asyncio
async def test_seq_is_assigned_in_recording_order_across_nodes(
    captured_outbox: list[dict[str, Any]],
) -> None:
    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    recorder.bind("MessageClassificationNode")(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(1, 1),
            latency_ms=1,
        )
    )
    recorder.bind("GenerationSynthesisNode")(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="SUCCESS",
            usage=_FakeUsage(1, 1),
            latency_ms=1,
        )
    )
    await recorder.close(status="SUCCESS")

    lines = captured_outbox[0]["lines"]
    assert [line["seq"] for line in lines] == [0, 1]


@pytest.mark.asyncio
async def test_on_attempt_callback_never_raises_even_if_credential_is_malformed(
    captured_outbox: list[dict[str, Any]],
) -> None:
    """`_safe_record_attempt` in streaming.py already wraps this, but the recorder's
    own `bind()` closure must not be the weak link if that guard is ever removed."""

    recorder = UsageRecorder(request_id="r1", purpose="CHAT")
    on_attempt = recorder.bind("GenerationSynthesisNode")

    class _BrokenUsage:
        @property
        def input_tokens(self) -> int:
            raise RuntimeError("boom")

    on_attempt(
        AttemptOutcome(
            credential=_credential(),
            attempt=0,
            status="SUCCESS",
            usage=_BrokenUsage(),  # type: ignore[arg-type]
            latency_ms=1,
        )
    )
    await recorder.close(status="SUCCESS")

    # The broken line was dropped, not raised - request still closes cleanly.
    assert captured_outbox == []
