"""Tests for offline price lookup, including a spike proof that
`litellm.cost_per_token()` never makes a network call."""

from decimal import Decimal

from app.core import cost_calculator
from app.core.cost_calculator import (
    COST_STATUS_FREE,
    COST_STATUS_PRICED,
    COST_STATUS_UNPRICED,
    calculate_actual,
    estimate,
)


def test_calculate_actual_priced_model() -> None:
    result = calculate_actual(
        model_name="gpt-4o-mini",
        source_type="CLOUD_API",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result.cost_status == COST_STATUS_PRICED
    assert result.cost_usd is not None
    assert result.cost_usd > Decimal("0")
    assert result.estimated_cost_usd == result.cost_usd


def test_calculate_actual_unpriced_model_falls_back_to_estimate(caplog) -> None:
    result = calculate_actual(
        model_name="totally-unknown-model-xyz",
        source_type="CLOUD_API",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result.cost_status == COST_STATUS_UNPRICED
    assert result.cost_usd is None
    assert result.estimated_cost_usd == Decimal(str(cost_calculator.settings.BUDGET_RESERVATION_FALLBACK_USD))
    assert "UNPRICED" in caplog.text or "no LiteLLM price" in caplog.text


def test_calculate_actual_self_hosted_is_free_without_calling_litellm(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("SELF_HOSTED must never call litellm.cost_per_token")

    monkeypatch.setattr(cost_calculator.litellm, "cost_per_token", _boom)

    result = calculate_actual(
        model_name="local-llama",
        source_type="SELF_HOSTED",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result == cost_calculator.CostResult(
        cost_usd=Decimal("0"), estimated_cost_usd=Decimal("0"), cost_status=COST_STATUS_FREE
    )


def test_estimate_priced_model() -> None:
    value = estimate(model_name="gpt-4o-mini", source_type="CLOUD_API", input_tokens=1000, max_output_tokens=500)

    assert value > Decimal("0")


def test_estimate_unpriced_model_uses_fallback() -> None:
    value = estimate(
        model_name="totally-unknown-model-xyz", source_type="CLOUD_API", input_tokens=1000, max_output_tokens=500
    )

    assert value == Decimal(str(cost_calculator.settings.BUDGET_RESERVATION_FALLBACK_USD))


def test_estimate_self_hosted_is_zero() -> None:
    value = estimate(model_name="local-llama", source_type="SELF_HOSTED", input_tokens=1000, max_output_tokens=500)

    assert value == Decimal("0")


def test_embedding_shaped_call_is_priced_like_completion() -> None:
    """Embedding usage has no output_tokens - calculate_actual must accept 0."""

    result = calculate_actual(
        model_name="text-embedding-3-small",
        source_type="CLOUD_API",
        input_tokens=800,
        output_tokens=0,
    )

    assert result.cost_status == COST_STATUS_PRICED
    assert result.cost_usd is not None
    assert result.cost_usd > Decimal("0")


def test_cost_per_token_never_opens_a_network_connection(monkeypatch) -> None:
    """Spike proof: LiteLLM's pricing lookup must not touch the network -
    see docs/product/DECISIONS.md "Cost Tracking" for why LITELLM_LOCAL_MODEL_COST_MAP
    matters (litellm fetches its price table from GitHub on import/unknown-model
    otherwise)."""

    import socket

    calls: list[object] = []
    original_connect = socket.socket.connect

    def spy_connect(self, address):
        calls.append(address)
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy_connect)

    calculate_actual(model_name="gpt-4o-mini", source_type="CLOUD_API", input_tokens=100, output_tokens=50)
    calculate_actual(model_name="totally-unknown-model-xyz", source_type="CLOUD_API", input_tokens=100, output_tokens=50)
    estimate(model_name="gpt-4o-mini", source_type="CLOUD_API", input_tokens=100, max_output_tokens=50)

    assert calls == [], f"litellm made unexpected network connection(s): {calls}"
