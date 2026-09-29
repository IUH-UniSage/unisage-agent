"""Tests for cost calculation from backend-java's model prices.

The parity cases use the numbers `litellm.cost_per_token()` returned for the same token counts
and unit prices, so the switch away from LiteLLM does not change what a call costs."""

from collections.abc import Iterator
from decimal import Decimal

import pytest

from app.core.pricing.snapshot import parse_pricing_snapshot, set_current_pricing_snapshot
from app.core.usage import cost_calculator
from app.core.usage.cost_calculator import (
    COST_STATUS_FREE,
    COST_STATUS_PRICED,
    COST_STATUS_UNPRICED,
    calculate_actual,
    estimate,
)

FALLBACK = Decimal(str(cost_calculator.settings.BUDGET_RESERVATION_FALLBACK_USD))


@pytest.fixture(autouse=True)
def _prices() -> Iterator[None]:
    set_current_pricing_snapshot(
        parse_pricing_snapshot(
            {
                "version": 1,
                "prices": [
                    {
                        "provider": "openai",
                        "modelName": "gpt-4o-mini",
                        "inputPerMillion": 0.15,
                        "outputPerMillion": 0.6,
                        "cachedInputPerMillion": 0.075,
                    },
                    {
                        "provider": "openai",
                        "modelName": "gpt-4o",
                        "inputPerMillion": 2.5,
                        "outputPerMillion": 10,
                        "cachedInputPerMillion": 1.25,
                    },
                    {
                        "provider": "openai",
                        "modelName": "text-embedding-3-small",
                        "inputPerMillion": 0.02,
                        "outputPerMillion": None,
                        "cachedInputPerMillion": None,
                    },
                ],
            }
        )
    )
    yield
    set_current_pricing_snapshot(None)


@pytest.mark.parametrize(
    ("model", "input_tokens", "output_tokens", "cached_tokens", "litellm_cost"),
    [
        ("gpt-4o-mini", 1000, 500, 200, "0.000435"),
        ("gpt-4o-mini", 1000, 500, 0, "0.00045"),
        ("text-embedding-3-small", 5000, 0, 0, "0.0001"),
        ("gpt-4o", 1234, 567, 1000, "0.007505"),
    ],
)
def test_calculate_actual_matches_litellm(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int, litellm_cost: str
) -> None:
    result = calculate_actual(
        provider="openai",
        model_name=model,
        source_type="CLOUD_API",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )

    assert result.cost_status == COST_STATUS_PRICED
    assert result.cost_usd == Decimal(litellm_cost)
    assert result.estimated_cost_usd == result.cost_usd


def test_price_lookup_is_per_provider() -> None:
    result = calculate_actual(
        provider="google",
        model_name="gpt-4o-mini",
        source_type="CLOUD_API",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result.cost_status == COST_STATUS_UNPRICED


def test_calculate_actual_unpriced_model_falls_back_to_estimate(caplog) -> None:
    result = calculate_actual(
        provider="openai",
        model_name="totally-unknown-model-xyz",
        source_type="CLOUD_API",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result.cost_status == COST_STATUS_UNPRICED
    assert result.cost_usd is None
    assert result.estimated_cost_usd == FALLBACK
    assert "UNPRICED" in caplog.text


def test_no_snapshot_loaded_means_unpriced() -> None:
    set_current_pricing_snapshot(None)

    result = calculate_actual(
        provider="openai",
        model_name="gpt-4o-mini",
        source_type="CLOUD_API",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result.cost_status == COST_STATUS_UNPRICED


def test_calculate_actual_self_hosted_is_free() -> None:
    result = calculate_actual(
        provider=None,
        model_name="local-llama",
        source_type="SELF_HOSTED",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result == cost_calculator.CostResult(
        cost_usd=Decimal("0"), estimated_cost_usd=Decimal("0"), cost_status=COST_STATUS_FREE
    )


def test_estimate_priced_model_ignores_cache() -> None:
    value = estimate(
        provider="openai",
        model_name="gpt-4o-mini",
        source_type="CLOUD_API",
        input_tokens=1000,
        max_output_tokens=500,
    )

    assert value == Decimal("0.00045")


def test_estimate_unpriced_model_uses_fallback() -> None:
    value = estimate(
        provider="openai",
        model_name="totally-unknown-model-xyz",
        source_type="CLOUD_API",
        input_tokens=1000,
        max_output_tokens=500,
    )

    assert value == FALLBACK


def test_estimate_self_hosted_is_zero() -> None:
    value = estimate(
        provider=None,
        model_name="local-llama",
        source_type="SELF_HOSTED",
        input_tokens=1000,
        max_output_tokens=500,
    )

    assert value == Decimal("0")


def test_cached_tokens_above_input_are_capped() -> None:
    result = calculate_actual(
        provider="openai",
        model_name="gpt-4o-mini",
        source_type="CLOUD_API",
        input_tokens=100,
        output_tokens=0,
        cached_tokens=500,
    )

    assert result.cost_usd == Decimal("0.0000075")
