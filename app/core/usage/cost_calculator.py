"""Cost of a provider call from backend-java's model prices (`app.core.pricing.snapshot`).

Prices are looked up by (provider, model name) - the same key the SA sees on the Pricing tab -
so what the dashboard shows, what the budget reserves and what is logged come from one table.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal

from app.core.config import settings
from app.core.pricing.snapshot import ModelPrice, get_current_pricing_snapshot

logger = logging.getLogger(__name__)

SOURCE_TYPE_SELF_HOSTED = "SELF_HOSTED"

# Mirrors Java's `UsageCostStatus` enum.
COST_STATUS_PRICED = "PRICED"
COST_STATUS_UNPRICED = "UNPRICED"
COST_STATUS_FREE = "FREE"

_ONE_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class CostResult:
    cost_usd: Decimal | None
    estimated_cost_usd: Decimal
    cost_status: str


def calculate_actual(
    *,
    provider: str | None,
    model_name: str,
    source_type: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> CostResult:
    """Cost of one already-completed call, from its real token usage.

    `SELF_HOSTED` is always `FREE` - self-hosted models never have a public price.
    """

    if source_type == SOURCE_TYPE_SELF_HOSTED:
        return CostResult(
            cost_usd=Decimal("0"), estimated_cost_usd=Decimal("0"), cost_status=COST_STATUS_FREE
        )

    price = _lookup(provider, model_name)
    if price is None:
        logger.warning(
            "cost_calculator: no price for %s/%r - marking UNPRICED", provider, model_name
        )
        fallback = Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD))
        return CostResult(
            cost_usd=None, estimated_cost_usd=fallback, cost_status=COST_STATUS_UNPRICED
        )

    cost = _cost(price, input_tokens, output_tokens, cached_tokens)
    return CostResult(cost_usd=cost, estimated_cost_usd=cost, cost_status=COST_STATUS_PRICED)


def estimate(
    *,
    provider: str | None,
    model_name: str,
    source_type: str,
    input_tokens: int,
    max_output_tokens: int,
) -> Decimal:
    """Upper-bound estimate for a Redis reservation, before the real call happens."""

    if source_type == SOURCE_TYPE_SELF_HOSTED:
        return Decimal("0")

    price = _lookup(provider, model_name)
    if price is None:
        return Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD))
    return _cost(price, input_tokens, max_output_tokens, cached_tokens=0)


def _lookup(provider: str | None, model_name: str) -> ModelPrice | None:
    snapshot = get_current_pricing_snapshot()
    return snapshot.lookup(provider, model_name) if snapshot is not None else None


def _cost(price: ModelPrice, input_tokens: int, output_tokens: int, cached_tokens: int) -> Decimal:
    # Providers count cached tokens inside input_tokens; they are billed at the cached rate
    # instead of the input rate, not on top of it.
    cached = min(max(cached_tokens, 0), input_tokens)
    cached_rate = (
        price.cached_input_per_million
        if price.cached_input_per_million is not None
        else price.input_per_million
    )
    output_rate = price.output_per_million or Decimal("0")
    total = (
        (input_tokens - cached) * price.input_per_million
        + cached * cached_rate
        + output_tokens * output_rate
    )
    return total / _ONE_MILLION
