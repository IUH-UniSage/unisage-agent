"""Offline LLM/embedding price lookup - Cost Tracking plan Task 5.

The only module in `app/` allowed to `import litellm` (see
`tests/core/test_no_raw_provider_clients.py`) - only for `cost_per_token()`,
pure price-table arithmetic, never to call a provider. `LITELLM_LOCAL_MODEL_COST_MAP`
is forced to `True` before the import so litellm never fetches its pricing
table from GitHub (see docs/product/DECISIONS.md "Cost Tracking" for why this
matters: that fetch would be an uncontrolled network call outside the
SSRF-guarded provider factory).
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from decimal import Decimal  # noqa: E402

import litellm  # noqa: E402

from app.core.config import settings  # noqa: E402

logger = logging.getLogger(__name__)

SOURCE_TYPE_SELF_HOSTED = "SELF_HOSTED"

# Mirrors Java's `UsageCostStatus` enum (plan.md "Data Model").
COST_STATUS_PRICED = "PRICED"
COST_STATUS_UNPRICED = "UNPRICED"
COST_STATUS_FREE = "FREE"


@dataclass(frozen=True, slots=True)
class CostResult:
    cost_usd: Decimal | None
    estimated_cost_usd: Decimal
    cost_status: str


def _cost_per_token(model_name: str, input_tokens: int, output_tokens: int, cached_tokens: int) -> Decimal | None:
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model_name,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
        )
    except Exception:
        return None
    return Decimal(str(prompt_cost)) + Decimal(str(completion_cost))


def calculate_actual(
    *,
    model_name: str,
    source_type: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> CostResult:
    """Cost of one already-completed call, from its real token usage.

    `SELF_HOSTED` is always `FREE` (plan.md "Architecture Decisions") - never
    asked of LiteLLM, since self-hosted models never have a public price.
    """

    if source_type == SOURCE_TYPE_SELF_HOSTED:
        return CostResult(cost_usd=Decimal("0"), estimated_cost_usd=Decimal("0"), cost_status=COST_STATUS_FREE)

    cost = _cost_per_token(model_name, input_tokens, output_tokens, cached_tokens)
    if cost is None:
        logger.warning("cost_calculator: no LiteLLM price for model %r - marking UNPRICED", model_name)
        fallback = estimate(
            model_name=model_name,
            source_type=source_type,
            input_tokens=input_tokens,
            max_output_tokens=output_tokens,
        )
        return CostResult(cost_usd=None, estimated_cost_usd=fallback, cost_status=COST_STATUS_UNPRICED)

    return CostResult(cost_usd=cost, estimated_cost_usd=cost, cost_status=COST_STATUS_PRICED)


def estimate(*, model_name: str, source_type: str, input_tokens: int, max_output_tokens: int) -> Decimal:
    """Upper-bound estimate for a Redis reservation, before the real call happens."""

    if source_type == SOURCE_TYPE_SELF_HOSTED:
        return Decimal("0")

    cost = _cost_per_token(model_name, input_tokens, max_output_tokens, cached_tokens=0)
    if cost is None:
        return Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD))
    return cost
