"""In-memory snapshot of model prices from `GET /internal/model-pricing/snapshot`.

Same pattern as `app.core.budget.snapshot` (module-level cache, atomic rebind, never moves
backwards). Fail-open: a failed refresh keeps the cached snapshot, and with no snapshot at all
every call is priced as UNPRICED by `cost_calculator` - never an error in the request path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.integrations.backend_java_client import BackendJavaClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. `output_per_million` is None for embedding models;
    `cached_input_per_million` None means cached tokens bill at the input price."""

    input_per_million: Decimal
    output_per_million: Decimal | None
    cached_input_per_million: Decimal | None


@dataclass(frozen=True)
class PricingSnapshot:
    version: int
    prices: dict[tuple[str, str], ModelPrice]

    def lookup(self, provider: str | None, model_name: str | None) -> ModelPrice | None:
        if not provider or not model_name:
            return None
        return self.prices.get(_key(provider, model_name))


def _key(provider: str, model_name: str) -> tuple[str, str]:
    return provider.strip().lower(), model_name.strip().lower()


def _decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def parse_pricing_snapshot(payload: dict[str, Any]) -> PricingSnapshot:
    """Pure parse, no I/O."""

    prices: dict[tuple[str, str], ModelPrice] = {}
    for raw in payload.get("prices") or []:
        input_price = _decimal(raw.get("inputPerMillion"))
        if input_price is None:
            continue
        prices[_key(str(raw["provider"]), str(raw["modelName"]))] = ModelPrice(
            input_per_million=input_price,
            output_per_million=_decimal(raw.get("outputPerMillion")),
            cached_input_per_million=_decimal(raw.get("cachedInputPerMillion")),
        )
    return PricingSnapshot(version=int(payload["version"]), prices=prices)


_current_pricing_snapshot: PricingSnapshot | None = None


def get_current_pricing_snapshot() -> PricingSnapshot | None:
    return _current_pricing_snapshot


def set_current_pricing_snapshot(snapshot: PricingSnapshot | None) -> None:
    global _current_pricing_snapshot
    _current_pricing_snapshot = snapshot


async def refresh_pricing_snapshot(
    client: BackendJavaClient | None = None,
) -> PricingSnapshot | None:
    """Never raises; returns the snapshot now in effect (old or new)."""

    try:
        payload = await (client or BackendJavaClient()).get_model_pricing_snapshot()
        snapshot = parse_pricing_snapshot(payload)
    except Exception:
        logger.warning(
            "refresh_pricing_snapshot: failed to load from Java - keeping previous snapshot",
            exc_info=True,
        )
        return get_current_pricing_snapshot()

    current = get_current_pricing_snapshot()
    if current is not None and snapshot.version < current.version:
        return current

    set_current_pricing_snapshot(snapshot)
    return snapshot
