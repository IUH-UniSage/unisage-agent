from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest

from app.core.pricing import snapshot as pricing_snapshot
from app.core.pricing.snapshot import (
    get_current_pricing_snapshot,
    parse_pricing_snapshot,
    refresh_pricing_snapshot,
    set_current_pricing_snapshot,
)


class _FakeClient:
    def __init__(
        self, payload: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self._payload = payload
        self._error = error

    async def get_model_pricing_snapshot(self) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        assert self._payload is not None
        return self._payload


def _payload(version: int, input_price: float = 0.15) -> dict[str, Any]:
    return {
        "version": version,
        "prices": [
            {
                "provider": "openai",
                "modelName": "gpt-4o-mini",
                "inputPerMillion": input_price,
                "outputPerMillion": 0.6,
                "cachedInputPerMillion": 0.075,
            },
            {
                "provider": "google",
                "modelName": "gemini-embedding-001",
                "inputPerMillion": 0.15,
                "outputPerMillion": None,
                "cachedInputPerMillion": None,
            },
        ],
    }


@pytest.fixture(autouse=True)
def _reset_snapshot() -> Iterator[None]:
    set_current_pricing_snapshot(None)
    yield
    set_current_pricing_snapshot(None)


def test_parse_and_lookup_ignore_case_and_whitespace() -> None:
    snapshot = parse_pricing_snapshot(_payload(3))

    price = snapshot.lookup("OpenAI", " GPT-4o-mini ")
    assert price is not None
    assert price.input_per_million == Decimal("0.15")
    assert price.cached_input_per_million == Decimal("0.075")
    embedding = snapshot.lookup("google", "gemini-embedding-001")
    assert embedding is not None
    assert embedding.output_per_million is None
    assert snapshot.lookup("openai", "unknown") is None
    assert snapshot.lookup(None, "gpt-4o-mini") is None


@pytest.mark.asyncio
async def test_refresh_swaps_in_new_snapshot() -> None:
    loaded = await refresh_pricing_snapshot(_FakeClient(_payload(5)))

    assert loaded is not None
    assert loaded.version == 5
    assert get_current_pricing_snapshot() is loaded


@pytest.mark.asyncio
async def test_refresh_failure_keeps_previous_snapshot() -> None:
    await refresh_pricing_snapshot(_FakeClient(_payload(5)))

    kept = await refresh_pricing_snapshot(_FakeClient(error=RuntimeError("java down")))

    assert kept is not None
    assert kept.version == 5


@pytest.mark.asyncio
async def test_refresh_failure_with_nothing_loaded_returns_none() -> None:
    assert await refresh_pricing_snapshot(_FakeClient(error=RuntimeError("java down"))) is None


@pytest.mark.asyncio
async def test_refresh_never_moves_backwards() -> None:
    await refresh_pricing_snapshot(_FakeClient(_payload(7, input_price=0.2)))

    current = await refresh_pricing_snapshot(_FakeClient(_payload(6)))

    assert current is not None
    assert current.version == 7
    assert pricing_snapshot.get_current_pricing_snapshot() is current
