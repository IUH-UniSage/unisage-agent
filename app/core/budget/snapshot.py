"""In-memory snapshot of enabled budgets from `GET /internal/budgets/snapshot`.
Mirrors `app.core.registry.model_registry`'s snapshot pattern (module-level cache, frozen
dataclass, atomic swap via a single name rebind) but for budgets instead of
credentials.

Fail-open by design (budget enforcement is a soft limit) - a failed load/refresh
keeps whatever snapshot is already cached (or `None` if none has ever loaded), never
raises into a Chat/Extraction/Embedding request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.integrations.backend_java_client import BackendJavaClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetEntry:
    scope: str  # SYSTEM | PROVIDER | PURPOSE
    scope_provider: str | None
    scope_purpose: str | None
    period: str  # DAILY | MONTHLY
    limit_usd: Decimal
    action: str  # ALERT | THROTTLE | BLOCK
    throttle_max_concurrency: int | None

    @property
    def scope_key(self) -> str:
        """`SYSTEM` | `PURPOSE:<name>` | `PROVIDER:<name-lowercase>`. Lowercased for
        PROVIDER only, matching how Java compares `scopeProvider` case-insensitively
        against `ChatModel.provider`."""

        if self.scope == "PROVIDER":
            return f"PROVIDER:{(self.scope_provider or '').lower()}"
        if self.scope == "PURPOSE":
            return f"PURPOSE:{self.scope_purpose}"
        return "SYSTEM"


@dataclass(frozen=True)
class BudgetSnapshot:
    version: int
    entries: tuple[BudgetEntry, ...]

    def for_scope_key(self, scope_key: str, period: str) -> BudgetEntry | None:
        """The one budget (if any) matching both `scope_key` and `period` - at most
        one exists, enforced by Java's 3 partial unique indexes."""

        for entry in self.entries:
            if entry.scope_key == scope_key and entry.period == period:
                return entry
        return None


def _parse_entry(raw: dict[str, Any]) -> BudgetEntry:
    return BudgetEntry(
        scope=str(raw["scope"]),
        scope_provider=raw.get("scopeProvider"),
        scope_purpose=raw.get("scopePurpose"),
        period=str(raw["period"]),
        limit_usd=Decimal(str(raw["limitUsd"])),
        action=str(raw["action"]),
        throttle_max_concurrency=raw.get("throttleMaxConcurrency"),
    )


def parse_budget_snapshot(payload: dict[str, Any]) -> BudgetSnapshot:
    """Pure parse of the raw `GET /internal/budgets/snapshot` JSON - no I/O, cheap to
    unit test against a hand-built payload."""

    entries = tuple(_parse_entry(raw) for raw in payload.get("budgets") or [])
    return BudgetSnapshot(version=int(payload["version"]), entries=entries)


# Process-local cache - same "single name rebind is atomic under the GIL" trick as
# app.core.registry.model_registry._current_snapshot.
_current_budget_snapshot: BudgetSnapshot | None = None


def get_current_budget_snapshot() -> BudgetSnapshot | None:
    """`None` means the snapshot was never loaded yet (or budget enforcement is
    effectively off - callers must treat that as "no budgets configured", not an
    error, matching the soft-limit posture)."""

    return _current_budget_snapshot


def set_current_budget_snapshot(snapshot: BudgetSnapshot) -> None:
    global _current_budget_snapshot
    _current_budget_snapshot = snapshot


async def refresh_budget_snapshot(client: BackendJavaClient | None = None) -> BudgetSnapshot | None:
    """Loads the latest snapshot from Java and swaps it in. On any failure (Java
    down, malformed response, ...), logs and keeps whatever was already cached -
    never raises. Returns the snapshot now in effect (old or new)."""

    try:
        payload = await (client or BackendJavaClient()).get_budget_snapshot()
        snapshot = parse_budget_snapshot(payload)
    except Exception:
        logger.warning(
            "refresh_budget_snapshot: failed to load from Java - keeping previous snapshot",
            exc_info=True,
        )
        return get_current_budget_snapshot()

    current = get_current_budget_snapshot()
    if current is not None and snapshot.version <= current.version:
        # Stale/duplicate response (a concurrent refresh already landed a newer one) -
        # never move backwards.
        return current

    set_current_budget_snapshot(snapshot)
    return snapshot
