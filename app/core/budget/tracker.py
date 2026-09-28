"""Python side of Redis Lua-backed budget reservation. Fail-open throughout (soft
limit): any Redis error is logged and treated as "OK"/allow, never raised into a
Chat/Extraction/Embedding request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import redis.asyncio as redis_asyncio

from app.core.budget.period import current_period_key, period_ttl_seconds
from app.core.budget.snapshot import BudgetEntry, get_current_budget_snapshot
from app.core.config import settings

logger = logging.getLogger(__name__)

_LUA_DIR = Path(__file__).parent / "lua"
_INFLIGHT_TTL_SECONDS = 3600  # safety TTL so an abandoned inflight counter self-heals

ReserveResult = Literal["OK", "REJECT_EXCEEDED", "REJECT_THROTTLED"]
AcquireResult = Literal["OK", "DENY_EXCEEDED", "DENY_THROTTLED"]

# Fail-open on any Redis/script error - see module docstring.
_RESERVE_FAIL_OPEN: ReserveResult = "OK"
_ACQUIRE_FAIL_OPEN: AcquireResult = "OK"


def to_micro_usd(amount: Decimal) -> int:
    """Round-half-up to a micro-USD integer so Redis can use `INCRBY` on a plain
    integer counter instead of `INCRBYFLOAT`, which drifts under repeated adds."""

    return int((amount * 1_000_000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class _ScopeKeys:
    reserved: str
    committed: str
    inflight: str


def _scope_keys(scope_key: str, period: str, period_key: str) -> _ScopeKeys:
    suffix = f"{scope_key}:{period}:{period_key}"
    return _ScopeKeys(
        reserved=f"budget:reserved:{suffix}",
        committed=f"budget:committed:{suffix}",
        inflight=f"budget:inflight:{scope_key}",  # shared across periods, not per-period
    )


def _entries_for_scope(scope_key: str) -> list[BudgetEntry]:
    """Every enabled budget for this scope, across BOTH periods - a scope can have a
    DAILY and a MONTHLY budget enabled simultaneously (Java's unique index is per
    (scope[,ref], period), not per scope alone)."""

    snapshot = get_current_budget_snapshot()
    if snapshot is None:
        return []
    return [entry for entry in snapshot.entries if entry.scope_key == scope_key]


class BudgetTracker:
    """One instance is enough for the whole process - scripts are loaded once and
    `register_script`'s returned callable re-EVALSHAs on every call, re-uploading
    automatically if Redis ever evicts the script cache (`NOSCRIPT`)."""

    def __init__(self, redis_client: redis_asyncio.Redis | None = None) -> None:
        self._redis = redis_client or redis_asyncio.Redis.from_url(settings.REDIS_URL)
        self._reserve_request = self._redis.register_script(
            (_LUA_DIR / "reserve_request.lua").read_text()
        )
        self._acquire_provider = self._redis.register_script(
            (_LUA_DIR / "acquire_provider.lua").read_text()
        )
        self._release_provider = self._redis.register_script(
            (_LUA_DIR / "release_provider.lua").read_text()
        )
        self._settle_request = self._redis.register_script(
            (_LUA_DIR / "settle_request.lua").read_text()
        )

    async def reserve_request(
        self, *, request_id: str, purpose: str, estimate_usd: Decimal
    ) -> ReserveResult:
        """Request-level reservation against the SYSTEM and PURPOSE:<purpose> scopes."""

        scopes = _entries_for_scope("SYSTEM") + _entries_for_scope(f"PURPOSE:{purpose}")
        if not scopes:
            return "OK"

        keys: list[str] = []
        argv: list[Any] = [
            request_id,
            to_micro_usd(estimate_usd),
            _INFLIGHT_TTL_SECONDS,
            settings.BUDGET_RESERVATION_TTL_SECONDS,
            int(time.time()),
            "budget:resv:expiry",
            f"budget:resv:{request_id}",
            len(scopes),
        ]
        for entry in scopes:
            period_key = current_period_key(entry.period)
            scope_keys = _scope_keys(entry.scope_key, entry.period, period_key)
            keys.extend([scope_keys.reserved, scope_keys.committed, scope_keys.inflight])
            argv.extend(
                [
                    entry.action,
                    to_micro_usd(entry.limit_usd),
                    entry.throttle_max_concurrency or 0,
                    period_ttl_seconds(entry.period, period_key),
                ]
            )

        try:
            result = await self._reserve_request(keys=keys, args=argv)
            return result.decode() if isinstance(result, bytes) else result
        except Exception:
            logger.warning(
                "BudgetTracker.reserve_request failed for requestId=%s - fail-open (allow)",
                request_id,
                exc_info=True,
            )
            return _RESERVE_FAIL_OPEN

    async def acquire_provider(
        self, *, request_id: str, seq: int, provider: str, estimate_usd: Decimal
    ) -> AcquireResult:
        """Attempt-level reservation against the PROVIDER:<provider> scope."""

        scope_key = f"PROVIDER:{provider.lower()}"
        scopes = _entries_for_scope(scope_key)

        keys: list[str] = []
        argv: list[Any] = [
            request_id,
            seq,
            to_micro_usd(estimate_usd),
            _INFLIGHT_TTL_SECONDS,
            settings.BUDGET_RESERVATION_TTL_SECONDS,
            f"budget:resv:{request_id}",
            len(scopes),
        ]
        for entry in scopes:
            period_key = current_period_key(entry.period)
            sk = _scope_keys(entry.scope_key, entry.period, period_key)
            keys.extend([sk.reserved, sk.committed, sk.inflight])
            argv.extend(
                [
                    entry.action,
                    to_micro_usd(entry.limit_usd),
                    entry.throttle_max_concurrency or 0,
                    period_ttl_seconds(entry.period, period_key),
                ]
            )

        # Fallback committed key when no PROVIDER budget is configured at all - a DAILY
        # key by convention, purely for bookkeeping.
        fallback_period_key = current_period_key("DAILY")
        fallback_committed_key = _scope_keys(scope_key, "DAILY", fallback_period_key).committed
        argv.append(fallback_committed_key)

        try:
            result = await self._acquire_provider(keys=keys, args=argv)
            return result.decode() if isinstance(result, bytes) else result
        except Exception:
            logger.warning(
                "BudgetTracker.acquire_provider failed for requestId=%s seq=%s - fail-open (allow)",
                request_id,
                seq,
                exc_info=True,
            )
            return _ACQUIRE_FAIL_OPEN

    async def release_provider(self, *, request_id: str, seq: int, actual_usd: Decimal) -> None:
        """Releases a provider-attempt reservation and commits its real cost. Never
        raises - a release failure must not block a provider attempt from finishing."""

        try:
            await self._release_provider(
                keys=[],
                args=[request_id, seq, to_micro_usd(actual_usd), f"budget:resv:{request_id}"],
            )
        except Exception:
            logger.warning(
                "BudgetTracker.release_provider failed for requestId=%s seq=%s - reservation "
                "for this attempt may leak until release_expired_reservations sweeps it",
                request_id,
                seq,
                exc_info=True,
            )

    async def settle_request(self, *, request_id: str, actual_total_usd: Decimal) -> None:
        """Commits a request's real total cost and releases its reservation. Never
        raises - called from UsageRecorder.close()'s try/finally, must not prevent the
        outbox enqueue that follows it."""

        try:
            await self._settle_request(
                keys=[],
                args=[
                    request_id,
                    f"budget:resv:{request_id}",
                    f"budget:settled:{request_id}",
                    24 * 3600,
                    to_micro_usd(actual_total_usd),
                ],
            )
        except Exception:
            logger.warning(
                "BudgetTracker.settle_request failed for requestId=%s - reservation may leak "
                "until release_expired_reservations sweeps it",
                request_id,
                exc_info=True,
            )


_default_tracker: BudgetTracker | None = None


def get_default_tracker() -> BudgetTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = BudgetTracker()
    return _default_tracker
