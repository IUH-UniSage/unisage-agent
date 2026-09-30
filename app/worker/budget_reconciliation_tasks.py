"""Two Beat-scheduled budget upkeep jobs, both plain sync functions (not Celery
tasks themselves - see `app.worker.usage_outbox_tasks` for why: tests call them
directly without going through Celery's task machinery).

`release_expired_reservations_once`: sweeps abandoned reservations (a crashed
process that never reached `UsageRecorder.close()`) so their reserved/inflight
capacity isn't lost forever - see `release_expired_reservations.lua`.

`reconcile_budget_committed_once`: corrects `committed` counter drift against
Java's own DB total for every enabled budget, but only when the usage outbox is
fully drained - a non-empty outbox means Java's total doesn't yet reflect
everything this process has recorded, so reconciling against it now would
silently erase real spend instead of correcting drift.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import redis

from app.core.budget.period import current_period_key
from app.core.budget.snapshot import get_current_budget_snapshot
from app.core.config import settings
from app.core.usage.usage_outbox import OUTBOX_KEY
from app.integrations.backend_java_client import BackendJavaClient
from app.worker.usage_outbox_tasks import PROCESSING_KEY

logger = logging.getLogger(__name__)

_LUA_DIR = Path(__file__).parent.parent / "core" / "budget" / "lua"
_EXPIRY_ZSET_KEY = "budget:resv:expiry"
_DEFAULT_MAX_BATCH = 500
# A reconciliation delta bigger than this fraction of Java's own total is
# suspicious enough to want a human to look, even though it's still applied -
# budget is a soft limit, so a self-correcting drift should never block on it.
_DRIFT_WARNING_THRESHOLD = 0.05


def release_expired_reservations_once(
    *, max_batch: int = _DEFAULT_MAX_BATCH, redis_client: redis.Redis | None = None
) -> int:
    """Returns how many expired reservations were released this run."""

    owns_connection = redis_client is None
    conn = redis_client
    if conn is None:
        try:
            conn = redis.Redis.from_url(settings.REDIS_URL)
        except Exception:
            logger.warning(
                "release_expired_reservations: Redis unavailable, skipping this run",
                exc_info=True,
            )
            return 0

    try:
        script = (_LUA_DIR / "release_expired_reservations.lua").read_text()
        released = conn.eval(script, 0, str(int(time.time())), _EXPIRY_ZSET_KEY, str(max_batch))
        if released:
            logger.info(
                "release_expired_reservations: released %d abandoned reservation(s)",
                len(released),
            )
        return len(released)
    except Exception:
        logger.warning("release_expired_reservations: run failed", exc_info=True)
        return 0
    finally:
        if owns_connection and conn is not None:
            conn.close()


def _pending_and_committed(
    conn: redis.Redis, *, committed_key: str, check_script: str
) -> tuple[int, int]:
    pending, committed = conn.eval(check_script, 3, OUTBOX_KEY, PROCESSING_KEY, committed_key)
    return int(pending), int(committed)


def reconcile_budget_committed_once(
    *,
    redis_client: redis.Redis | None = None,
    backend_client: BackendJavaClient | None = None,
) -> dict[str, int]:
    """Returns `{"reconciled": N, "skipped": N}` - `skipped` counts scope-periods
    left alone this run because the outbox still had pending items."""

    snapshot = get_current_budget_snapshot()
    if snapshot is None or not snapshot.entries:
        return {"reconciled": 0, "skipped": 0}

    owns_connection = redis_client is None
    conn = redis_client
    if conn is None:
        try:
            conn = redis.Redis.from_url(settings.REDIS_URL)
        except Exception:
            logger.warning(
                "reconcile_budget_committed: Redis unavailable, skipping this run", exc_info=True
            )
            return {"reconciled": 0, "skipped": 0}

    check_script = (_LUA_DIR / "reconcile_check.lua").read_text()
    client = backend_client if backend_client is not None else BackendJavaClient()
    reconciled = 0
    skipped = 0
    # One Java call per distinct (period, periodKey) covers every scope in one
    # response - group the snapshot's entries by that pair up front.
    by_period: dict[tuple[str, str], list[str]] = {}
    for entry in snapshot.entries:
        period_key = current_period_key(entry.period)
        by_period.setdefault((entry.period, period_key), []).append(entry.scope_key)

    try:
        for (period, period_key), scope_keys in by_period.items():
            totals: dict[str, int] | None = None
            for scope_key in scope_keys:
                committed_key = f"budget:committed:{scope_key}:{period}:{period_key}"
                pending, committed = _pending_and_committed(
                    conn, committed_key=committed_key, check_script=check_script
                )
                if pending > 0:
                    skipped += 1
                    continue

                if totals is None:
                    try:
                        response = asyncio.run(
                            client.get_period_totals(period=period, period_key=period_key)
                        )
                    except Exception:
                        logger.warning(
                            "reconcile_budget_committed: failed to fetch period totals "
                            "for period=%s periodKey=%s",
                            period,
                            period_key,
                            exc_info=True,
                        )
                        break
                    totals = response.get("totals") or {}

                db_total = int(totals.get(scope_key, 0))
                delta = db_total - committed
                if delta != 0:
                    conn.incrby(committed_key, delta)
                drift_ratio = abs(delta) / db_total if db_total > 0 else (1.0 if delta else 0.0)
                if drift_ratio > _DRIFT_WARNING_THRESHOLD:
                    logger.warning(
                        "reconcile_budget_committed: scope=%s period=%s periodKey=%s drifted "
                        "%.1f%% (redis=%d db=%d) - corrected",
                        scope_key,
                        period,
                        period_key,
                        drift_ratio * 100,
                        committed,
                        db_total,
                    )
                reconciled += 1
    finally:
        if owns_connection:
            conn.close()

    return {"reconciled": reconciled, "skipped": skipped}
