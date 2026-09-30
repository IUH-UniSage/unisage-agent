"""Tests for app/worker/budget_reconciliation_tasks.py - both functions call real
Lua scripts (`eval`), so a hand-rolled fake can't stand in; these run against a
real local Redis, same as tests/core/budget/test_tracker.py."""

import time
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import redis
import redis.asyncio as redis_asyncio

from app.core.budget.period import current_period_key
from app.core.budget.snapshot import BudgetEntry, BudgetSnapshot, set_current_budget_snapshot
from app.core.budget.tracker import BudgetTracker
from app.core.usage.usage_outbox import OUTBOX_KEY
from app.worker.budget_reconciliation_tasks import (
    reconcile_budget_committed_once,
    release_expired_reservations_once,
)


@pytest_asyncio.fixture
async def redis_client():
    client = redis_asyncio.Redis.from_url("redis://localhost:6379/0")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
def sync_redis():
    conn = redis.Redis.from_url("redis://localhost:6379/0")
    yield conn
    conn.close()


def _system_block(limit_usd: str = "10") -> BudgetEntry:
    return BudgetEntry(
        scope="SYSTEM",
        scope_provider=None,
        scope_purpose=None,
        period="DAILY",
        limit_usd=Decimal(limit_usd),
        action="BLOCK",
        throttle_max_concurrency=None,
    )


@pytest.mark.asyncio
async def test_release_expired_reservations_frees_reserved_and_inflight(
    redis_client, sync_redis
) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block(),)))
    tracker = BudgetTracker(redis_client)
    await tracker.reserve_request(
        request_id="expired-req", purpose="CHAT", estimate_usd=Decimal("1.0")
    )
    # Force this reservation into the past so it's picked up as expired.
    await redis_client.zadd("budget:resv:expiry", {"expired-req": time.time() - 10})

    released = release_expired_reservations_once(redis_client=sync_redis)

    assert released == 1
    reserved = await redis_client.get("budget:reserved:SYSTEM:DAILY:" + current_period_key("DAILY"))
    assert reserved is None or int(reserved) == 0
    assert await redis_client.exists("budget:resv:expired-req") == 0


@pytest.mark.asyncio
async def test_release_expired_reservations_leaves_unexpired_ones_alone(
    redis_client, sync_redis
) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block(),)))
    tracker = BudgetTracker(redis_client)
    await tracker.reserve_request(
        request_id="still-active", purpose="CHAT", estimate_usd=Decimal("1.0")
    )

    released = release_expired_reservations_once(redis_client=sync_redis)

    assert released == 0
    assert await redis_client.exists("budget:resv:still-active") == 1


def test_reconcile_corrects_drift_against_java_total(sync_redis) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block(),)))
    key = "budget:committed:SYSTEM:DAILY:" + current_period_key("DAILY")
    sync_redis.set(key, 1000)

    fake_client = AsyncMock()
    fake_client.get_period_totals.return_value = {"totals": {"SYSTEM": 5000}}

    result = reconcile_budget_committed_once(redis_client=sync_redis, backend_client=fake_client)

    assert result == {"reconciled": 1, "skipped": 0}
    assert int(sync_redis.get(key)) == 5000


def test_reconcile_skips_when_outbox_has_pending_items(sync_redis) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block(),)))
    key = "budget:committed:SYSTEM:DAILY:" + current_period_key("DAILY")
    sync_redis.set(key, 1000)
    sync_redis.lpush(OUTBOX_KEY, '{"fake": "payload"}')

    fake_client = AsyncMock()
    fake_client.get_period_totals.return_value = {"totals": {"SYSTEM": 5000}}

    result = reconcile_budget_committed_once(redis_client=sync_redis, backend_client=fake_client)

    assert result == {"reconciled": 0, "skipped": 1}
    fake_client.get_period_totals.assert_not_called()
    assert int(sync_redis.get(key)) == 1000


def test_reconcile_is_a_no_op_with_no_budgets_configured(sync_redis) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=()))

    result = reconcile_budget_committed_once(redis_client=sync_redis)

    assert result == {"reconciled": 0, "skipped": 0}
