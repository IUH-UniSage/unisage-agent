"""BudgetTracker tests against a real local Redis (EVAL/Lua semantics can't be
faithfully faked) - flushes db 0 before each test, so this must never point at a
shared/production Redis instance."""

import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio
import redis.asyncio as redis_asyncio

from app.core.budget.snapshot import BudgetEntry, BudgetSnapshot, set_current_budget_snapshot
from app.core.budget.tracker import BudgetTracker, to_micro_usd


@pytest_asyncio.fixture
async def redis_client():
    client = redis_asyncio.Redis.from_url("redis://localhost:6379/0")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
def tracker(redis_client):
    return BudgetTracker(redis_client)


def _system_block(limit_usd: str, period: str = "DAILY") -> BudgetEntry:
    return BudgetEntry(
        scope="SYSTEM",
        scope_provider=None,
        scope_purpose=None,
        period=period,
        limit_usd=Decimal(limit_usd),
        action="BLOCK",
        throttle_max_concurrency=None,
    )


def _system_throttle(limit_usd: str, cap: int, period: str = "DAILY") -> BudgetEntry:
    return BudgetEntry(
        scope="SYSTEM",
        scope_provider=None,
        scope_purpose=None,
        period=period,
        limit_usd=Decimal(limit_usd),
        action="THROTTLE",
        throttle_max_concurrency=cap,
    )


def test_to_micro_usd_rounds_half_up() -> None:
    assert to_micro_usd(Decimal("0.0000015")) == 2
    assert to_micro_usd(Decimal("0.0000014")) == 1


@pytest.mark.asyncio
async def test_drift_free_across_ten_thousand_small_commits(redis_client, tracker) -> None:
    """10,000 x $0.0000015 committed one at a time must sum to exactly the
    round-half-up total - proof INCRBY on a micro-USD integer never drifts the way
    INCRBYFLOAT would."""

    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block("1000"),)))

    for i in range(10_000):
        request_id = f"drift-{i}"
        await tracker.reserve_request(
            request_id=request_id, purpose="CHAT", estimate_usd=Decimal("0.0000015")
        )
        await tracker.settle_request(request_id=request_id, actual_total_usd=Decimal("0.0000015"))

    committed = await redis_client.get("budget:committed:SYSTEM:DAILY:" + _today())
    assert int(committed) == 10_000 * to_micro_usd(Decimal("0.0000015"))


@pytest.mark.asyncio
async def test_block_budget_rejects_once_exhausted(tracker) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block("0.0001"),)))

    r1 = await tracker.reserve_request(
        request_id="r1", purpose="CHAT", estimate_usd=Decimal("0.00005")
    )
    r2 = await tracker.reserve_request(
        request_id="r2", purpose="CHAT", estimate_usd=Decimal("0.00005")
    )
    r3 = await tracker.reserve_request(
        request_id="r3", purpose="CHAT", estimate_usd=Decimal("0.00005")
    )

    assert r1 == "OK"
    assert r2 == "OK"
    assert r3 == "REJECT_EXCEEDED"


@pytest.mark.asyncio
async def test_fifty_concurrent_reservations_against_a_block_budget_sized_for_ten(tracker) -> None:
    """50 concurrent reserve calls against a budget that fits exactly 10 estimates -
    at most 10 may succeed, proving the Lua script's check-then-write is atomic
    under real concurrency, not just sequential calls."""

    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block("0.0010"),)))
    estimate = Decimal("0.0001")  # limit / estimate = 10

    async def attempt(i: int) -> str:
        return await tracker.reserve_request(
            request_id=f"concurrent-{i}", purpose="CHAT", estimate_usd=estimate
        )

    results = await asyncio.gather(*(attempt(i) for i in range(50)))
    ok_count = sum(1 for r in results if r == "OK")
    assert ok_count == 10
    assert all(r in ("OK", "REJECT_EXCEEDED") for r in results)


@pytest.mark.asyncio
async def test_throttle_denies_once_inflight_hits_cap_after_limit_reached(
    redis_client, tracker
) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_throttle("0.0001", 2),)))

    # Fill the limit first via a settled (committed) request, then keep 2 requests
    # inflight so the 3rd is throttled.
    await tracker.reserve_request(request_id="fill", purpose="CHAT", estimate_usd=Decimal("0.0001"))
    await tracker.settle_request(request_id="fill", actual_total_usd=Decimal("0.0001"))

    r1 = await tracker.reserve_request(
        request_id="t1", purpose="CHAT", estimate_usd=Decimal("0.00001")
    )
    r2 = await tracker.reserve_request(
        request_id="t2", purpose="CHAT", estimate_usd=Decimal("0.00001")
    )
    r3 = await tracker.reserve_request(
        request_id="t3", purpose="CHAT", estimate_usd=Decimal("0.00001")
    )

    assert r1 == "OK"
    assert r2 == "OK"
    assert r3 == "REJECT_THROTTLED"


@pytest.mark.asyncio
async def test_release_provider_is_idempotent(redis_client, tracker) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=()))

    await tracker.acquire_provider(
        request_id="idem", seq=1, provider="openai", estimate_usd=Decimal("0.00002")
    )
    await tracker.release_provider(request_id="idem", seq=1, actual_usd=Decimal("0.00001"))
    await tracker.release_provider(request_id="idem", seq=1, actual_usd=Decimal("0.00001"))

    committed = await redis_client.get("budget:committed:PROVIDER:openai:DAILY:" + _today())
    assert int(committed) == to_micro_usd(Decimal("0.00001"))


@pytest.mark.asyncio
async def test_settle_request_is_idempotent(redis_client, tracker) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=(_system_block("10"),)))

    await tracker.reserve_request(
        request_id="idem2", purpose="CHAT", estimate_usd=Decimal("0.0001")
    )
    await tracker.settle_request(request_id="idem2", actual_total_usd=Decimal("0.00005"))
    await tracker.settle_request(request_id="idem2", actual_total_usd=Decimal("0.00005"))

    committed = await redis_client.get("budget:committed:SYSTEM:DAILY:" + _today())
    assert int(committed) == to_micro_usd(Decimal("0.00005"))
    assert await redis_client.exists("budget:resv:idem2") == 0


@pytest.mark.asyncio
async def test_scope_with_both_daily_and_monthly_budgets_shares_one_inflight_counter(
    redis_client, tracker
) -> None:
    """A SYSTEM scope with a DAILY and a MONTHLY budget enabled at once must only
    increment its shared inflight key once per reservation, not once per pair -
    otherwise a THROTTLE cap of N would only ever admit N/2 concurrent requests."""

    set_current_budget_snapshot(
        BudgetSnapshot(
            version=1,
            entries=(_system_throttle("100", 5, "DAILY"), _system_throttle("100", 5, "MONTHLY")),
        )
    )

    await tracker.reserve_request(request_id="dual1", purpose="CHAT", estimate_usd=Decimal("0.01"))

    inflight = await redis_client.get("budget:inflight:SYSTEM")
    assert int(inflight) == 1


@pytest.mark.asyncio
async def test_no_configured_budget_is_a_pure_pass_through(tracker) -> None:
    set_current_budget_snapshot(BudgetSnapshot(version=1, entries=()))

    result = await tracker.reserve_request(
        request_id="none", purpose="CHAT", estimate_usd=Decimal("999")
    )
    assert result == "OK"


def _today() -> str:
    from app.core.budget.period import current_period_key

    return current_period_key("DAILY")
