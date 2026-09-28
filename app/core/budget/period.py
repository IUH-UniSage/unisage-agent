"""Calendar DAILY/MONTHLY `periodKey` generation, cut in `settings.APP_TIMEZONE`.
Must produce the exact same string Java's `UsagePeriodCalculator.periodBoundsUtc()`
parses back (`yyyy-MM-dd` for DAILY, `yyyy-MM` for MONTHLY) - the two sides never
exchange this value over the wire except as a plain string, so format drift here
would silently point Java's reconciliation at the wrong period.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.core.config import settings


def current_period_key(period: str, *, now: datetime | None = None) -> str:
    local_now = (now or datetime.now(ZoneInfo(settings.APP_TIMEZONE))).astimezone(
        ZoneInfo(settings.APP_TIMEZONE)
    )
    if period == "DAILY":
        return local_now.date().isoformat()
    if period == "MONTHLY":
        return local_now.strftime("%Y-%m")
    raise ValueError(f"unknown budget period: {period!r}")


def period_ttl_seconds(period: str, period_key: str) -> int:
    """Seconds until 3 days after this period ends. Computed from `period_key` itself
    (not "now"), so a TTL set once at reservation time stays correct even if the
    process clock drifts across a period boundary before the key would otherwise
    expire."""

    end_date = _period_end_date(period, period_key)
    end_dt = _combine_end_of_day(end_date)
    expiry = end_dt + timedelta(days=3)
    now = datetime.now(ZoneInfo(settings.APP_TIMEZONE))
    return max(1, int((expiry - now).total_seconds()))


def _period_end_date(period: str, period_key: str) -> date:
    if period == "DAILY":
        return date.fromisoformat(period_key) + timedelta(days=1)
    if period == "MONTHLY":
        year, month = (int(part) for part in period_key.split("-"))
        if month == 12:
            return date(year + 1, 1, 1)
        return date(year, month + 1, 1)
    raise ValueError(f"unknown budget period: {period!r}")


def _combine_end_of_day(end_date: date) -> datetime:
    return datetime.combine(end_date, datetime.min.time(), tzinfo=ZoneInfo(settings.APP_TIMEZONE))
