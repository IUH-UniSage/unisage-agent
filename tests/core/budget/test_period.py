from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.budget.period import current_period_key, period_ttl_seconds


def test_daily_period_key_crosses_at_midnight_in_app_timezone() -> None:
    tz = ZoneInfo("Asia/Ho_Chi_Minh")
    before_midnight = datetime(2026, 9, 27, 23, 59, 59, tzinfo=tz)
    after_midnight = datetime(2026, 9, 28, 0, 0, 1, tzinfo=tz)

    assert current_period_key("DAILY", now=before_midnight) == "2026-09-27"
    assert current_period_key("DAILY", now=after_midnight) == "2026-09-28"


def test_monthly_period_key_crosses_at_month_boundary() -> None:
    tz = ZoneInfo("Asia/Ho_Chi_Minh")
    end_of_month = datetime(2026, 9, 30, 23, 59, 59, tzinfo=tz)
    start_of_next = datetime(2026, 10, 1, 0, 0, 1, tzinfo=tz)

    assert current_period_key("MONTHLY", now=end_of_month) == "2026-09"
    assert current_period_key("MONTHLY", now=start_of_next) == "2026-10"


def test_period_ttl_is_end_of_period_plus_three_days() -> None:
    ttl = period_ttl_seconds("DAILY", "2026-09-27")
    # From well within 2026-09-27 to 2026-09-30 00:00 VN time is under 4 days.
    assert 0 < ttl <= 4 * 24 * 3600
