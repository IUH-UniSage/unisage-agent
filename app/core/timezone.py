from datetime import UTC, datetime
from zoneinfo import ZoneInfo

ICT_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def now_utc() -> datetime:
    """Return current datetime in UTC timezone."""
    return datetime.now(UTC)


def now_ict() -> datetime:
    """Return current datetime in ICT (Vietnam GMT+7) timezone."""
    return datetime.now(ICT_TIMEZONE)


def format_iso(dt: datetime | None = None) -> str:
    """Format datetime to ISO-8601 string."""
    if dt is None:
        dt = now_utc()
    return dt.isoformat()
