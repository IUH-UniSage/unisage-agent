"""Tests for app/core/usage/usage_outbox.py (enqueue side only; the Celery drain worker
is a separate piece, not covered here)."""

from typing import Any

import pytest

from app.core.usage.usage_outbox import OUTBOX_KEY, enqueue_usage_payload


class _FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.pushed: list[tuple[str, Any]] = []
        self._fail = fail

    async def lpush(self, name: str, *values: Any) -> Any:
        if self._fail:
            raise ConnectionError("redis down")
        self.pushed.append((name, values))
        return len(values)

    async def aclose(self) -> Any:
        return None


@pytest.mark.asyncio
async def test_enqueue_pushes_json_onto_the_outbox_key() -> None:
    redis_client = _FakeRedis()
    await enqueue_usage_payload({"requestId": "r1", "lines": []}, redis_client=redis_client)

    assert len(redis_client.pushed) == 1
    key, values = redis_client.pushed[0]
    assert key == OUTBOX_KEY
    assert '"requestId": "r1"' in values[0]


@pytest.mark.asyncio
async def test_enqueue_never_raises_when_redis_is_unreachable() -> None:
    redis_client = _FakeRedis(fail=True)

    # Must not raise - a Redis outage cannot be allowed to break the calling request.
    await enqueue_usage_payload({"requestId": "r1", "lines": []}, redis_client=redis_client)
