"""Infra smoke test for the Task 0.5 integration harness - NOT a business-logic
(registry lifecycle) test.

Only proves the harness itself works: every service is reachable, Redis
pub/sub round-trips across containers, Beat is actually ticking, the fake
provider answers from inside the agent network, and the reset fixture can run
twice back to back without error. It deliberately does NOT assert anything
about seeded `ChatModel` rows, `modelPurpose`/`status`/`revision`, or any
other DB-backed registry state - `ModelRegistryIntegrationSeeder` and
`POST /internal/test/registry/reset` (backend-java, todo.md Task 0.5's Java
half) are not implemented yet. Registry-lifecycle acceptance lives in
"Checkpoint: Registry lifecycle" (todo.md) once that lands.

Every test here is marked `integration` and only runs inside the
`test-runner` service of `docker-compose.integration.yml` - see README.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
import pytest
import redis as redis_sync

from app.core.config import settings
from app.worker.celery_app import BEAT_HEARTBEAT_REDIS_KEY

pytestmark = pytest.mark.integration


def test_backend_java_internal_version_endpoint_is_reachable(
    backend_internal_client: httpx.Client,
) -> None:
    """Reachability only - NOT a business-logic assertion.

    Asserts the internal API contract (shared-secret + CIDR gate, per Task
    0.1) lets this network's test-runner through and the endpoint answers.
    Does not assert what version number comes back, and does not assert any
    ChatModel is seeded - that needs backend-java's seeder (out of scope
    here, see the module docstring).
    """

    resp = backend_internal_client.get("/internal/model-registry/version")
    assert resp.status_code == 200, (
        f"/internal/model-registry/version -> {resp.status_code}: {resp.text[:300]}"
    )


def test_unisage_agent_health_endpoint_is_reachable(unisage_agent_client: httpx.Client) -> None:
    resp = unisage_agent_client.get("/api/v1/health")
    assert resp.status_code == 200


def test_fake_provider_reachable_from_agent_network(fake_provider_client: httpx.Client) -> None:
    resp = fake_provider_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_redis_pubsub_round_trips_on_the_registry_channel(
    registry_redis: redis_sync.Redis,
) -> None:
    """Publishes on `MODEL_REGISTRY_CHANNEL` and reads it back - proves the
    pub/sub path Java's "publish after commit" (plan.md) will use later is
    actually wired, without needing Java to publish anything itself yet."""

    pubsub = registry_redis.pubsub()
    pubsub.subscribe(settings.MODEL_REGISTRY_CHANNEL)
    try:
        # First message after subscribe is always the subscribe confirmation.
        confirm = pubsub.get_message(timeout=5)
        assert confirm is not None and confirm["type"] == "subscribe"

        registry_redis.publish(settings.MODEL_REGISTRY_CHANNEL, '{"version": 1}')

        message = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = pubsub.get_message(timeout=1)
            if message is not None and message["type"] == "message":
                break
        assert message is not None and message["type"] == "message"
        assert message["data"] == b'{"version": 1}'
    finally:
        pubsub.close()


def test_celery_beat_has_ticked_at_least_once(registry_redis: redis_sync.Redis) -> None:
    """Polls the heartbeat key `beat_heartbeat` writes (app/worker/celery_app.py)
    rather than parsing container logs - a real signal Beat's scheduler loop
    actually ran a task, not just that the process started."""

    deadline = time.monotonic() + 15
    last_tick: str | None = None
    while time.monotonic() < deadline:
        last_tick = registry_redis.get(BEAT_HEARTBEAT_REDIS_KEY)
        if last_tick is not None:
            break
        time.sleep(1)

    assert last_tick is not None, (
        "celery-beat never wrote to "
        f"{BEAT_HEARTBEAT_REDIS_KEY!r} within 15s - is the celery-beat service running?"
    )
    assert time.time() - float(last_tick) < 15, "heartbeat key is stale, not a recent tick"


def test_fixture_reset_runs_twice_in_a_row_without_error(
    registry_reset_fn: Callable[[], None],
) -> None:
    """Task 0.5 acceptance: "fixture reset chạy 2 lần liên tiếp không lỗi".

    Runs the full 7-step sequential reset (stop Beat, purge this run's
    queue, drain active tasks, best-effort Java reset, clear `mr:*` keys,
    reset the fake provider, restart Beat) twice back to back. The Java
    reset step degrades to a warning rather than raising until backend-java's
    reset endpoint exists (see conftest.py's `_call_java_reset`) - this test
    still needs to pass on the infra parts alone.
    """

    registry_reset_fn()
    registry_reset_fn()
