"""Fixtures for the `integration` test suite - only usable inside the
`test-runner` service of `docker-compose.integration.yml` (see README.md).

Everything here assumes the compose network's env vars are present
(BACKEND_JAVA_INTERNAL_BASE_URL, FAKE_LLM_PROVIDER_ADMIN_BASE_URL,
CELERY_BEAT_CONTAINER_NAME, ...) - importing this file outside that container
without them will raise at fixture-use time with a clear message, not
silently degrade.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

import docker
import httpx
import pytest
import redis as redis_sync

from app.core.config import settings
from app.worker.celery_app import celery_app

# --- Env this conftest needs, beyond what app.core.config.settings already
# covers (those are test-runner-only, not shared with the app's own settings
# surface - see docker-compose.integration.yml's test-runner service). ---


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set - this fixture only runs inside the test-runner "
            f"service of docker-compose.integration.yml (see tests/e2e/README.md)"
        )
    return value


BACKEND_JAVA_INTERNAL_BASE_URL = "BACKEND_JAVA_INTERNAL_BASE_URL"
FAKE_LLM_PROVIDER_ADMIN_BASE_URL = "FAKE_LLM_PROVIDER_ADMIN_BASE_URL"
FAKE_LLM_PROVIDER_BASE_URL = "FAKE_LLM_PROVIDER_BASE_URL"
UNISAGE_AGENT_BASE_URL = "UNISAGE_AGENT_BASE_URL"
CELERY_BEAT_CONTAINER_NAME = "CELERY_BEAT_CONTAINER_NAME"

REGISTRY_RESET_TIMEOUT_S = 30
REGISTRY_KEY_PREFIX = "mr:"


def pytest_configure(config: pytest.Config) -> None:
    # Belt-and-suspenders registration - the canonical registration lives in
    # pyproject.toml's [tool.pytest.ini_options], but tests/e2e is also meant
    # to be runnable as a standalone directory inside the test-runner
    # container without depending on that file being picked up from the
    # container's rootdir the same way it is locally.
    config.addinivalue_line(
        "markers",
        "integration: requires the docker-compose.integration.yml stack - see tests/e2e/README.md",
    )


@pytest.fixture(scope="session")
def docker_client() -> Iterator[docker.DockerClient]:
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def registry_redis() -> Iterator[redis_sync.Redis]:
    """Client for the registry/circuit-breaker/lock Redis DB (DB 0) - never
    the Celery broker/backend DBs."""

    client = redis_sync.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def backend_internal_client() -> Iterator[httpx.Client]:
    """Calls backend-java's `/internal/**` directly - this container plays the
    role of "Python" for the internal API contract (todo.md's Checkpoint:
    Registry lifecycle), gated the same way real Python is: shared secret
    header, CIDR allowlist (this network's own subnet)."""

    base_url = _required_env(BACKEND_JAVA_INTERNAL_BASE_URL)
    with httpx.Client(
        base_url=base_url,
        headers={"X-Internal-Secret": settings.APP_INTERNAL_SECRET_KEY},
        timeout=10.0,
    ) as client:
        yield client


@pytest.fixture(scope="session")
def fake_provider_admin_client() -> Iterator[httpx.Client]:
    base_url = _required_env(FAKE_LLM_PROVIDER_ADMIN_BASE_URL)
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        yield client


@pytest.fixture(scope="session")
def fake_provider_client() -> Iterator[httpx.Client]:
    """Plain (non-admin) surface - `/healthz`, `/v1/chat/completions`, ..."""

    base_url = _required_env(FAKE_LLM_PROVIDER_BASE_URL)
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        yield client


@pytest.fixture(scope="session")
def unisage_agent_client() -> Iterator[httpx.Client]:
    base_url = _required_env(UNISAGE_AGENT_BASE_URL)
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        yield client


def _purge_harness_queue() -> None:
    """`celery purge -Q <this run's queue>` - never a broker-wide purge/flush.

    Uses kombu's low-level `queue_purge` (rather than
    `celery_app.control.purge()`, which purges every queue this app knows
    about) so a purge here can never reach another run's or another
    service's queue, even if they happen to share the same Redis broker DB.
    """

    queue_name = celery_app.conf.task_default_queue
    with celery_app.connection_for_write() as conn:
        conn.default_channel.queue_purge(queue_name)


def _wait_for_no_active_tasks(timeout_s: float = REGISTRY_RESET_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    inspect = celery_app.control.inspect(timeout=2)
    while time.monotonic() < deadline:
        active = inspect.active() or {}
        if not any(tasks for tasks in active.values()):
            return
        time.sleep(1)
    raise TimeoutError(
        f"celery worker still reported active tasks after {timeout_s}s - "
        "refusing to reset half-way through a running task"
    )


def _clear_registry_redis_keys(client: redis_sync.Redis) -> None:
    """Deletes every `mr:*` key - never `FLUSHDB` (this Redis DB is shared
    with nothing else on DB 0, but the rule is "never FLUSHDB" regardless,
    per plan.md - a future key on DB 0 that isn't `mr:*` must survive this)."""

    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=f"{REGISTRY_KEY_PREFIX}*", count=500)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break


def _call_java_reset(client: httpx.Client) -> bool:
    """Calls `POST /internal/test/registry/reset`. Returns True if it ran.

    That endpoint (and the `ModelRegistryIntegrationSeeder` bean behind it)
    do not exist in backend-java yet (todo.md Task 0.5's Java-side piece -
    out of scope for this change, tracked separately). Until it lands, this
    degrades to a no-op with a clear warning rather than failing or hanging
    the fixture - the point of the smoke test is to prove the *infra* resets
    cleanly, which does not depend on that endpoint existing.
    """

    try:
        resp = client.post("/internal/test/registry/reset")
    except httpx.HTTPError as exc:
        import warnings

        warnings.warn(
            f"POST /internal/test/registry/reset unreachable ({exc!r}) - "
            "backend-java's Task 0.5 reset endpoint/seeder is not implemented yet, "
            "skipping the DB-level reset step",
            stacklevel=2,
        )
        return False

    if resp.status_code == 404:
        import warnings

        warnings.warn(
            "POST /internal/test/registry/reset -> 404 - backend-java has not added the "
            '@Profile("integration") reset endpoint yet, skipping the DB-level reset step',
            stacklevel=2,
        )
        return False

    if resp.status_code == 409:
        # A verification job is still RUNNING with an unexpired lease - the
        # caller (registry_reset fixture) is expected to have already waited
        # for `_wait_for_no_active_tasks`, so a 409 here means Java's own
        # lease bookkeeping disagrees; surface it rather than retry forever.
        raise RuntimeError(
            "POST /internal/test/registry/reset -> 409 (job still RUNNING with an "
            "unexpired lease) after celery reported no active tasks"
        )

    resp.raise_for_status()
    return True


@pytest.fixture(scope="session")
def registry_reset_fn(
    docker_client: docker.DockerClient,
    registry_redis: redis_sync.Redis,
    backend_internal_client: httpx.Client,
    fake_provider_admin_client: httpx.Client,
) -> Iterator[Callable[[], None]]:
    """Returns a callable that runs todo.md's 7-step sequential reset once per call.

    A callable, not a value, on purpose: some tests (this task's own smoke
    test) need to run the reset more than once *within the same test* to
    prove it's safe to repeat, which a cached fixture value can't express.
    `registry_reset` below is the convenience wrapper for the common case
    of "just give me a clean slate before this module's tests run".
    """

    def _reset_once() -> None:
        beat_name = _required_env(CELERY_BEAT_CONTAINER_NAME)

        # 1. Stop Beat - no new verification jobs get scheduled mid-reset.
        beat_container = docker_client.containers.get(beat_name)
        beat_container.stop(timeout=10)

        try:
            # 2. Purge only this run's queue.
            _purge_harness_queue()
            # 3. Wait for any already-active task to drain.
            _wait_for_no_active_tasks()
            # 4. Java-side reset (best-effort until Task 0.5's Java half lands).
            _call_java_reset(backend_internal_client)
            # 5. Registry-namespaced Redis keys only.
            _clear_registry_redis_keys(registry_redis)
            # 6. Fake provider back to its default mode/key.
            fake_provider_admin_client.post("/reset")
        finally:
            # 7. Beat back up - always, even if a step above raised, so one
            # failed reset doesn't leave every later test without Beat.
            beat_container.start()

    yield _reset_once


@pytest.fixture(scope="module")
def registry_reset(registry_reset_fn: Callable[[], None]) -> None:
    """Convenience wrapper: resets once at the start of this test module."""

    registry_reset_fn()
