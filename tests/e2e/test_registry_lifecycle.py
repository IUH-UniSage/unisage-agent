"""Checkpoint: Registry lifecycle (todo.md, section by that exact heading).

Exercises the model-registry business logic for real against the live compose
stack - claim/lease fencing, promotion idempotency, rotation races, the
embedding-identity-free rotation-no-downtime guarantee, stale health reports,
secret non-leakage, and the api-gateway block on `/internal/**` - as opposed
to `test_model_registry_smoke.py`, which only proves the harness itself is
reachable.

This module plays two roles a real deployment splits across two callers:
- `backend_internal_client` (session fixture, conftest.py) plays Python's
  role against backend-java's `/internal/**` namespace (shared secret + CIDR,
  no JWT - see plan.md "Internal API contract").
- `sa_client` (this module) plays a logged-in Super Admin against the
  JWT-gated `/chat-models` namespace, to drive credential rotation the same
  way a human operator would.

Every test here is marked `integration` and only runs inside the
`test-runner` service of `docker-compose.integration.yml` - see README.md.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import asyncpg
import httpx
import pytest
import pytest_asyncio
import redis as redis_sync

pytestmark = pytest.mark.integration

SA_LOGIN_CODE = "SA-001"
SA_LOGIN_PASSWORD = "123123"  # matches DEFAULT_SUPERADMIN_PASS in docker-compose.integration.yml

# Same Postgres the backend-java service itself uses (see that service's
# DB_* env vars in docker-compose.integration.yml) - a separate connection so
# tests can force conditions (an expired lease) that no HTTP endpoint exposes
# on purpose, per todo.md's own fencing-token test skeletons (Task 0.3).
JAVA_DB_DSN = "postgresql://postgres:123456@postgres:5432/assistant_DB"


@pytest.fixture(scope="module")
def sa_client(registry_reset, backend_internal_client: httpx.Client) -> Iterator[httpx.Client]:
    """A `/chat-models` (SA-facing, JWT-gated) client - logs in as the seeded
    Super Admin (DataInitializer, not @Profile-gated, so it exists in every
    profile including `integration`). Depends on `registry_reset` so this
    module always starts from the freshly-seeded state."""

    # httpx.URL str()s with a trailing slash - naive f-string concatenation
    # here produced a literal "//auth/login" (double slash), which the
    # server's routing/filters treat differently from "/auth/login".
    base_url = str(backend_internal_client.base_url).rstrip("/")
    login_resp = httpx.post(
        f"{base_url}/auth/login",
        json={"code": SA_LOGIN_CODE, "password": SA_LOGIN_PASSWORD},
        timeout=10.0,
    )
    login_resp.raise_for_status()
    token = login_resp.json()["data"]["accessToken"]

    with httpx.Client(
        base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=10.0
    ) as client:
        yield client


@pytest_asyncio.fixture
async def java_db_pool() -> Iterator[asyncpg.Pool]:
    """Function-scoped (not module-scoped) - pytest-asyncio's strict mode
    binds each test to its own event loop by default, and a pooled
    connection created on one loop can't be reused safely on another."""

    pool = await asyncpg.create_pool(JAVA_DB_DSN, min_size=1, max_size=3)
    try:
        yield pool
    finally:
        await pool.close()


def _get_chat_model(sa_client: httpx.Client, purpose: str, priority: int | None = None) -> dict:
    resp = sa_client.get("/chat-models", params={"modelPurpose": purpose})
    resp.raise_for_status()
    rows = resp.json()["data"]["data"]
    if priority is not None:
        rows = [r for r in rows if r["priority"] == priority]
    assert rows, f"no seeded {purpose} row (priority={priority}) found - was the seeder run?"
    return rows[0]


def _claim_one(backend_internal_client: httpx.Client) -> dict:
    resp = backend_internal_client.post(
        "/internal/model-registry/verifications/claim", params={"limit": 1}
    )
    resp.raise_for_status()
    jobs = resp.json()
    assert jobs, "expected exactly one claimable job - none returned"
    return jobs[0]


def _submit_result(
    backend_internal_client: httpx.Client,
    job_id: str,
    lease_token: str,
    result_type: str = "OK",
) -> httpx.Response:
    return backend_internal_client.post(
        f"/internal/model-registry/verifications/{job_id}/result",
        json={"leaseToken": lease_token, "resultType": result_type},
    )


def _trigger_new_candidate(
    sa_client: httpx.Client, chat_model: dict, new_api_key: str, new_base_url: str | None = None
) -> None:
    """PUT /chat-models/{id} with a new apiKey - plan.md "Credential rotation":
    never writes the field directly, creates a QUEUED verification job instead."""

    resp = sa_client.put(
        f"/chat-models/{chat_model['id']}",
        json={
            "sourceType": chat_model["sourceType"],
            "llmProvider": chat_model["llmProvider"],
            "llmModelName": chat_model["llmModelName"],
            "modelSourceRef": chat_model.get("modelSourceRef"),
            "apiKey": new_api_key,
            "apiBaseUrl": new_base_url or chat_model["apiBaseUrl"],
            "maxRpm": chat_model["maxRpm"],
            "priority": chat_model.get("priority"),
        },
    )
    resp.raise_for_status()


def _version(backend_internal_client: httpx.Client) -> int:
    resp = backend_internal_client.get("/internal/model-registry/version")
    resp.raise_for_status()
    return resp.json()["version"]


def _snapshot_key_for(backend_internal_client: httpx.Client, chat_model_id: str) -> str | None:
    resp = backend_internal_client.get("/internal/model-registry/snapshot")
    resp.raise_for_status()
    for entries in resp.json()["purposes"].values():
        for entry in entries:
            if entry["id"] == chat_model_id:
                return entry["apiKey"]
    return None


class _RedisChannelCounter:
    """Counts messages seen on MODEL_REGISTRY_CHANNEL from the moment it's
    constructed - used to assert "exactly one publish" per todo.md's
    duplicate-result and rotation-race scenarios."""

    def __init__(self, redis_client: redis_sync.Redis, channel: str) -> None:
        self._pubsub = redis_client.pubsub()
        self._pubsub.subscribe(channel)
        confirm = self._pubsub.get_message(timeout=5)
        assert confirm is not None and confirm["type"] == "subscribe"

    def count_within(self, seconds: float) -> int:
        deadline = time.monotonic() + seconds
        count = 0
        while time.monotonic() < deadline:
            message = self._pubsub.get_message(timeout=0.5)
            if message is not None and message["type"] == "message":
                count += 1
        return count

    def close(self) -> None:
        self._pubsub.close()


# --- Seeder + reset -------------------------------------------------------


def test_seed_reset_produces_expected_purposes_and_status(
    sa_client: httpx.Client, registry_reset_fn
) -> None:
    """ "Seeder đã mở rộng theo schema mới; reset tuần tự vẫn chạy lặp không lỗi" -
    runs the reset a second time (module fixture already ran it once) and
    asserts the resulting rows match plan.md's seed shape: 2x CHAT (priority
    1/2), 1x EMBEDDING, 1x EXTRACTION, all ACTIVE, revision 1."""

    registry_reset_fn()

    for purpose, expected_count in (("CHAT", 2), ("EMBEDDING", 1), ("EXTRACTION", 1)):
        resp = sa_client.get("/chat-models", params={"modelPurpose": purpose})
        resp.raise_for_status()
        rows = resp.json()["data"]["data"]
        assert len(rows) == expected_count, (
            f"{purpose}: expected {expected_count} row(s), got {rows}"
        )
        for row in rows:
            assert row["status"] == "ACTIVE", row
            assert row["revision"] == 1, row
            assert row["hasApiKey"] is True, row


# --- Claim/lease fencing ---------------------------------------------------


@pytest.mark.asyncio
async def test_stale_lease_result_is_rejected_after_expiry(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    java_db_pool: asyncpg.Pool,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "CHAT", priority=1)
    _trigger_new_candidate(sa_client, chat_model, new_api_key="rotated-key-stale-lease")

    job = _claim_one(backend_internal_client)
    old_token = job["leaseToken"]

    # Force the lease into the past directly - no HTTP endpoint exposes this
    # on purpose (todo.md Task 0.3's fencing-token skeleton does the same via
    # direct SQL).
    async with java_db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE chat_model_verifications SET lease_until = now() - interval '1 minute' WHERE id = $1",
            uuid.UUID(job["jobId"]),
        )

    resp = _submit_result(backend_internal_client, job["jobId"], old_token, "OK")
    assert resp.status_code == 409, resp.text


@pytest.mark.asyncio
async def test_lease_reclaimed_after_expiry_old_token_409_new_token_applies(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    java_db_pool: asyncpg.Pool,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "CHAT", priority=1)
    _trigger_new_candidate(sa_client, chat_model, new_api_key="rotated-key-reclaim")

    first_claim = _claim_one(backend_internal_client)
    old_token = first_claim["leaseToken"]

    async with java_db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE chat_model_verifications SET lease_until = now() - interval '1 minute' WHERE id = $1",
            uuid.UUID(first_claim["jobId"]),
        )

    second_claim = _claim_one(backend_internal_client)
    assert second_claim["jobId"] == first_claim["jobId"]
    new_token = second_claim["leaseToken"]
    assert new_token != old_token

    stale_resp = _submit_result(backend_internal_client, first_claim["jobId"], old_token, "OK")
    assert stale_resp.status_code == 409, stale_resp.text

    fresh_resp = _submit_result(backend_internal_client, first_claim["jobId"], new_token, "OK")
    assert fresh_resp.status_code == 200, fresh_resp.text
    assert fresh_resp.json()["applied"] is True


# --- Duplicate result idempotency -----------------------------------------


def test_duplicate_result_applied_once_revision_and_version_bump_once(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    registry_redis: redis_sync.Redis,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "CHAT", priority=2)
    version_before = _version(backend_internal_client)

    _trigger_new_candidate(sa_client, chat_model, new_api_key="rotated-key-duplicate")
    job = _claim_one(backend_internal_client)

    counter = _RedisChannelCounter(registry_redis, channel="model-registry:updates")
    try:
        first = _submit_result(backend_internal_client, job["jobId"], job["leaseToken"], "OK")
        assert first.status_code == 200, first.text
        assert first.json() == {"applied": True, "duplicate": False}

        second = _submit_result(backend_internal_client, job["jobId"], job["leaseToken"], "OK")
        assert second.status_code == 200, second.text
        assert second.json() == {"applied": False, "duplicate": True}

        message_count = counter.count_within(3.0)
    finally:
        counter.close()

    assert message_count == 1, (
        f"expected exactly 1 publish on the registry channel, saw {message_count}"
    )

    version_after = _version(backend_internal_client)
    assert version_after == version_before + 1

    updated = sa_client.get(f"/chat-models/{chat_model['id']}").raise_for_status().json()["data"]
    assert updated["revision"] == chat_model["revision"] + 1


# --- Rotation race + no-downtime -------------------------------------------


def test_rotation_race_superseded_job_result_rejected_no_promotion_no_event(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    registry_redis: redis_sync.Redis,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "EXTRACTION")
    original_key = _snapshot_key_for(backend_internal_client, chat_model["id"])
    assert original_key is not None

    _trigger_new_candidate(sa_client, chat_model, new_api_key="rotation-race-key-n")
    job_n = _claim_one(backend_internal_client)

    # SA edits the credential again while job N is still RUNNING - generation
    # bumps to N+1 and job N becomes SUPERSEDED in the same transaction
    # (plan.md "Credential rotation").
    _trigger_new_candidate(sa_client, chat_model, new_api_key="rotation-race-key-n-plus-1")

    counter = _RedisChannelCounter(registry_redis, channel="model-registry:updates")
    try:
        # job N's lease is still valid and its token is still correct - the
        # only thing that changed is candidate_generation - so this must 409,
        # never promote key-n, per todo.md R4.2/plan.md "Verify OK" note.
        stale_ok = _submit_result(
            backend_internal_client, job_n["jobId"], job_n["leaseToken"], "OK"
        )
        assert stale_ok.status_code == 409, stale_ok.text
        message_count = counter.count_within(2.0)
    finally:
        counter.close()
    assert message_count == 0, "SUPERSEDED job's result must never publish a registry-change event"

    # Snapshot still shows the ORIGINAL key - key-n was never promoted, and
    # neither was key-n-plus-1 (its own job hasn't been verified yet).
    assert _snapshot_key_for(backend_internal_client, chat_model["id"]) == original_key

    # Now let job N+1 (the current one) succeed - only then does the key change.
    job_n_plus_1 = _claim_one(backend_internal_client)
    ok_resp = _submit_result(
        backend_internal_client, job_n_plus_1["jobId"], job_n_plus_1["leaseToken"], "OK"
    )
    assert ok_resp.status_code == 200, ok_resp.text
    assert ok_resp.json()["applied"] is True

    assert (
        _snapshot_key_for(backend_internal_client, chat_model["id"]) == "rotation-race-key-n-plus-1"
    )


def test_rotation_has_no_downtime_snapshot_keeps_old_key_while_candidate_pending(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "CHAT", priority=1)
    original_key = _snapshot_key_for(backend_internal_client, chat_model["id"])
    assert original_key is not None

    _trigger_new_candidate(sa_client, chat_model, new_api_key="no-downtime-candidate-key")

    # Candidate exists (job QUEUED, not yet claimed/verified) - snapshot must
    # still serve the OLD key, not error, not the candidate.
    assert _snapshot_key_for(backend_internal_client, chat_model["id"]) == original_key

    job = _claim_one(backend_internal_client)
    # Still pending (claimed but no result submitted yet) - still the old key.
    assert _snapshot_key_for(backend_internal_client, chat_model["id"]) == original_key

    ok_resp = _submit_result(backend_internal_client, job["jobId"], job["leaseToken"], "OK")
    assert ok_resp.status_code == 200, ok_resp.text

    assert (
        _snapshot_key_for(backend_internal_client, chat_model["id"]) == "no-downtime-candidate-key"
    )


# --- Stale health report ----------------------------------------------------


def test_stale_health_report_is_not_applied(
    sa_client: httpx.Client,
    backend_internal_client: httpx.Client,
    registry_reset_fn,
) -> None:
    registry_reset_fn()
    chat_model = _get_chat_model(sa_client, "CHAT", priority=1)
    stale_revision = chat_model["revision"] - 1
    assert stale_revision >= 0

    resp = backend_internal_client.post(
        f"/internal/model-registry/credentials/{chat_model['id']}/health",
        json={
            "credentialRevision": stale_revision,
            "snapshotVersion": _version(backend_internal_client),
            "errorType": "PERMANENT",
            "errorCode": "invalid_api_key",
            "message": "test-induced stale health report",
            "occurredAt": "2026-01-01T00:00:00Z",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"applied": False}

    # Row must be unaffected - a stale report must never DISABLE it.
    unchanged = sa_client.get(f"/chat-models/{chat_model['id']}").raise_for_status().json()["data"]
    assert unchanged["status"] == "ACTIVE"


# --- Secret non-leakage -----------------------------------------------------


def test_seeded_key_plaintext_never_appears_in_sa_responses_or_java_logs(
    sa_client: httpx.Client,
    docker_client,
    registry_reset_fn,
) -> None:
    registry_reset_fn()

    # Every SA-facing (non-/internal/**) response captured across this
    # module's own test run, re-fetched fresh here, must never carry a
    # seeded key's plaintext - only `hasApiKey` (bool).
    canaries = ["seed-chat-key-1", "seed-chat-key-2", "seed-embedding-key", "seed-extraction-key"]
    for purpose in ("CHAT", "EMBEDDING", "EXTRACTION"):
        resp = sa_client.get("/chat-models", params={"modelPurpose": purpose})
        resp.raise_for_status()
        body = resp.text
        for canary in canaries:
            assert canary not in body, (
                f"seed key leaked via SA-facing /chat-models?modelPurpose={purpose}"
            )

    container = docker_client.containers.get("e2e-backend-java-1")
    logs = container.logs(tail=5000).decode("utf-8", errors="replace")
    for canary in canaries:
        assert canary not in logs, f"seed key {canary!r} leaked into backend-java container logs"


# --- Gateway blocks the internal path ---------------------------------------


def test_gateway_blocks_internal_snapshot_path() -> None:
    with httpx.Client(base_url="http://api-gateway:8889", timeout=10.0) as client:
        resp = client.get("/api/v1/master/internal/model-registry/snapshot")
    assert resp.status_code == 404, resp.text
