"""Verify-before-active claim loop (plan.md "Verification lifecycle") — exercised entirely
against fake `BackendJavaClient`-shaped stubs, never a live Java or provider.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from app.core.llm.embedding_probe import EmbeddingFingerprint
from app.core.ssrf_guard import SsrfBlockedError
from app.integrations.backend_java_client import BackendJavaConnectionError, BackendJavaHTTPError
from app.worker import verification_tasks


class _FakeClient:
    """Structural stand-in for `BackendJavaClient` — records every `post_verification_result`
    call's kwargs and lets a test script `claim_verifications`/`post_verification_result`
    behavior without any network."""

    def __init__(
        self,
        jobs: list[dict[str, Any]],
        *,
        result_side_effects: list[Any] | None = None,
    ) -> None:
        self.jobs = jobs
        self.result_calls: list[dict[str, Any]] = []
        self._result_side_effects = list(result_side_effects or [])

    async def claim_verifications(self, *, limit: int) -> list[dict[str, Any]]:
        self.claimed_limit = limit
        return self.jobs

    async def post_verification_result(self, **kwargs: Any) -> dict[str, Any]:
        self.result_calls.append(kwargs)
        if self._result_side_effects:
            effect = self._result_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
        return {"applied": True, "duplicate": False}


def _chat_job(job_id: str = "job-1", lease_token: str = "lease-1") -> dict[str, Any]:
    return {
        "jobId": job_id,
        "leaseToken": lease_token,
        "attempt": 1,
        "leaseUntil": "2026-09-26T00:01:00",
        "credential": {
            "chatModelId": "cm-1",
            "modelPurpose": "CHAT",
            "sourceType": "CLOUD_API",
            "provider": "openai",
            "modelName": "gpt-4o-mini",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "apiKey": "sk-canary-secret-value",
            "maxRpm": 500,
        },
    }


def _embedding_job(job_id: str = "job-2", lease_token: str = "lease-2") -> dict[str, Any]:
    job = _chat_job(job_id=job_id, lease_token=lease_token)
    job["credential"]["modelPurpose"] = "EMBEDDING"
    job["credential"]["modelName"] = "text-embedding-3-small"
    return job


@pytest.mark.asyncio
async def test_success_path_submits_ok_with_matching_lease_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    client = _FakeClient([_chat_job(job_id="job-1", lease_token="lease-1")])

    claimed = await verification_tasks.run_verification_batch(client=client, limit=5)

    assert claimed == 1
    assert client.claimed_limit == 5
    assert len(client.result_calls) == 1
    call = client.result_calls[0]
    assert call["job_id"] == "job-1"
    assert call["lease_token"] == "lease-1"
    assert call["result_type"] == "OK"


@pytest.mark.asyncio
async def test_transient_provider_failure_submits_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential
        raise ConnectionError("connect timed out")  # unrecognized -> classify_llm_error fails open TRANSIENT

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    client = _FakeClient([_chat_job()])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert client.result_calls[0]["result_type"] == "TRANSIENT"


@pytest.mark.asyncio
async def test_permanent_provider_failure_submits_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential
        raise SsrfBlockedError("blocked host")  # classify_llm_error always -> PERMANENT

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    client = _FakeClient([_chat_job()])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert client.result_calls[0]["result_type"] == "PERMANENT"


@pytest.mark.asyncio
async def test_failure_message_is_redacted_never_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-canary-secret-value"

    async def fake_completion(credential: Any) -> None:
        raise ValueError(f"authentication failed for key {credential.api_key}")

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    client = _FakeClient([_chat_job()])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    call = client.result_calls[0]
    assert secret not in call["message"]
    assert "[REDACTED]" in call["message"]
    assert call["error_code"] == "ValueError"


@pytest.mark.asyncio
async def test_embedding_job_result_includes_dimension_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = EmbeddingFingerprint(dimension=2, vectors=((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)))

    def fake_measure(credential: Any) -> EmbeddingFingerprint:
        del credential
        return fingerprint

    monkeypatch.setattr(verification_tasks, "_measure_embedding_fingerprint_sync", fake_measure)

    client = _FakeClient([_embedding_job(job_id="job-2", lease_token="lease-2")])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    call = client.result_calls[0]
    assert call["job_id"] == "job-2"
    assert call["result_type"] == "OK"
    assert call["embedding_dimension"] == 2
    assert call["embedding_fingerprint"] == fingerprint.flattened()


@pytest.mark.asyncio
async def test_409_on_result_submission_logs_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    lease_lost = BackendJavaHTTPError("POST", "/x", 409, {"error": "VERIFICATION_LEASE_LOST"})
    client = _FakeClient([_chat_job()], result_side_effects=[lease_lost])

    # Must not raise - a 409 is expected, handled, and terminal for this job.
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert len(client.result_calls) == 1


@pytest.mark.asyncio
async def test_network_error_on_result_submission_retries_with_same_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)
    monkeypatch.setattr(verification_tasks, "_RESULT_SUBMIT_RETRY_DELAY_SECONDS", 0.0)

    network_error = BackendJavaConnectionError("POST", "/x", ConnectionError("refused"))
    client = _FakeClient(
        [_chat_job(lease_token="lease-retry")],
        result_side_effects=[network_error, network_error],
    )

    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert len(client.result_calls) == 3
    assert all(call["lease_token"] == "lease-retry" for call in client.result_calls)


@pytest.mark.asyncio
async def test_network_error_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(credential: Any) -> None:
        del credential

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)
    monkeypatch.setattr(verification_tasks, "_RESULT_SUBMIT_RETRY_DELAY_SECONDS", 0.0)

    network_error = BackendJavaConnectionError("POST", "/x", ConnectionError("refused"))
    client = _FakeClient(
        [_chat_job()],
        result_side_effects=[network_error, network_error, network_error],
    )

    # Must not raise even after exhausting retries - the batch moves on.
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert len(client.result_calls) == 3  # exactly _RESULT_SUBMIT_MAX_ATTEMPTS, no more


@pytest.mark.asyncio
async def test_one_job_failure_does_not_abort_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []

    async def fake_completion(credential: Any) -> None:
        call_order.append(credential.id)
        if credential.id == "cm-1":
            raise ValueError("boom")

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)

    job_a = _chat_job(job_id="job-a", lease_token="lease-a")
    job_b = _chat_job(job_id="job-b", lease_token="lease-b")
    client = _FakeClient([job_a, job_b])

    claimed = await verification_tasks.run_verification_batch(client=client, limit=5)

    assert claimed == 2
    assert len(client.result_calls) == 2
    assert {call["job_id"] for call in client.result_calls} == {"job-a", "job-b"}


@pytest.mark.asyncio
async def test_no_jobs_claimed_is_a_no_op() -> None:
    client = _FakeClient([])

    claimed = await verification_tasks.run_verification_batch(client=client, limit=5)

    assert claimed == 0
    assert client.result_calls == []


def test_celery_task_takes_no_arguments() -> None:
    """plan.md/todo.md are explicit: the Celery task must not accept the claimed credential
    (or anything else) as an argument — claimed jobs must only ever exist in a local variable
    inside one task run, never on the broker."""

    from app.worker.celery_app import verify_pending_credentials

    underlying = getattr(verify_pending_credentials, "run", verify_pending_credentials)
    signature = inspect.signature(underlying)
    assert signature.parameters == {}


def test_celery_task_ignores_its_result() -> None:
    from app.worker.celery_app import verify_pending_credentials

    assert verify_pending_credentials.ignore_result is True


def test_run_verification_batch_signature_has_no_credential_parameter() -> None:
    """The plain async function callers actually invoke never accepts a credential either -
    only a client/limit, matching the Celery task's own no-argument contract."""

    signature = inspect.signature(verification_tasks.run_verification_batch)
    assert set(signature.parameters) == {"client", "limit"}
