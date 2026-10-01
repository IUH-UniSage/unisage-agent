"""Tests for `app.core.observability.alerting` — debounce/redaction behavior, and one test per
wiring point proving `alert_credential_failure` is actually called from the site that
knows a failure is PERMANENT/exhausted, not just that the function works in isolation.

No live Redis, no live Slack anywhere here: `send_slack_alert` is monkeypatched to a
recorder, and Redis is a hand-rolled fake (same spirit as `test_model_router.py`'s
`FakeAsyncRedis`/`DownRedis`) since this repo has no `fakeredis` dependency.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.core.observability.alerting as alerting
import app.core.registry.model_registry as model_registry
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import CredentialConfig, ModelRegistrySnapshot, parse_snapshot
from app.core.registry.model_router import ModelRouter
from app.core.security.ssrf_guard import SsrfBlockedError

# ── shared fixtures / test doubles ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


def _credential(
    *, id: str = "cred-1", api_key: str = "sk-canary-secret-value", **overrides: Any
) -> CredentialConfig:
    defaults: dict[str, Any] = dict(
        id=id,
        revision=1,
        source_type="CLOUD_API",
        provider="openai",
        model_name="gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        priority=1,
        max_rpm=None,
        api_key=api_key,
    )
    defaults.update(overrides)
    return CredentialConfig(**defaults)


class FakeDebounceRedis:
    """Minimal `SET NX EX` stand-in — a `set(..., nx=True)` on a key already present
    returns a falsy value (nothing acquired), exactly like a real Redis miss."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    async def set(self, name: str, value: Any, *, nx: bool = False, ex: int | None = None) -> Any:
        del value, ex
        if nx and name in self.keys:
            return None
        self.keys.add(name)
        return True

    async def aclose(self) -> None:
        pass


class DownRedis:
    """Every call raises — simulates Redis being completely unreachable."""

    async def set(self, name: str, value: Any, *, nx: bool = False, ex: int | None = None) -> Any:
        raise ConnectionError("simulated Redis outage")

    async def aclose(self) -> None:
        pass


def _set_snapshot(*, version: int, chat: tuple[CredentialConfig, ...]) -> ModelRegistrySnapshot:
    snapshot = ModelRegistrySnapshot(
        version=version,
        generated_at=parse_snapshot(
            {"version": version, "generatedAt": "2026-09-26T00:00:00Z", "purposes": {}}
        ).generated_at,
        purposes={"CHAT": chat},
        embedding_index_identity=None,
    )
    model_registry._current_snapshot = snapshot
    return snapshot


# ── core alerting.py behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_many_concurrent_permanent_failures_send_exactly_one_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    async def fake_send(message: str) -> None:
        sent.append(message)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)
    redis_client = FakeDebounceRedis()
    credential = _credential()

    await asyncio.gather(
        *[
            alerting.alert_credential_failure(
                credential, "PERMANENT", "boom", redis_client=redis_client
            )
            for _ in range(20)
        ]
    )

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_two_different_credentials_get_two_separate_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    async def fake_send(message: str) -> None:
        sent.append(message)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)
    redis_client = FakeDebounceRedis()

    await alerting.alert_credential_failure(
        _credential(id="cred-a"), "PERMANENT", "boom a", redis_client=redis_client
    )
    await alerting.alert_credential_failure(
        _credential(id="cred-b"), "PERMANENT", "boom b", redis_client=redis_client
    )

    assert len(sent) == 2


@pytest.mark.asyncio
async def test_same_credential_two_incident_types_both_alert_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    async def fake_send(message: str) -> None:
        sent.append(message)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)
    redis_client = FakeDebounceRedis()
    credential = _credential(id="cred-x")

    await alerting.alert_credential_failure(
        credential, "PERMANENT", "circuit break", redis_client=redis_client
    )
    await alerting.alert_credential_failure(
        credential, "VERIFICATION_FAILED_PERMANENT", "verify failed", redis_client=redis_client
    )
    # Same incident type again within the window - debounced.
    await alerting.alert_credential_failure(
        credential, "PERMANENT", "circuit break again", redis_client=redis_client
    )

    assert len(sent) == 2


@pytest.mark.asyncio
async def test_message_never_contains_the_credentials_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def fake_send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)
    secret = "sk-canary-secret-value-1234567890"
    credential = _credential(api_key=secret)
    redis_client = FakeDebounceRedis()

    await alerting.alert_credential_failure(
        credential,
        "PERMANENT",
        f"authentication failed using key {secret}",
        redis_client=redis_client,
    )

    assert len(sent) == 1
    # The whole Block Kit payload, not just the plain-text `text` fallback -
    # the secret must never survive anywhere in it (blocks/attachments included).
    serialized = json.dumps(sent[0])
    assert secret not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.asyncio
async def test_redis_down_alerts_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    """This codebase's established pattern (circuit breaker, hot-reload, verification
    lock) is "degrade, don't crash the caller" when Redis is unreachable. For alerting
    specifically this repo chose "alert anyway" over "skip the alert": losing the
    debounce guard risks a duplicate Slack message, while skipping the alert risks
    total silence during exactly the kind of outage a human most needs to hear about -
    a duplicate is the smaller failure of the two."""

    sent: list[str] = []

    async def fake_send(message: str) -> None:
        sent.append(message)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)

    await alerting.alert_credential_failure(
        _credential(), "PERMANENT", "boom", redis_client=DownRedis()
    )

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_no_available_credential_has_no_single_credential_but_still_debounces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    async def fake_send(message: str) -> None:
        sent.append(message)

    monkeypatch.setattr(alerting, "send_slack_alert", fake_send)
    redis_client = FakeDebounceRedis()

    await alerting.alert_credential_failure(
        None, "NO_AVAILABLE_CREDENTIAL", "exhausted", purpose="CHAT", redis_client=redis_client
    )
    await alerting.alert_credential_failure(
        None,
        "NO_AVAILABLE_CREDENTIAL",
        "exhausted again",
        purpose="CHAT",
        redis_client=redis_client,
    )
    await alerting.alert_credential_failure(
        None,
        "NO_AVAILABLE_CREDENTIAL",
        "exhausted",
        purpose="EXTRACTION",
        redis_client=redis_client,
    )

    assert len(sent) == 2  # CHAT collapses to one, EXTRACTION is a separate scope


# ── wiring: app.core.registry.model_router ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_router_permanent_failure_calls_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.registry import model_router as model_router_module

    alert_mock = AsyncMock()
    monkeypatch.setattr(model_router_module, "alert_credential_failure", alert_mock)

    class _FakeBackend:
        async def report_health(self, **kwargs: Any) -> dict[str, Any]:
            return {"applied": True}

    router = ModelRouter(redis_client=FakeDebounceRedis(), backend_client=_FakeBackend())
    credential = _credential(id="cred-perm")

    await router.record_failure(
        credential, SsrfBlockedError("blocked host"), snapshot_version=1, purpose="CHAT"
    )

    alert_mock.assert_awaited_once()
    args, kwargs = alert_mock.call_args
    assert args[0] is credential
    assert args[1] == "PERMANENT"
    assert kwargs["purpose"] == "CHAT"


@pytest.mark.asyncio
async def test_model_router_transient_failure_also_calls_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Product decision (spec.md "Slack alerting"): a TRANSIENT failure alerts
    too, not just PERMANENT - a human should hear about a provider blip the
    first time it happens, not only once the circuit breaker gives up on the
    credential entirely. The 15-minute debounce (tested elsewhere in this
    file) is what keeps a sustained blip from flooding the channel, not
    withholding the alert in the first place."""

    from app.core.registry import model_router as model_router_module

    alert_mock = AsyncMock()
    monkeypatch.setattr(model_router_module, "alert_credential_failure", alert_mock)

    class _FakeBackend:
        async def report_health(self, **kwargs: Any) -> dict[str, Any]:
            return {"applied": True}

    router = ModelRouter(redis_client=FakeDebounceRedis(), backend_client=_FakeBackend())
    credential = _credential(id="cred-transient")

    # An unrecognized exception fails open to TRANSIENT (llm_error_classifier's
    # documented behavior) - a routine, still-retrying failure with cooldown, not an
    # exhaustion/exclusion.
    await router.record_failure(
        credential, ConnectionError("connect timed out"), snapshot_version=1, purpose="CHAT"
    )

    alert_mock.assert_awaited_once()
    args, kwargs = alert_mock.await_args
    assert args[0] is credential
    assert args[1] == "TRANSIENT"
    assert kwargs["purpose"] == "CHAT"


@pytest.mark.asyncio
async def test_model_router_exhaustion_calls_alert_with_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.registry import model_router as model_router_module

    alert_mock = AsyncMock()
    monkeypatch.setattr(model_router_module, "alert_credential_failure", alert_mock)

    _set_snapshot(version=1, chat=())
    router = ModelRouter(redis_client=FakeDebounceRedis())

    with pytest.raises(NoAvailableCredentialError):
        await router.get_next_credential("CHAT")

    alert_mock.assert_awaited_once()
    args, kwargs = alert_mock.call_args
    assert args[0] is None
    assert args[1] == "NO_AVAILABLE_CREDENTIAL"
    assert kwargs["purpose"] == "CHAT"


# ── wiring: app.worker.celery_app (embed_chunks) ────────────────────────────


@patch("app.worker.celery_app.publish_ingestion_event")
@patch("app.worker.celery_app.BackendJavaClient")
@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.build_embedder")
def test_embed_chunks_calls_alert_on_embedding_provider_error(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    mock_backend_client_cls: MagicMock,
    mock_publish: MagicMock,
) -> None:
    from app.core.errors.llm_error_classifier import EmbeddingProviderError
    from app.worker.celery_app import celery_app, embed_chunks
    from app.worker.embedding_job_errors import IngestionJobFailedError

    celery_app.conf.update(
        broker_url="memory://",
        result_backend="cache+memory://",
        task_always_eager=True,
        task_eager_propagates=True,
        task_store_eager_result=True,
    )

    credential = _credential(id="embed-cred", api_key="sk-embed-secret")
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(
        side_effect=EmbeddingProviderError("provider auth failed", credential=credential)
    )
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()
    mock_backend_client_cls.return_value.report_health = AsyncMock(return_value={"applied": True})

    alert_mock = AsyncMock()

    with (
        patch("app.worker.celery_app.alert_credential_failure", alert_mock),
        patch(
            "app.worker.celery_app.get_current_snapshot",
            return_value=ModelRegistrySnapshot(
                version=1, generated_at=None, purposes={}, embedding_index_identity=None
            ),
        ),
        pytest.raises(IngestionJobFailedError),
    ):
        embed_chunks.apply(
            args=(
                "doc-1",
                "docs/handbook.pdf",
                [{"chunk_index": 0, "content": "chunk content", "region_type": "text"}],
                "CNTT",
                2,
            )
        )

    alert_mock.assert_awaited_once()
    args, kwargs = alert_mock.call_args
    assert args[0] is credential
    assert args[1] == "EMBEDDING_PROVIDER_FAILURE"
    assert kwargs["purpose"] == "EMBEDDING"


# ── wiring: app.worker.verification_tasks ───────────────────────────────────


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


class _FakeVerificationClient:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs
        self.result_calls: list[dict[str, Any]] = []

    async def claim_verifications(self, *, limit: int) -> list[dict[str, Any]]:
        return self.jobs

    async def post_verification_result(self, **kwargs: Any) -> dict[str, Any]:
        self.result_calls.append(kwargs)
        return {"applied": True, "duplicate": False}


@pytest.mark.asyncio
async def test_verification_permanent_result_calls_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.worker import verification_tasks

    async def fake_completion(credential: Any) -> None:
        del credential
        raise SsrfBlockedError("blocked host")  # classify_llm_error always -> PERMANENT

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)
    alert_mock = AsyncMock()
    monkeypatch.setattr(verification_tasks, "alert_credential_failure", alert_mock)

    client = _FakeVerificationClient([_chat_job()])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert client.result_calls[0]["result_type"] == "PERMANENT"
    alert_mock.assert_awaited_once()
    args, kwargs = alert_mock.call_args
    assert args[1] == "VERIFICATION_FAILED_PERMANENT"
    assert kwargs["purpose"] == "CHAT"


@pytest.mark.asyncio
async def test_verification_transient_result_never_calls_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents the known gap: the claim response carries `attempt` but not
    `maxAttempts`, so Python cannot tell a still-retrying TRANSIENT result apart from
    one that is about to exhaust Java's attempt budget - it never guesses, so a
    TRANSIENT result is never alerted here, even one that happens to be the last
    attempt."""

    from app.worker import verification_tasks

    async def fake_completion(credential: Any) -> None:
        del credential
        raise ConnectionError("connect timed out")  # unrecognized -> fails open TRANSIENT

    monkeypatch.setattr(verification_tasks, "_run_minimal_completion", fake_completion)
    alert_mock = AsyncMock()
    monkeypatch.setattr(verification_tasks, "alert_credential_failure", alert_mock)

    client = _FakeVerificationClient([_chat_job()])
    await verification_tasks.run_verification_batch(client=client, limit=5)

    assert client.result_calls[0]["result_type"] == "TRANSIENT"
    alert_mock.assert_not_awaited()
