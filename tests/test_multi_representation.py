import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import app.core.registry.model_registry as model_registry
import app.core.registry.model_router as model_router_module
import app.core.usage.usage_outbox as usage_outbox_module
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import CredentialConfig, ModelRegistrySnapshot, parse_snapshot
from app.core.registry.model_router import ModelRouter
from app.rag.enrichment.multi_representation import (
    MalformedExtractionResponseError,
    MultiRepresentationEnricher,
)
from app.schemas.ingestion import Chunk, RegionType


@pytest.fixture
def captured_outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def _fake_enqueue(payload: dict[str, Any], *, redis_client: Any = None) -> None:
        del redis_client
        captured.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(usage_outbox_module, "enqueue_usage_payload", _fake_enqueue)
    return captured


def _function_model(text: str) -> FunctionModel:
    """A `FunctionModel` whose `run()` returns `text` as one plain-text response - same
    double `tests/llm_mocks.py`'s `make_sync_llm_model` builds, inlined here so a
    failing-call variant (below) can live next to it."""

    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=function)


def _failing_model(exc: Exception) -> FunctionModel:
    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        raise exc

    return FunctionModel(function=function)


def _success_payload(*, questions: int = 3) -> str:
    return json.dumps({"summary": "ok", "questions": [f"Q{i}?" for i in range(1, questions + 1)]})


def _chunk() -> Chunk:
    return Chunk(chunk_index=0, content="Some chunk content.", region_type=RegionType.TEXT)


def test_enrich_returns_summary_and_configured_question_count() -> None:
    model = _function_model(
        json.dumps({"summary": "A short summary.", "questions": ["Q1?", "Q2?", "Q3?"]})
    )
    enricher = MultiRepresentationEnricher(model=model, question_count=3)

    enriched = enricher.enrich(_chunk())

    assert enriched.summary == "A short summary."
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]
    assert enriched.chunk == _chunk()


def test_enrich_falls_back_to_empty_on_malformed_json() -> None:
    enricher = MultiRepresentationEnricher(
        model=_function_model("not valid json"), question_count=3
    )

    enriched = enricher.enrich(_chunk())

    assert enriched.summary == ""
    assert enriched.questions == []


def test_enrich_falls_back_to_empty_on_short_question_list() -> None:
    payload = json.dumps({"summary": "A short summary.", "questions": ["Only one?"]})
    enricher = MultiRepresentationEnricher(model=_function_model(payload), question_count=3)

    enriched = enricher.enrich(_chunk())

    assert enriched.summary == ""
    assert enriched.questions == []


def test_enrich_accepts_json_wrapped_in_a_code_fence() -> None:
    """Nothing forces strict JSON mode the way OpenAI's `response_format=json_object` used to
    (no provider-agnostic equivalent) - `_extract_json_object` must tolerate a code fence the
    same way `message_classification._load_json_object` already does for CHAT."""

    payload = "```json\n" + _success_payload() + "\n```"
    enricher = MultiRepresentationEnricher(model=_function_model(payload), question_count=3)

    enriched = enricher.enrich(_chunk())

    assert enriched.summary == "ok"
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]


# ── model_router-driven failover ─────────────────────────
#
# These tests exercise the registry-resolved path (no `model` injected), so
# `enrich()`/`enrich_tracked()` goes through `model_router` AND
# `app.graph.streaming.run_agent_text_with_failover`. Same spirit as
# `tests/api/test_chat_stream_errors.py`: a hand-rolled fake Redis + fake
# `backend-java` client installed as the process-wide default router, `build_model`
# monkeypatched (in BOTH the module that picks the first credential and
# `app.graph.streaming`, which builds every failover replacement) so a credential
# resolves to a `FunctionModel` double instead of a real provider SDK object - no live
# Redis/backend-java/provider network access anywhere here.


class _FakeRedis:
    """No-expiry stand-in for `redis.asyncio.Redis` - a key marked once stays
    marked for the test's lifetime (these tests don't need cooldown-TTL expiry)."""

    def __init__(self) -> None:
        self._blocked: set[str] = set()

    async def set(self, name: str, _value: Any, *, ex: int | None = None) -> Any:
        del ex
        self._blocked.add(name)
        return True

    async def exists(self, name: str) -> int:
        return 1 if name in self._blocked else 0

    async def aclose(self) -> Any:
        return None


class _FakeBackendClient:
    """Structural stand-in for `BackendJavaClient` - only `report_health()`, the one
    method `ModelRouter.record_failure()` calls."""

    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    async def report_health(self, **kwargs: Any) -> None:
        self.reports.append(kwargs)


@pytest.fixture
def fake_router(monkeypatch: pytest.MonkeyPatch) -> tuple[ModelRouter, _FakeBackendClient]:
    """Installs a `ModelRouter` backed by `_FakeRedis`/`_FakeBackendClient` as the
    process-wide default (`get_default_router()`), and resets the module-level registry
    snapshot - both revert automatically via `monkeypatch`."""

    backend = _FakeBackendClient()
    router = ModelRouter(redis_client=_FakeRedis(), backend_client=backend)
    monkeypatch.setattr(model_router_module, "_default_router", router)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)
    return router, backend


def _credential(credential_id: str, priority: int, *, provider: str = "openai") -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider=provider,
        model_name="gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        priority=priority,
        max_rpm=None,
        api_key="sk-test",
    )


def _set_extraction_snapshot(*, version: int, extraction: tuple[CredentialConfig, ...]) -> None:
    snapshot = ModelRegistrySnapshot(
        version=version,
        generated_at=parse_snapshot(
            {"version": version, "generatedAt": "2026-09-26T00:00:00Z", "purposes": {}}
        ).generated_at,
        purposes={"EXTRACTION": extraction},
        embedding_index_identity=None,
    )
    model_registry._current_snapshot = snapshot


def _patch_build_model(
    monkeypatch: pytest.MonkeyPatch, models_by_credential_id: dict[str, FunctionModel]
) -> None:
    """`enrich_tracked` builds the FIRST credential's model itself
    (`app.rag.enrichment.multi_representation.build_model`); every failover replacement is
    built by `run_agent_text_with_failover` instead (`app.graph.streaming.build_model`) - both
    names must resolve to the same fake or a failover attempt would try a real provider SDK."""

    def fake_build_model(credential: CredentialConfig) -> FunctionModel:
        return models_by_credential_id[credential.id]

    monkeypatch.setattr("app.rag.enrichment.multi_representation.build_model", fake_build_model)
    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model)


def test_provider_failure_falls_back_to_next_extraction_credential(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _router, backend = fake_router
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback = _credential("cred-fallback", priority=2)
    _set_extraction_snapshot(version=1, extraction=(cred_primary, cred_fallback))

    _patch_build_model(
        monkeypatch,
        {
            "cred-primary": _failing_model(RuntimeError("primary down")),
            "cred-fallback": _function_model(_success_payload()),
        },
    )

    enricher = MultiRepresentationEnricher(question_count=3)
    enriched = enricher.enrich(_chunk())

    assert enriched.summary == "ok"
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]
    # The primary credential's failure was reported to Java before falling back.
    assert len(backend.reports) == 1
    assert backend.reports[0]["credential_id"] == "cred-primary"


def test_malformed_response_on_fallback_credential_reported_permanent(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed response from the PRIMARY credential just falls back to empty (see
    `test_enrich_falls_back_to_empty_on_malformed_json` above) - the same malformed shape
    from a FALLBACK credential is different: it's reported to `model_router.record_failure()`
    as `MalformedExtractionResponseError` instead of being handed to the ingest pipeline."""

    _router, _backend = fake_router
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback_bad = _credential("cred-fallback-bad", priority=2)
    cred_fallback_good = _credential("cred-fallback-good", priority=3)
    _set_extraction_snapshot(
        version=1, extraction=(cred_primary, cred_fallback_bad, cred_fallback_good)
    )

    _patch_build_model(
        monkeypatch,
        {
            "cred-primary": _failing_model(RuntimeError("primary down")),
            "cred-fallback-bad": _function_model("not valid json"),
            "cred-fallback-good": _function_model(_success_payload()),
        },
    )

    real_record_failure = model_router_module.record_failure
    recorded_calls: list[tuple[CredentialConfig, Exception, int]] = []

    async def spy_record_failure(
        credential: CredentialConfig,
        exc: Exception,
        *,
        snapshot_version: int,
        purpose: str | None = None,
    ) -> None:
        recorded_calls.append((credential, exc, snapshot_version))
        await real_record_failure(
            credential, exc, snapshot_version=snapshot_version, purpose=purpose
        )

    monkeypatch.setattr(model_router_module, "record_failure", spy_record_failure)

    enricher = MultiRepresentationEnricher(question_count=3)
    enriched = enricher.enrich(_chunk())

    assert enriched.summary == "ok"
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]

    malformed_reports = [
        (cred, exc) for cred, exc, _version in recorded_calls if cred.id == "cred-fallback-bad"
    ]
    assert len(malformed_reports) == 1
    reported_credential, reported_exc = malformed_reports[0]
    assert isinstance(reported_exc, MalformedExtractionResponseError)
    assert reported_credential.id == "cred-fallback-bad"


def test_total_credential_exhaustion_propagates_instead_of_returning_empty(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrast with the malformed-primary-response case
    (`test_enrich_falls_back_to_empty_on_malformed_json`): a single bad-looking response is
    swallowed into an empty result so it doesn't kill the batch, but every EXTRACTION
    credential being unusable is a different kind of failure (a real outage, not "this
    chunk's response happened to look malformed") - it must propagate so the caller
    (`embed_chunks_task`) records this chunk as FAILED instead of silently degrading every
    remaining chunk in the batch to placeholder embeddings."""

    _router, backend = fake_router
    cred_only = _credential("cred-only", priority=1)
    _set_extraction_snapshot(version=1, extraction=(cred_only,))

    _patch_build_model(monkeypatch, {"cred-only": _failing_model(RuntimeError("provider outage"))})

    # RuntimeError classifies as TRANSIENT (unrecognized -> fail open), so the exclusion
    # is a cooldown rather than an immediate PERMANENT exclusion - either way, with only
    # one credential in the snapshot, the very next `get_next_credential()` call has
    # nothing left to return.
    enricher = MultiRepresentationEnricher(question_count=3)

    with pytest.raises(NoAvailableCredentialError):
        enricher.enrich(_chunk())

    assert len(backend.reports) == 1
    assert backend.reports[0]["credential_id"] == "cred-only"


def test_snapshot_version_reported_is_the_one_captured_at_failure_not_a_later_drift(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same invariant `tests/core/test_model_router.py`'s
    `test_health_report_uses_snapshot_version_captured_at_failure_not_later_refresh` proves for
    `ModelRouter` itself: the health report must carry the snapshot version that was current
    when the failing credential was selected, not whatever the snapshot has drifted to by the
    time the report is actually sent - even though the drift happens *during* the provider call,
    before `record_failure()` is even invoked."""

    _router, backend = fake_router
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback = _credential("cred-fallback", priority=2)
    _set_extraction_snapshot(version=1, extraction=(cred_primary, cred_fallback))

    def _fail_and_drift_snapshot(
        _messages: list[ModelMessage], _agent_info: AgentInfo
    ) -> ModelResponse:
        # Simulates a hot-reload landing while the primary credential's call is in
        # flight - the snapshot version bumps before `record_failure()` is called.
        _set_extraction_snapshot(version=99, extraction=(cred_primary, cred_fallback))
        raise RuntimeError("primary down")

    _patch_build_model(
        monkeypatch,
        {
            "cred-primary": FunctionModel(function=_fail_and_drift_snapshot),
            "cred-fallback": _function_model(_success_payload()),
        },
    )

    enricher = MultiRepresentationEnricher(question_count=3)
    enriched = enricher.enrich(_chunk())

    assert enriched.summary == "ok"
    assert len(backend.reports) == 1
    assert backend.reports[0]["credential_id"] == "cred-primary"
    assert backend.reports[0]["snapshot_version"] == 1


def test_enrich_works_regardless_of_which_provider_the_credential_is(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of routing through `app.core.llm.provider_models.build_model()`
    (the same factory `get_graph_models()`/CHAT uses) instead of always speaking the OpenAI
    wire format directly: an EXTRACTION credential with `provider="google"` must work exactly
    like one with `provider="openai"` - no special-cased base URL, no 404."""

    cred = _credential("cred-google", priority=1, provider="google")
    _set_extraction_snapshot(version=1, extraction=(cred,))
    _patch_build_model(monkeypatch, {"cred-google": _function_model(_success_payload())})

    enriched = MultiRepresentationEnricher(question_count=3).enrich(_chunk())

    assert enriched.summary == "ok"
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]


# --- Usage recording: one enrich() call = one purpose=EXTRACTION request ---


def test_enrich_success_records_one_line_with_purpose_extraction(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    _router, _backend = fake_router
    cred = _credential("cred-1", priority=1)
    _set_extraction_snapshot(version=1, extraction=(cred,))
    _patch_build_model(monkeypatch, {"cred-1": _function_model(_success_payload())})

    enricher = MultiRepresentationEnricher(question_count=3)
    enricher.enrich(_chunk())

    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    assert payload["purpose"] == "EXTRACTION"
    assert payload["conversationId"] is None
    assert payload["userMessageId"] is None
    assert payload["status"] == "SUCCESS"
    assert len(payload["lines"]) == 1
    line = payload["lines"][0]
    assert line["nodeName"] == "multi_representation_enrich"
    assert line["chatModelId"] == "cred-1"


def test_enrich_failover_records_error_then_success_line_different_credentials(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    _router, _backend = fake_router
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback = _credential("cred-fallback", priority=2)
    _set_extraction_snapshot(version=1, extraction=(cred_primary, cred_fallback))

    _patch_build_model(
        monkeypatch,
        {
            "cred-primary": _failing_model(RuntimeError("primary down")),
            "cred-fallback": _function_model(_success_payload()),
        },
    )

    enricher = MultiRepresentationEnricher(question_count=3)
    enricher.enrich(_chunk())

    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    assert payload["purpose"] == "EXTRACTION"
    assert payload["conversationId"] is None
    assert payload["userMessageId"] is None
    # A failover recovered - the graph-level result succeeded, so the parent downgrades
    # from what would be SUCCESS to PARTIAL because one line failed along the way.
    assert payload["status"] == "PARTIAL"

    lines = payload["lines"]
    assert len(lines) == 2
    assert lines[0]["status"] == "ERROR"
    assert lines[0]["attempt"] == 0
    assert lines[0]["chatModelId"] == "cred-primary"
    assert lines[0]["errorCode"] == "RuntimeError"

    assert lines[1]["status"] == "SUCCESS"
    assert lines[1]["attempt"] == 1
    assert lines[1]["chatModelId"] == "cred-fallback"


def test_enrich_with_injected_model_records_nothing(
    captured_outbox: list[dict[str, Any]],
) -> None:
    """Test-injected model (no registry credential resolved) skips the registry/router
    entirely - matches OpenAIEmbedder's identical precedent."""

    enricher = MultiRepresentationEnricher(
        model=_function_model(_success_payload()), question_count=3
    )

    enricher.enrich(_chunk())

    assert captured_outbox == []


@pytest.mark.asyncio
async def test_enrich_called_from_a_running_event_loop_still_records_usage(
    fake_router: tuple[ModelRouter, _FakeBackendClient],
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    cred = _credential("cred-1", priority=1)
    _set_extraction_snapshot(version=1, extraction=(cred,))
    _patch_build_model(monkeypatch, {"cred-1": _function_model(_success_payload())})

    result = MultiRepresentationEnricher(question_count=3).enrich(_chunk())

    assert result.summary == "ok"
    assert len(captured_outbox) == 1
    assert captured_outbox[0]["status"] == "SUCCESS"
