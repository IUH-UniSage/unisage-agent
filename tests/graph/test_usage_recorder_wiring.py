"""Usage-recording verification: line count/attempt/status through a
real graph run (classification + query transformation + generation, with and
without failover), and that the outbox still gets the payload when the SSE
consumer never reads the queue (client-disconnect equivalent - `run_and_persist`
runs independently of its queue's reader by design, see streaming_session.py's
own module docstring)."""

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

import app.core.model_registry as model_registry
import app.core.model_router as model_router_module
import app.core.usage_outbox as usage_outbox_module
from app.core.config import settings
from app.core.model_registry import CredentialConfig, ModelRegistrySnapshot, parse_snapshot
from app.core.model_router import ModelRouter
from app.core.usage_recorder import UsageRecorder
from app.graph.queue_items import QueueItem
from app.graph.streaming_session import run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import (
    FakeRetrievalService,
    make_classification_llm_model,
    make_streaming_llm_model,
    make_streaming_llm_model_that_fails_after,
)


class _FakeRedis:
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
    async def report_health(self, **kwargs: Any) -> None:
        del kwargs


@pytest.fixture
def fake_router(monkeypatch: pytest.MonkeyPatch) -> ModelRouter:
    router = ModelRouter(redis_client=_FakeRedis(), backend_client=_FakeBackendClient())
    monkeypatch.setattr(model_router_module, "_default_router", router)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)
    return router


@pytest.fixture
def captured_outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the real Redis-backed enqueue with an in-memory capture."""

    captured: list[dict[str, Any]] = []

    async def _fake_enqueue(payload: dict[str, Any], *, redis_client: Any = None) -> None:
        del redis_client
        # Round-trips through JSON, same as the real wire shape.
        captured.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(usage_outbox_module, "enqueue_usage_payload", _fake_enqueue)
    # UsageRecorder.close() does `from app.core.usage_outbox import enqueue_usage_payload` as a
    # LOCAL import - patching the module attribute above is what that import sees at call time.
    return captured


def _credential(
    credential_id: str, *, provider: str = "openai", priority: int = 1
) -> CredentialConfig:
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


def _set_snapshot(*, version: int, chat: tuple[CredentialConfig, ...]) -> None:
    snapshot = ModelRegistrySnapshot(
        version=version,
        generated_at=parse_snapshot(
            {"version": version, "generatedAt": "2026-09-26T00:00:00Z", "purposes": {}}
        ).generated_at,
        purposes={"CHAT": chat},
        embedding_index_identity=None,
    )
    model_registry._current_snapshot = snapshot


def _graph_input() -> GraphInput:
    return GraphInput(
        conversation_id="conv-1",
        user_message="Điều kiện học bổng loại giỏi là gì?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )


def _java_client() -> BackendJavaClient:
    import httpx

    return BackendJavaClient(
        base_url="http://java.test",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"id": "msg-2"})),
    )


class _SessionCtx:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        return None


def _models(
    generation: FunctionModel,
    *,
    query_transformation: Callable[[str], FunctionModel],
    credential: CredentialConfig,
    snapshot_version: int,
    chunks: Sequence[RetrievedChunk],
) -> GraphModels:
    return GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=query_transformation("hyde doc"),
        generation=generation,
        retrieval=FakeRetrievalService(list(chunks)),
        generation_credential=credential,
        snapshot_version=snapshot_version,
    )


_DUMMY_CHUNK = RetrievedChunk(
    chunk_id="c1", content="dummy retrieved content", source="s", score=0.9
)


@pytest.mark.asyncio
async def test_successful_advisory_turn_records_exactly_3_lines(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred = _credential("cred-1")
    _set_snapshot(version=1, chat=(cred,))

    models = _models(
        make_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        query_transformation=mock_sync_llm_model,
        credential=cred,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    await run_and_persist(
        java_client=_java_client(),
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        usage_recorder=UsageRecorder(request_id="req-1", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    assert payload["requestId"] == "req-1"
    assert payload["status"] == "SUCCESS"
    lines = payload["lines"]
    assert len(lines) == 3
    assert [line["nodeName"] for line in lines] == [
        "MessageClassificationNode",
        "QueryTransformationNode",
        "GenerationSynthesisNode",
    ]
    assert all(line["status"] == "SUCCESS" for line in lines)
    assert all(line["attempt"] == 0 for line in lines)
    assert all(line["chatModelId"] == "cred-1" for line in lines)


@pytest.mark.asyncio
async def test_generation_failover_records_4_lines_with_failed_and_succeeded_attempts(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred_primary = _credential("cred-primary", provider="openai", priority=1)
    cred_fallback = _credential("cred-fallback", provider="anthropic", priority=2)
    _set_snapshot(version=1, chat=(cred_primary, cred_fallback))

    failing_model = make_streaming_llm_model_that_fails_after([], RuntimeError("primary down"))
    fallback_model = make_streaming_llm_model(["x", "y", "z"])

    def fake_build_model(credential: CredentialConfig) -> FunctionModel:
        assert credential.id == cred_fallback.id
        return fallback_model

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model)

    models = _models(
        failing_model,
        query_transformation=mock_sync_llm_model,
        credential=cred_primary,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    await run_and_persist(
        java_client=_java_client(),
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        usage_recorder=UsageRecorder(request_id="req-2", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    # The graph itself still completed successfully (transparent failover, no
    # chunk streamed before the switch) - but one line failed, so the parent
    # status must downgrade from SUCCESS to PARTIAL, not stay SUCCESS.
    assert payload["status"] == "PARTIAL"
    lines = payload["lines"]
    assert len(lines) == 4
    assert [line["nodeName"] for line in lines] == [
        "MessageClassificationNode",
        "QueryTransformationNode",
        "GenerationSynthesisNode",
        "GenerationSynthesisNode",
    ]

    failed_line, succeeded_line = lines[2], lines[3]
    assert failed_line["status"] == "ERROR"
    assert failed_line["attempt"] == 0
    assert failed_line["chatModelId"] == "cred-primary"
    assert failed_line["provider"] == "openai"

    assert succeeded_line["status"] == "SUCCESS"
    assert succeeded_line["attempt"] == 1
    assert succeeded_line["chatModelId"] == "cred-fallback"
    assert succeeded_line["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_payload_still_enqueued_when_the_sse_consumer_never_reads_the_queue(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    captured_outbox: list[dict[str, Any]],
) -> None:
    """`run_and_persist` is scheduled as an independent `asyncio.create_task()`
    and keeps running to completion even if nothing ever drains `queue` (the
    real-world equivalent of a client disconnect cancelling the SSE generator,
    per this module's own docstring) - the outbox enqueue must still happen."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred = _credential("cred-1")
    _set_snapshot(version=1, chat=(cred,))

    models = _models(
        make_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        query_transformation=mock_sync_llm_model,
        credential=cred,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    # An unbounded queue nobody ever reads from - simulates the SSE generator
    # having been cancelled already.
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    await run_and_persist(
        java_client=_java_client(),
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        usage_recorder=UsageRecorder(request_id="req-3", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    assert len(captured_outbox) == 1
    assert captured_outbox[0]["requestId"] == "req-3"
