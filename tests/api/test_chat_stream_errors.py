"""Tests for the SSE error contract (plan.md "SSE error contract", todo.md
Task 11): `event: error` immediately before `event: done`, the pre-first-
chunk failover boundary in `stream_agent_text()`, and the no-mixed-content
invariant.

Drives `run_and_persist` + `_sse_token_generator` directly (same level as
`tests/graph/test_streaming_session.py`) rather than the full HTTP
endpoint, so the queue's typed items and the exact SSE bytes can both be
asserted without an extra ASGI layer in the way. No live Redis/backend-java
anywhere here: `app.core.model_router`'s process-wide default router is
swapped for one backed by hand-rolled fakes (same spirit as
`tests/core/test_model_router.py`), and `app.graph.streaming.build_model` is
monkeypatched so a "fallback credential" resolves to a `FunctionModel`
double instead of a real provider SDK object.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

import app.core.model_registry as model_registry
import app.core.model_router as model_router_module
from app.api.v1.chat import _sse_token_generator
from app.core.config import settings
from app.core.model_registry import CredentialConfig, ModelRegistrySnapshot, parse_snapshot
from app.core.model_router import ModelRouter
from app.graph.queue_items import DoneItem, ErrorItem, QueueItem, TokenItem
from app.graph.stream_error_codes import LLM_STREAM_INTERRUPTED, LLM_UNAVAILABLE
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
    """No-expiry stand-in for `redis.asyncio.Redis` - only `set`/`exists`/
    `aclose`, and a key marked once stays marked for the test's lifetime
    (no cooldown-TTL behavior needed here)."""

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
    """Structural stand-in for `BackendJavaClient` - only `report_health()`,
    the one method `ModelRouter.record_failure()` calls."""

    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    async def report_health(self, **kwargs: Any) -> None:
        self.reports.append(kwargs)


@pytest.fixture
def fake_router(monkeypatch: pytest.MonkeyPatch) -> ModelRouter:
    """Installs a `ModelRouter` backed by `_FakeRedis`/`_FakeBackendClient` as
    the process-wide default (`get_default_router()`) so `stream_agent_text()`
    never touches a real Redis/backend-java, and resets the module-level
    registry snapshot too - both revert automatically via `monkeypatch`."""

    router = ModelRouter(redis_client=_FakeRedis(), backend_client=_FakeBackendClient())
    monkeypatch.setattr(model_router_module, "_default_router", router)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)
    return router


def _credential(credential_id: str, priority: int) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider="openai",
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


def _java_client() -> tuple[BackendJavaClient, dict[str, Any]]:
    patched: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            patched["body"] = json.loads(request.read())
        return httpx.Response(200, json={"id": "msg-2", "status": "COMPLETED"})

    client = BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))
    return client, patched


class _SessionCtx:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        return None


async def _drain(queue: "asyncio.Queue[QueueItem]") -> list[QueueItem]:
    items: list[QueueItem] = []
    while True:
        item = await queue.get()
        items.append(item)
        if isinstance(item, DoneItem):
            return items


def _models_with_generation(
    generation: FunctionModel,
    *,
    query_transformation: Callable[[str], FunctionModel],
    credential: CredentialConfig | None,
    snapshot_version: int | None,
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
async def test_error_after_first_chunk_emits_error_then_done_no_tokens_after(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred = _credential("cred-1", priority=1)
    _set_snapshot(version=1, chat=(cred,))

    generation = make_streaming_llm_model_that_fails_after(["a", "b"], RuntimeError("boom"))
    models = _models_with_generation(
        generation,
        query_transformation=mock_sync_llm_model,
        credential=cred,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    java_client, patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    items = await _drain(queue)
    tokens = [item.text for item in items if isinstance(item, TokenItem)]
    # `pydantic_ai`'s `stream_text(delta=True)` debounces/groups deltas
    # (0.1s default) - "a"/"b" may arrive as one or two queue items, but the
    # concatenated text must be exactly what the model produced before it
    # failed, and no token item may follow the error.
    assert tokens
    assert "".join(tokens) == "ab"

    # Error must be the second-to-last item, immediately before Done - never
    # a token after it.
    assert isinstance(items[-1], DoneItem)
    assert isinstance(items[-2], ErrorItem)
    assert items[-2].code == LLM_STREAM_INTERRUPTED
    assert items[-2].retryable is True
    assert patched["body"]["status"] == "ERROR"

    # Render through the real SSE generator and check the exact wire shape.
    async def _replay() -> "asyncio.Queue[QueueItem]":
        replay: asyncio.Queue[QueueItem] = asyncio.Queue()
        for item in items:
            await replay.put(item)
        return replay

    replay_queue = await _replay()
    body = "".join([chunk async for chunk in _sse_token_generator(replay_queue)])
    assert body.count("event: token") == len(tokens)
    assert body.count("event: error") == 1
    assert body.count("event: done") == 1
    assert body.rindex("event: error") < body.rindex("event: done")
    assert body.rindex("event: token") < body.rindex("event: error")
    error_payload = json.loads(body.split("event: error\ndata: ")[1].split("\n\n")[0])
    assert error_payload == {
        "code": "LLM_STREAM_INTERRUPTED",
        "message": error_payload["message"],
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_error_before_first_chunk_falls_back_transparently(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback = _credential("cred-fallback", priority=2)
    _set_snapshot(version=1, chat=(cred_primary, cred_fallback))

    failing_model = make_streaming_llm_model_that_fails_after([], RuntimeError("primary down"))
    fallback_model = make_streaming_llm_model(["x", "y", "z"])

    def fake_build_model(credential: CredentialConfig) -> FunctionModel:
        assert credential.id == cred_fallback.id
        return fallback_model

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model)

    models = _models_with_generation(
        failing_model,
        query_transformation=mock_sync_llm_model,
        credential=cred_primary,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    java_client, patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    items = await _drain(queue)
    assert not any(isinstance(item, ErrorItem) for item in items)
    tokens = [item.text for item in items if isinstance(item, TokenItem)]
    assert "".join(tokens) == "xyz"
    assert patched["body"]["status"] == "COMPLETED"
    assert patched["body"]["content"] == "xyz"


@pytest.mark.asyncio
async def test_leading_empty_chunk_does_not_block_failover(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thinking-model provider (Gemini 3) can emit an empty/marker delta
    before any real text - `stream_agent_text()` used to treat ANY yielded
    chunk (including "") as "already streamed", permanently blocking
    failover for a request the user never actually saw output for. This is
    the same scenario as `test_error_before_first_chunk_falls_back_transparently`
    except the failing model yields one empty token first."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred_primary = _credential("cred-primary", priority=1)
    cred_fallback = _credential("cred-fallback", priority=2)
    _set_snapshot(version=1, chat=(cred_primary, cred_fallback))

    failing_model = make_streaming_llm_model_that_fails_after([""], RuntimeError("primary down"))
    fallback_model = make_streaming_llm_model(["x", "y", "z"])

    def fake_build_model(credential: CredentialConfig) -> FunctionModel:
        assert credential.id == cred_fallback.id
        return fallback_model

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model)

    models = _models_with_generation(
        failing_model,
        query_transformation=mock_sync_llm_model,
        credential=cred_primary,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    java_client, patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    items = await _drain(queue)
    assert not any(isinstance(item, ErrorItem) for item in items)
    tokens = [item.text for item in items if isinstance(item, TokenItem)]
    assert "".join(tokens) == "xyz"
    assert patched["body"]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_error_after_first_chunk_still_marks_the_credential_failed(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the point of no return, this call can't retry (see
    `test_error_after_first_chunk_emits_error_then_done_no_tokens_after`),
    but the credential must still be marked/reported so the *next* request
    picks a different one instead of hitting the same failing credential
    again - the exact "lượt sau vẫn chọn gemini 3.8" gap this closes."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred = _credential("cred-1", priority=1)
    _set_snapshot(version=1, chat=(cred,))

    class _FakeBackendClient:
        def __init__(self) -> None:
            self.reports: list[dict[str, Any]] = []

        async def report_health(self, **kwargs: Any) -> None:
            self.reports.append(kwargs)

    backend_client = _FakeBackendClient()
    router = ModelRouter(redis_client=_FakeRedis(), backend_client=backend_client)
    monkeypatch.setattr(model_router_module, "_default_router", router)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)

    generation = make_streaming_llm_model_that_fails_after(["a", "b"], RuntimeError("boom"))
    models = _models_with_generation(
        generation,
        query_transformation=mock_sync_llm_model,
        credential=cred,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    java_client, _patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    await _drain(queue)

    assert len(backend_client.reports) == 1
    assert backend_client.reports[0]["credential_id"] == cred.id


@pytest.mark.asyncio
async def test_error_before_first_chunk_credential_exhausted_is_llm_unavailable(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    cred = _credential("cred-only", priority=1)
    _set_snapshot(version=1, chat=(cred,))

    failing_model = make_streaming_llm_model_that_fails_after([], RuntimeError("down"))
    models = _models_with_generation(
        failing_model,
        query_transformation=mock_sync_llm_model,
        credential=cred,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )

    java_client, patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )

    items = await _drain(queue)
    assert not any(isinstance(item, TokenItem) for item in items)
    assert isinstance(items[-1], DoneItem)
    assert isinstance(items[-2], ErrorItem)
    assert items[-2].code == LLM_UNAVAILABLE
    assert items[-2].retryable is False
    assert patched["body"]["status"] == "ERROR"


@pytest.mark.asyncio
async def test_no_response_ever_mixes_content_from_two_models(
    fake_router: ModelRouter,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct proof of the invariant, both directions:

    - Failure BEFORE any chunk streamed: the final accumulated text is
      composed entirely of the fallback model's tokens - none of the
      (never-sent) primary model's would-be output leaks in, because it
      genuinely never streamed anything.
    - Failure AFTER a chunk streamed: the final accumulated text is composed
      entirely of the primary model's tokens - the fallback model is never
      even invoked (`fake_build_model` below would raise if it were), so
      nothing of "model B" can ever appear alongside "model A"'s output.
    """

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)

    # --- Direction 1: pre-first-chunk failure -> clean fallback. ---
    cred_a = _credential("model-a", priority=1)
    cred_b = _credential("model-b", priority=2)
    _set_snapshot(version=1, chat=(cred_a, cred_b))

    model_a_fails_immediately = make_streaming_llm_model_that_fails_after(
        [], RuntimeError("model A never got to speak")
    )
    model_b_success = make_streaming_llm_model(["MODEL-B-1", "MODEL-B-2"])

    def fake_build_model_fallback(credential: CredentialConfig) -> FunctionModel:
        assert credential.id == cred_b.id
        return model_b_success

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model_fallback)

    models = _models_with_generation(
        model_a_fails_immediately,
        query_transformation=mock_sync_llm_model,
        credential=cred_a,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )
    java_client, patched = _java_client()
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )
    items = await _drain(queue)
    tokens = [item.text for item in items if isinstance(item, TokenItem)]
    assert "".join(tokens) == "MODEL-B-1MODEL-B-2"
    assert all("MODEL-A" not in token for token in tokens)
    assert patched["body"]["content"] == "MODEL-B-1MODEL-B-2"

    # --- Direction 2: post-first-chunk failure -> no fallback attempted. ---
    cred_c = _credential("model-c", priority=1)
    cred_d = _credential("model-d", priority=2)
    _set_snapshot(version=1, chat=(cred_c, cred_d))

    model_c_partial_then_fails = make_streaming_llm_model_that_fails_after(
        ["MODEL-C-1"], RuntimeError("model C dies mid-stream")
    )

    def fake_build_model_never_called(credential: CredentialConfig) -> FunctionModel:
        raise AssertionError(
            f"fallback must never be attempted after a chunk already streamed "
            f"(got credential={credential.id!r})"
        )

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model_never_called)

    models = _models_with_generation(
        model_c_partial_then_fails,
        query_transformation=mock_sync_llm_model,
        credential=cred_c,
        snapshot_version=1,
        chunks=[_DUMMY_CHUNK],
    )
    java_client, patched = _java_client()
    queue = asyncio.Queue()
    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=models,
        queue=queue,
        session_factory=lambda: _SessionCtx(db_session),  # type: ignore[arg-type]
    )
    items = await _drain(queue)
    tokens = [item.text for item in items if isinstance(item, TokenItem)]
    assert tokens == ["MODEL-C-1"]
    assert all("MODEL-D" not in token for token in tokens)
    assert patched["body"]["content"] == "MODEL-C-1"
    assert patched["body"]["status"] == "ERROR"
