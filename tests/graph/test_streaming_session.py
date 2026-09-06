import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories.clarification_state import ClarificationStateRepository
from app.graph.streaming_session import run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService


def _models(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> GraphModels:
    return GraphModels(
        classification=mock_sync_llm_model("general_knowledge"),
        direct_llm=mock_streaming_llm_model(["4"]),
        query_transformation=mock_sync_llm_model("hyde"),
        generation=mock_streaming_llm_model(["ans"]),
        retrieval=FakeRetrievalService(),
    )


def _graph_input() -> GraphInput:
    return GraphInput(
        conversation_id="conv-1",
        user_message="2 + 2 bằng mấy?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )


@pytest.mark.asyncio
async def test_run_and_persist_patches_completed_and_signals_queue_end(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
) -> None:
    patched: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        patched["method"] = request.method
        patched["body"] = json.loads(request.read())
        return httpx.Response(200, json={"id": "msg-2", "status": "COMPLETED"})

    java_client = BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(handler)
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    class _SessionCtx:
        async def __aenter__(self) -> AsyncSession:
            return db_session

        async def __aexit__(self, *exc: object) -> None:
            return None

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization="Bearer token",
        graph_input=_graph_input(),
        models=_models(mock_sync_llm_model, mock_streaming_llm_model),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    assert patched["method"] == "PATCH"
    assert patched["body"]["status"] == "COMPLETED"
    assert patched["body"]["content"] == "4"

    tokens = []
    while True:
        token = await queue.get()
        if token is None:
            break
        tokens.append(token)
    assert tokens == ["4"]


@pytest.mark.asyncio
async def test_run_and_persist_patches_error_on_graph_exception(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("LLM provider exploded")

    monkeypatch.setattr("app.graph.streaming_session.run_graph", _boom)

    patched: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        patched["body"] = json.loads(request.read())
        return httpx.Response(200, json={"id": "msg-2", "status": "ERROR"})

    java_client = BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(handler)
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    class _SessionCtx:
        async def __aenter__(self) -> AsyncSession:
            return db_session

        async def __aexit__(self, *exc: object) -> None:
            return None

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=_models(mock_sync_llm_model, mock_streaming_llm_model),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    assert patched["body"]["status"] == "ERROR"
    # end-of-stream sentinel still arrives even though the graph failed.
    assert await queue.get() is None


@pytest.mark.asyncio
async def test_run_and_persist_persists_clarification_state_on_success(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg-2", "status": "COMPLETED"})

    java_client = BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(handler)
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    class _SessionCtx:
        async def __aenter__(self) -> AsyncSession:
            return db_session

        async def __aexit__(self, *exc: object) -> None:
            return None

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-42",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=_models(mock_sync_llm_model, mock_streaming_llm_model),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    repo = ClarificationStateRepository(db_session)
    # general_knowledge -> DirectLLMNode never touches confirmed_metadata,
    # but a row should still exist (upserted with the empty defaults).
    assert await repo.get_confirmed_metadata("conv-42") == {}


@pytest.mark.asyncio
async def test_queue_sentinel_still_arrives_when_java_patch_raises_unexpected_error(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 2 regression: before the fix, `run_and_persist` only wrapped the
    Java PATCH call in `except BackendJavaError` - an unexpected exception
    type (e.g. the `update_message`/`_request` `TypeError` on an empty
    response body, or any other bug) propagated straight out of
    `run_and_persist`, skipping `await queue.put(None)` entirely and hanging
    the SSE generator's `while True: token = await queue.get()` forever.

    Simulates that by making `update_message` raise a plain `TypeError`
    (deliberately NOT a `BackendJavaError`) and asserting the sentinel still
    reaches the queue - with a real deadline so this test fails fast, not by
    hanging, if the regression comes back.
    """

    async def _boom_update_message(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise TypeError("'NoneType' object is not a mapping")

    monkeypatch.setattr(BackendJavaClient, "update_message", _boom_update_message)

    java_client = BackendJavaClient(
        base_url="http://java.test",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    class _SessionCtx:
        async def __aenter__(self) -> AsyncSession:
            return db_session

        async def __aexit__(self, *exc: object) -> None:
            return None

    await run_and_persist(
        java_client=java_client,
        conversation_id="conv-1",
        assistant_message_id="msg-2",
        authorization=None,
        graph_input=_graph_input(),
        models=_models(mock_sync_llm_model, mock_streaming_llm_model),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    async def _drain_to_sentinel() -> None:
        while await queue.get() is not None:
            pass

    await asyncio.wait_for(_drain_to_sentinel(), timeout=2.0)

    # The clarification-state write (which runs AFTER the Java PATCH in the
    # function body) must still have happened too - the PATCH failure must
    # not short-circuit it.
    repo = ClarificationStateRepository(db_session)
    assert await repo.get_confirmed_metadata("conv-1") == {}
