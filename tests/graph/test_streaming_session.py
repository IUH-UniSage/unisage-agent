import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import httpx2
import openai
import pytest
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors.llm_failure import FailureReason
from app.core.errors.provider_errors import EmbeddingProviderError
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import ModelRegistryError
from app.core.usage.usage_recorder import UsageRecorder
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATES
from app.graph.queue_items import DoneItem, ErrorItem, QueueItem, TokenItem
from app.graph.stream_error_codes import LLM_STREAM_INTERRUPTED
from app.graph.streaming_session import _admin_warnings, _error_item_for, run_and_persist
from app.graph.streaming_state import AdminWarning, GraphInput, GraphModels, GraphOutput
from app.integrations.backend_java_client import BackendJavaClient
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model


def _models(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> GraphModels:
    return GraphModels(
        classification=make_classification_llm_model("off_topic"),
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
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    assert patched["method"] == "PATCH"
    assert patched["body"]["status"] == "COMPLETED"
    assert patched["body"]["content"] in OFF_TOPIC_TEMPLATES
    assert "citations" not in patched["body"]

    tokens = []
    while True:
        item = await queue.get()
        if isinstance(item, DoneItem):
            break
        assert isinstance(item, TokenItem)
        tokens.append(item.text)
    assert tokens == [patched["body"]["content"]]


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
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    assert patched["body"]["status"] == "ERROR"
    # An error item is pushed immediately before the end-of-stream sentinel
    # whenever the graph raised.
    error_item = await queue.get()
    assert isinstance(error_item, ErrorItem)
    assert error_item.code == LLM_STREAM_INTERRUPTED
    assert error_item.retryable is True
    assert isinstance(await queue.get(), DoneItem)


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
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    repo = ClarificationStateRepository(db_session)
    # off_topic never touches confirmed_metadata, but a row should still
    # exist (upserted with the empty defaults).
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
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    async def _drain_to_sentinel() -> None:
        while not isinstance(await queue.get(), DoneItem):
            pass

    await asyncio.wait_for(_drain_to_sentinel(), timeout=2.0)

    # The clarification-state write (which runs AFTER the Java PATCH in the
    # function body) must still have happened too - the PATCH failure must
    # not short-circuit it.
    repo = ClarificationStateRepository(db_session)
    assert await repo.get_confirmed_metadata("conv-1") == {}


@pytest.mark.asyncio
async def test_run_and_persist_sends_citations_when_graph_produced_them(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citation = {"index": 1, "documentId": "doc-1", "title": "Quy chế A"}

    async def _fake_graph(*_args: object, **_kwargs: object) -> GraphOutput:
        return GraphOutput(response_text="Học phí là 35 triệu [1].", citations=[citation])

    monkeypatch.setattr("app.graph.streaming_session.run_graph", _fake_graph)

    patched: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        patched["body"] = json.loads(request.read())
        return httpx.Response(200, json={"id": "msg-2", "status": "COMPLETED"})

    java_client = BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(handler)
    )
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    assert patched["body"]["citations"] == [citation]


@pytest.mark.asyncio
async def test_run_and_persist_reports_llm_unavailable_when_credentials_exhausted(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`NoAvailableCredentialError` (every CHAT credential cooling down/excluded
    before any chunk streamed) must map to `LLM_UNAVAILABLE`, not the generic
    `LLM_STREAM_INTERRUPTED` code."""

    async def _boom(*_args: object, **_kwargs: object) -> None:
        raise NoAvailableCredentialError("CHAT")

    monkeypatch.setattr("app.graph.streaming_session.run_graph", _boom)
    # CHAT credentials exist (just all cooling down/excluded) - with none configured at
    # all this would instead be the more specific LLM_NOT_CONFIGURED.
    monkeypatch.setattr(
        "app.core.errors.llm_failure.active_credentials_for", lambda _purpose: ("cred",)
    )

    java_client = BackendJavaClient(
        base_url="http://java.test",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"id": "msg-2"})),
    )
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

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
        usage_recorder=UsageRecorder(request_id="test-request", purpose="CHAT"),
        queue=queue,
        session_factory=lambda: _SessionCtx(),  # type: ignore[arg-type]
    )

    error_item = await queue.get()
    assert isinstance(error_item, ErrorItem)
    assert error_item.code == FailureReason.LLM_UNAVAILABLE
    assert error_item.retryable is False
    assert isinstance(await queue.get(), DoneItem)


def _embedding_not_configured() -> EmbeddingProviderError:
    """What retrieval raises when no EMBEDDING credential is active (see
    `OpenAIEmbedder._resolve_from_registry`)."""

    try:
        raise ModelRegistryError("no ACTIVE EMBEDDING credential")
    except ModelRegistryError as cause:
        error = EmbeddingProviderError(str(cause))
        error.__cause__ = cause
        return error


@pytest.mark.parametrize(
    ("exc", "code", "message_fragment"),
    [
        (
            NoAvailableCredentialError(
                "CHAT",
                last_error=openai.AuthenticationError(
                    "bad key",
                    response=httpx2.Response(401, request=httpx2.Request("POST", "http://p.test")),
                    body=None,
                ),
            ),
            "LLM_AUTH_FAILED",
            "HTTP 401",
        ),
        (
            _embedding_not_configured(),
            "LLM_NOT_CONFIGURED",
            "Mô hình Embedding",
        ),
        (KeyError("bug"), "LLM_STREAM_INTERRUPTED", "KeyError"),
    ],
)
def test_error_item_names_the_actual_cause(
    exc: Exception, code: str, message_fragment: str
) -> None:
    """The SSE `event: error` must tell the user which model failed and why (bad key,
    retrieval embedding not configured, ...) - not one generic sentence for everything."""

    item = _error_item_for(exc, streamed_any=False)

    assert item.code == code
    assert message_fragment in item.message


def _auth_failure() -> NoAvailableCredentialError:
    return NoAvailableCredentialError(
        "CHAT",
        last_error=openai.AuthenticationError(
            "bad key",
            response=httpx2.Response(401, request=httpx2.Request("POST", "http://p.test")),
            body=None,
        ),
    )


@pytest.mark.parametrize(
    ("exc", "message_start", "retryable"),
    [
        (_auth_failure(), "Trợ lý AI đang tạm ngưng do sự cố hệ thống.", False),
        (
            NoAvailableCredentialError(
                "CHAT",
                last_error=openai.APITimeoutError(request=httpx2.Request("POST", "http://p.test")),
            ),
            "Trợ lý AI đang bận hoặc tạm thời gián đoạn",
            True,
        ),
        (KeyError("bug"), "Trợ lý AI đang tạm ngưng do sự cố hệ thống.", True),
    ],
)
def test_error_item_for_students_is_a_plain_category(
    exc: Exception, message_start: str, retryable: bool
) -> None:
    item = _error_item_for(exc, streamed_any=False, detailed=False, reference="a1b2c3d4")

    assert item.message.startswith(message_start)
    assert item.message.endswith("(Mã tham chiếu: a1b2c3d4)")
    assert "HTTP" not in item.message and "KeyError" not in item.message
    assert item.retryable is retryable
    # The precise cause still travels as the code - for logs/support, not for display.
    assert item.code != ""


def test_error_item_for_admins_keeps_the_detail_and_reference() -> None:
    item = _error_item_for(_auth_failure(), streamed_any=False, detailed=True, reference="a1b2c3d4")

    assert item.code == "LLM_AUTH_FAILED"
    assert "HTTP 401" in item.message
    assert "a1b2c3d4" in item.message


def test_partial_answer_failure_is_flagged_for_students() -> None:
    item = _error_item_for(_auth_failure(), streamed_any=True, detailed=False, reference=None)

    assert item.message.startswith("Câu trả lời bị gián đoạn giữa chừng.")


def _out_of_credits() -> GraphOutput:
    return GraphOutput(
        response_text="Hiện chưa có quy định...",
        used_ticket_fallback=True,
        admin_warnings=[
            AdminWarning(
                code="WEB_SEARCH_CREDITS_EXHAUSTED",
                message="Tìm kiếm web thất bại ...: Tài khoản Tavily đã hết credit (HTTP 432).",
            )
        ],
    )


def _input_with(permissions: list[str], user_id: str | None = "u1") -> GraphInput:
    return GraphInput(
        conversation_id="conv-1",
        user_message="Địa chỉ cơ sở Thanh Hóa?",
        is_first_turn=False,
        security=AcademicSecurityContext(user_id=user_id, permissions=permissions),
    )


def test_ai_admin_is_told_what_was_skipped() -> None:
    (warning,) = _admin_warnings(_out_of_credits(), _input_with(["LLM_TRACE_LOG_READ"]))

    assert warning.code == "WEB_SEARCH_CREDITS_EXHAUSTED"
    assert "hết credit" in warning.message


@pytest.mark.parametrize("permissions", [[], ["MESSAGE_SEND", "CHAT_MODEL_READ"]])
def test_students_and_guests_get_no_admin_warning(permissions: list[str]) -> None:
    assert _admin_warnings(_out_of_credits(), _input_with(permissions)) == []
