"""End-to-end flow tests through the real `POST /chat/stream` endpoint,
backend-java mocked via `httpx.MockTransport` and the LLM mocked via
`FunctionModel` - no live Java or Qdrant is reachable in the test
environment, so this is entirely mock-based (retrieval uses
`tests.llm_mocks.FakeRetrievalService`, a canned chunk list, rather than a
live `RetrievalService`). What IS exercised here for real: the full HTTP
endpoint, the graph orchestrator's branching, the clarification guard's
2-turn round trip, and the cancellation-safe persistence lifecycle - all
through the real `TestClient`, not by calling internal functions directly.

Not implemented in this file:
- JWT invalid -> 401: enforced by api-gateway (a separate repo/service not
  reachable from this test process) - this service only ever sees "header
  absent" (guest) or "header present and well-formed" (gateway already
  validated it). Malformed-header -> 400 IS covered (see
  tests/api/test_chat_stream_endpoint.py).
- retry-limit-reached scenario: covered at the unit level in
  tests/graph/test_security_context_node.py (the guard's own retry-count
  logic) - not duplicated here as a third HTTP round trip.
"""

import json
from collections.abc import Callable, Generator, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio.session import AsyncSession

from app.api.deps import get_backend_java_client, get_graph_models, get_session_factory
from app.core.config import settings
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATES
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model

_DUMMY_CHUNK = RetrievedChunk(
    chunk_id="c1", content="dummy retrieved content", source="s", score=0.9
)

_TRAINING_TYPE_ASK_FORM = (
    "Về việc miễn học phần Giáo dục Quốc phòng, quy định miễn giảm hiện khác nhau "
    "tuỳ theo hệ đào tạo sinh viên đang theo học [1].\n\nBạn đang học theo hệ đào tạo nào ạ?\n\n"
    "```json\n"
    '{"type": "ask_user_form", "fields": [{"field": "training_type", "options": '
    '[{"id": "chinh_quy"}, {"id": "lien_thong"}, {"id": "vlvh"}]}]}\n'
    "```"
)


class _JavaBackend:
    """In-memory fake of backend-java's conversation/message store, with just
    enough state (a per-conversation message list) to answer
    `GET /messages/conversation/{id}` truthfully across turns."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next_id = 1
        self._history: dict[str, list[dict[str, Any]]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.content else {}
        self.calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": body,
                "authorization": request.headers.get("authorization"),
            }
        )

        if request.method == "GET" and request.url.path.startswith("/messages/conversation/"):
            conversation_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=self._history.get(conversation_id, []))

        if request.method == "POST" and request.url.path == "/messages":
            message_id = f"msg-{self._next_id}"
            self._next_id += 1
            record = {"id": message_id, **body}
            self._history.setdefault(body["conversationId"], []).append(record)
            return httpx.Response(201, json=record)

        if request.method == "PATCH" and request.url.path.startswith("/messages/"):
            return httpx.Response(200, json={"status": body.get("status")})

        raise AssertionError(f"unexpected call {request.method} {request.url.path}")


@pytest.fixture(autouse=True)
def _clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.pop(get_backend_java_client, None)
    app.dependency_overrides.pop(get_graph_models, None)


def _install_java(java: _JavaBackend) -> None:
    app.dependency_overrides[get_backend_java_client] = lambda: BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(java.handler)
    )


def _session_factory(client: TestClient) -> async_sessionmaker[AsyncSession]:
    """Reach into the `client` fixture's `get_session_factory` override so this
    test can read back clarification state the same way `run_and_persist`
    itself wrote it (same in-memory SQLite engine as the request path)."""

    del client  # the override lives on the app, not the client object
    factory = app.dependency_overrides.get(get_session_factory)
    assert factory is not None, "client fixture must override get_session_factory"
    result: async_sessionmaker[AsyncSession] = factory()
    return result


@pytest.mark.asyncio
async def test_clarification_two_turn_round_trip_via_real_endpoint(
    client: TestClient,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors missing_metadata_clarification_design.md section 8's worked
    example: turn 1 surfaces an ask_user_form for training_type; turn 2
    confirms "chinh_quy" and resumes straight at QueryTransformationNode,
    skipping classification, with the value folded into confirmed_metadata.
    """

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    java = _JavaBackend()
    _install_java(java)
    session_factory = _session_factory(client)

    # --- Turn 1: question triggers a Type B clarification request. ---
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("HyDE: quy định miễn giảm GDQP"),
        generation=mock_streaming_llm_model([_TRAINING_TYPE_ASK_FORM]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "conversation_id": "conv-clarify",
            "message": "Sinh viên năm cuối có được miễn học phần Giáo dục Quốc phòng không?",
        },
    ) as response:
        assert response.status_code == 200
        turn1_body = "".join(response.iter_text())

    assert "training_type" in turn1_body

    async with session_factory() as session:
        repo = ClarificationStateRepository(session)
        pending = await repo.get_pending_clarification("conv-clarify")
    assert pending is not None
    assert pending.missing_fields == ["training_type"]
    assert pending.origin_node == "QueryTransformationNode"

    # --- Turn 2: user picks "Chính quy" - guard resolves deterministically,
    # classification is skipped (a misleading classification mock would
    # change the response if it were reached), advisory flow resumes. ---
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=make_classification_llm_model("off_topic"),  # must NOT be reached
        query_transformation=mock_sync_llm_model("HyDE: GDQP hệ chính quy"),
        generation=mock_streaming_llm_model(["Sinh viên hệ chính quy được miễn GDQP [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-clarify", "message": "Chính quy ạ"},
    ) as response:
        assert response.status_code == 200
        turn2_body = "".join(response.iter_text())

    assert "Sinh viên hệ chính quy được miễn GDQP" in turn2_body

    async with session_factory() as session:
        repo = ClarificationStateRepository(session)
        pending_after = await repo.get_pending_clarification("conv-clarify")
        confirmed_after = await repo.get_confirmed_metadata("conv-clarify")
    assert pending_after is None
    assert confirmed_after == {"training_type": "chinh_quy"}


@pytest.mark.asyncio
async def test_ticket_fallback_when_no_valid_context(
    client: TestClient,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    java = _JavaBackend()
    _install_java(java)
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("hyde"),
        generation=mock_streaming_llm_model(
            ["Hệ thống chưa tìm thấy quy định chính thức cho câu hỏi này."]
        ),
        retrieval=FakeRetrievalService(),
    )

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-fallback", "message": "Điều kiện học bổng là gì?"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "chưa tìm thấy" in body.lower()
    patches = [c for c in java.calls if c["method"] == "PATCH"]
    assert patches[0]["body"]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_guest_without_authorization_header_completes_full_round_trip(
    client: TestClient,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    java = _JavaBackend()
    _install_java(java)
    # off_topic is the fully static, model-independent path (no `generation`
    # call - see `OFF_TOPIC_TEMPLATES`). A general-knowledge question like this
    # one is classified as off_topic (see message_classification.yaml).
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=make_classification_llm_model("off_topic"),
        query_transformation=mock_sync_llm_model("hyde"),
        generation=mock_streaming_llm_model(["unused"]),
        retrieval=FakeRetrievalService(),
    )

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-guest", "message": "1 + 1 bằng mấy?"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    # body is SSE `data: "<json-escaped-string>"`; the template is picked at random.
    assert any(json.dumps(template, ensure_ascii=False) in body for template in OFF_TOPIC_TEMPLATES)
    assert all(call["authorization"] is None for call in java.calls)
