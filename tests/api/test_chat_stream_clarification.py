"""Start-of-turn panel gate and the cancel flow (SPEC-clarification-panel §2.3-§2.5).

Every refusal must happen before `POST /messages/turn`: no message, no quota.
"""

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_backend_java_client, get_graph_models, get_session_factory
from app.core.config import settings
from app.database.models import Base
from app.database.repositories.clarification_state import ClarificationRoundRepository
from app.graph.streaming_state import GraphModels
from app.integrations import backend_java_client
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app
from app.schemas.clarification import ClarificationPanel, PendingAdvisoryTask, PendingRound
from app.schemas.intent import ClassifiedTask
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model

CONVERSATION = "conv-1"
ASSISTANT_MESSAGE_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")


class _Java:
    def __init__(self, *, cancel_status: int = 200, turn_status: int = 201) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.cancel_status = cancel_status
        self.turn_status = turn_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.content else {}
        self.calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("/clarification"):
            return httpx.Response(self.cancel_status, json={})
        if request.url.path == "/messages/turn":
            if self.turn_status != 201:
                return httpx.Response(self.turn_status, json={"code": 2130, "message": "quota"})
            return httpx.Response(
                201,
                json={
                    "firstTurn": False,
                    "context": [],
                    "userMessage": {"id": "u-1", "status": "COMPLETED"},
                    "assistantMessage": {"id": str(ASSISTANT_MESSAGE_ID), "status": "STREAMING"},
                },
            )
        if request.method == "PATCH" and request.url.path.startswith("/messages/"):
            return httpx.Response(200, json={"status": body.get("status")})
        raise AssertionError(f"unexpected Java call {request.method} {request.url.path}")

    def paths(self) -> list[str]:
        return [path for _method, path, _body in self.calls]


def _streaming_model(chunks: list[str]) -> FunctionModel:
    async def stream(_messages: object, _info: object) -> Any:
        for chunk in chunks:
            yield chunk

    return FunctionModel(stream_function=stream)


@pytest.fixture
def java(client: TestClient) -> Iterator[_Java]:
    fake = _Java()
    app.dependency_overrides[get_backend_java_client] = lambda: BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(fake.handler)
    )
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=make_classification_llm_model("off_topic"),  # a resume never classifies
        query_transformation=FunctionModel(
            function=lambda _m, _i: ModelResponse(parts=[TextPart(content="hyde")])
        ),
        generation=_streaming_model(["Học phí hệ chính quy là ... [1]."]),
        retrieval=FakeRetrievalService(
            [RetrievedChunk(chunk_id="c1", content="học phí", source="s", score=0.9)]
        ),
    )
    yield fake


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_java_client, "JAVA_RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))


def _factory() -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = app.dependency_overrides[get_session_factory]()
    return factory


def _round() -> PendingRound:
    panel = ClarificationPanel.model_validate(
        {
            "panel_id": str(uuid.uuid4()),
            "questions": [
                {
                    "id": "q1",
                    "tab_label": "Điểm CK",
                    "prompt": "Điểm cuối kỳ",
                    "kind": "number",
                    "number": {"min": "0", "max": "10", "step": "0.01"},
                    "origin": "advisory",
                    "task_id": "T1",
                    "field": "ck",
                }
            ],
        }
    )
    return PendingRound(
        panel=panel,
        assistant_message_id=ASSISTANT_MESSAGE_ID,
        original_query="q",
        tasks=[
            PendingAdvisoryTask(
                task_id="T1", origin_tasks=[ClassifiedTask(intent="academic_advisory", query="q")]
            )
        ],
        created_at=datetime.now(UTC),
    )


def _seed(client: TestClient, *, claim: bool = False) -> PendingRound:
    pending = _round()

    async def seed() -> None:
        async with _factory()() as session:
            connection = await session.connection()
            await connection.run_sync(Base.metadata.create_all)
            repo = ClarificationRoundRepository(session)
            await repo.upsert_open(CONVERSATION, pending, {"program": "CNTT"})
            if claim:
                await repo.claim(CONVERSATION, pending.panel.panel_id, uuid.uuid4(), 210)
            await session.commit()

    assert client.portal is not None
    client.portal.call(seed)
    return pending


def _status(client: TestClient) -> str | None:
    async def read() -> str | None:
        async with _factory()() as session:
            return (await ClarificationRoundRepository(session).get_round(CONVERSATION)).status

    assert client.portal is not None
    return client.portal.call(read)


def _post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/api/v1/chat/stream", json={"conversation_id": CONVERSATION, **body})


def _submit(panel_id: uuid.UUID, answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"clarification": {"action": "submit", "panel_id": str(panel_id), "answers": answers}}


def _cancel(panel_id: uuid.UUID) -> dict[str, Any]:
    return {"clarification": {"action": "cancel", "panel_id": str(panel_id)}}


# --- §2.3 table ---------------------------------------------------------------


def test_clarification_without_any_round_is_stale(client: TestClient, java: _Java) -> None:
    response = _post(client, _cancel(uuid.uuid4()))
    assert response.status_code == 409 and response.json()["code"] == 4091
    assert java.calls == []


def test_message_while_panel_open_is_pending(client: TestClient, java: _Java) -> None:
    _seed(client)
    response = _post(client, {"message": "xin chào"})
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == 4092
    assert "panel_id" not in json.dumps(body)  # the capability token never leaks here
    assert java.calls == []


def test_message_while_processing_is_4093(client: TestClient, java: _Java) -> None:
    _seed(client, claim=True)
    response = _post(client, {"message": "xin chào"})
    assert response.status_code == 409 and response.json()["code"] == 4093
    assert java.calls == []


def test_wrong_panel_id_is_stale(client: TestClient, java: _Java) -> None:
    _seed(client)
    response = _post(client, _cancel(uuid.uuid4()))
    assert response.status_code == 409 and response.json()["code"] == 4091
    assert _status(client) == "OPEN"


def test_matching_panel_while_processing_is_stale(client: TestClient, java: _Java) -> None:
    pending = _seed(client, claim=True)
    response = _post(client, _cancel(pending.panel.panel_id))
    assert response.status_code == 409 and response.json()["code"] == 4091


def test_invalid_submit_is_4010_with_error_map(client: TestClient, java: _Java) -> None:
    pending = _seed(client)
    response = _post(
        client, _submit(pending.panel.panel_id, [{"question_id": "q1", "number": "11"}])
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 4010
    assert set(body["errors"]) == {"q1"}
    assert _status(client) == "OPEN"
    assert java.calls == []


def test_oversized_body_is_413_before_any_java_call(client: TestClient, java: _Java) -> None:
    pending = _seed(client)
    answers = [{"question_id": "q1", "number": "5"}]
    body = json.dumps({"conversation_id": CONVERSATION, **_submit(pending.panel.panel_id, answers)})
    response = client.post(
        "/api/v1/chat/stream",
        content=body + " " * 17_000,  # whitespace keeps the JSON valid
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413 and response.json()["code"] == 4131
    assert java.calls == []
    assert _status(client) == "OPEN"


# --- §2.5 cancel ------------------------------------------------------------------


def test_cancel_projects_then_clears_without_start_turn(client: TestClient, java: _Java) -> None:
    pending = _seed(client)
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": CONVERSATION, **_cancel(pending.panel.panel_id)},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: clarification_closed" in body
    assert f'"panel_id": "{pending.panel.panel_id}"' in body
    assert body.rstrip().endswith("event: done\ndata: {}")
    assert java.calls == [
        (
            "PATCH",
            f"/internal/messages/{ASSISTANT_MESSAGE_ID}/clarification",
            {"conversationId": CONVERSATION, "status": "cancelled"},
        )
    ]
    assert _status(client) is None


def test_cancel_projection_failure_restores_round_and_is_502(
    client: TestClient, java: _Java
) -> None:
    java.cancel_status = 500
    pending = _seed(client)
    response = _post(client, _cancel(pending.panel.panel_id))
    assert response.status_code == 502
    assert len(java.calls) == 4  # first try + 3 retries
    assert _status(client) == "OPEN"


def test_second_cancel_is_stale(client: TestClient, java: _Java) -> None:
    pending = _seed(client)
    assert _post(client, _cancel(pending.panel.panel_id)).status_code == 200
    response = _post(client, _cancel(pending.panel.panel_id))
    assert response.status_code == 409 and response.json()["code"] == 4091


# --- §2.4 submit ------------------------------------------------------------------


def _stream(client: TestClient, body: dict[str, Any]) -> tuple[int, str]:
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"conversation_id": CONVERSATION, **body}
    ) as response:
        return response.status_code, "".join(response.iter_text())


def test_submit_answers_and_resumes_the_original_question(
    client: TestClient, java: _Java, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    pending = _seed(client)
    status, body = _stream(
        client, _submit(pending.panel.panel_id, [{"question_id": "q1", "number": "6.5"}])
    )

    assert status == 200
    assert "Học phí hệ chính quy" in body and "event: done" in body
    turn = next(b for m, p, b in java.calls if p == "/messages/turn")
    assert turn["content"] == "Điểm CK: 6.5"
    answers = turn["metadata"]["clarification_answers"]
    assert answers["panel_id"] == str(pending.panel.panel_id)
    assert answers["items"][0]["display"] == "6.5"
    assert java.paths().count("/messages/turn") == 1
    assert _status(client) is None  # round consumed


def test_second_submit_of_the_same_panel_is_stale(
    client: TestClient, java: _Java, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    pending = _seed(client)
    body = _submit(pending.panel.panel_id, [{"question_id": "q1", "number": "6.5"}])
    assert _stream(client, body)[0] == 200
    response = _post(client, body)
    assert response.status_code == 409 and response.json()["code"] == 4091
    assert java.paths().count("/messages/turn") == 1


def test_start_turn_failure_hands_the_panel_back(client: TestClient, java: _Java) -> None:
    java.turn_status = 429
    pending = _seed(client)
    response = _post(
        client, _submit(pending.panel.panel_id, [{"question_id": "q1", "number": "6.5"}])
    )
    assert response.status_code == 429
    assert _status(client) == "OPEN"


def test_claimed_turn_past_its_deadline_writes_nothing(
    client: TestClient, java: _Java, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "CHAT_CLAIMED_TURN_DEADLINE_SECONDS", 0.001)
    pending = _seed(client)
    status, body = _stream(
        client, _submit(pending.panel.panel_id, [{"question_id": "q1", "number": "6.5"}])
    )
    assert status == 200
    assert "event: error" in body
    # start_turn happened before the claim's deadline started mattering; nothing after it.
    assert not any(p.startswith("/messages/") and m == "PATCH" for m, p, _b in java.calls)
    assert _status(client) == "PROCESSING"  # left for the lease to expire
