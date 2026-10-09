"""End to end through POST /chat/stream (fake Java + scripted LLM): a course-score
question missing the practice scores raises a panel; submitting it computes the
result without calling the extractor again (SPEC-calculation-node, success criteria)."""

import json
import re
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.api.deps import get_backend_java_client, get_graph_models
from app.graph.streaming_state import GraphModels
from app.integrations import backend_java_client
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app
from tests.llm_mocks import FakeRetrievalService

QUERY = "Môn 2 tín lý thuyết 1 tín thực hành, TX 8 GK 7 CK 6.5 thì tổng kết bao nhiêu?"


class _Java:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.turns = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.content else {}
        self.calls.append((request.method, request.url.path, body))
        if request.url.path == "/messages/turn":
            self.turns += 1
            return httpx.Response(
                201,
                json={
                    "firstTurn": False,
                    "context": [],
                    "userMessage": {"id": f"u-{self.turns}", "status": "COMPLETED"},
                    "assistantMessage": {"id": str(uuid.uuid4()), "status": "STREAMING"},
                },
            )
        return httpx.Response(200, json={})


class _Llm:
    """Answers non-streamed calls in order and counts them."""

    def __init__(self, *outputs: object) -> None:
        self.remaining = [o if isinstance(o, str) else json.dumps(o) for o in outputs]
        self.calls = 0

    def model(self) -> FunctionModel:
        def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            self.calls += 1
            return ModelResponse(parts=[TextPart(content=self.remaining.pop(0))])

        return FunctionModel(function=respond)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_java_client, "JAVA_RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))


@pytest.fixture
def java(client: TestClient) -> Iterator[_Java]:
    fake = _Java()
    app.dependency_overrides[get_backend_java_client] = lambda: BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(fake.handler)
    )
    yield fake


def _events(body: str) -> list[tuple[str, str]]:
    return re.findall(r"event: (\w+)\ndata: (.*)\n\n", body)


def _stream(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, str]]:
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"conversation_id": "conv-calc", **payload}
    ) as response:
        assert response.status_code == 200, response.read()
        return _events("".join(response.iter_text()))


def test_missing_practice_scores_then_answer_computes_the_course_score(
    client: TestClient, java: _Java
) -> None:
    llm = _Llm(
        {"tasks": [{"intent": "academic_calculation", "query": QUERY}], "confidence": 0.9},
        {
            "formula_id": "course_score",
            "params": {"tclt": 2, "tcth": 1, "tbtx": 8, "gk": 7, "ck": 6.5},
        },
        "Điểm chữ B nghĩa là bạn đã qua học phần.",
    )
    app.dependency_overrides[get_graph_models] = lambda: GraphModels(
        classification=llm.model(),
        query_transformation="unused",
        generation=llm.model(),
        retrieval=FakeRetrievalService(),
    )

    first = _stream(client, {"message": QUERY})
    panel = json.loads(next(data for name, data in first if name == "clarification"))
    (question,) = panel["questions"]
    assert (question["id"], question["kind"]) == ("q1", "number_list")
    assert "origin" not in question and "field" not in question
    assert llm.calls == 2  # classification + extractor

    second = _stream(
        client,
        {
            "clarification": {
                "action": "submit",
                "panel_id": panel["panel_id"],
                "answers": [{"question_id": "q1", "numbers": ["9", "8"]}],
            }
        },
    )
    text = "".join(json.loads(data) for name, data in second if name == "token")
    assert "Kết quả: ĐTKHP **7.5** · Điểm chữ **B** · Thang 4 **3.0**" in text
    assert "Điểm chữ B nghĩa là bạn đã qua học phần." in text
    assert llm.calls == 3  # only the note - no second classification or extraction
    assert not any(name == "clarification" for name, _ in second)

    turn = [body for _m, path, body in java.calls if path == "/messages/turn"][-1]
    assert turn["content"] == "Điểm TH: 9, 8"
    assert turn["metadata"]["clarification_answers"]["items"][0]["display"] == "9, 8"
    finalize = [
        body for m, path, body in java.calls if m == "PATCH" and path.startswith("/messages/")
    ][-1]
    assert finalize["metadata"]["calculation"]["items"][0]["status"] == "computed"
