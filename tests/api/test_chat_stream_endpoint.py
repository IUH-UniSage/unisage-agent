"""T1.13b/e: full round-trip through `POST /chat/stream` via TestClient,
with backend-java mocked (httpx.MockTransport) and the LLM mocked
(FunctionModel) - no live network call anywhere in this file.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel

from app.api.deps import get_backend_java_client, get_graph_models
from app.core.config import settings
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService


class _JavaBackend:
    """In-memory fake of backend-java's message store, driven by an
    httpx.MockTransport handler - captures every call for assertions."""

    def __init__(self, *, reject_user_message: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next_id = 1
        self._reject_user_message = reject_user_message

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.content else {}
        self.calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": body,
                "authorization": request.headers.get("authorization"),
                "x_internal_secret": request.headers.get("x-internal-secret"),
                "x_forwarded_for": request.headers.get("x-forwarded-for"),
            }
        )

        if (
            request.method == "POST"
            and request.url.path == "/messages"
            and body.get("role") == "USER"
            and self._reject_user_message is not None
        ):
            return httpx.Response(self._reject_user_message, json={"message": "rejected"})

        if request.method == "GET" and request.url.path.startswith("/messages/conversation/"):
            return httpx.Response(200, json=[])

        if request.method == "POST" and request.url.path == "/messages":
            message_id = f"msg-{self._next_id}"
            self._next_id += 1
            return httpx.Response(
                201, json={"id": message_id, "status": body.get("status", "COMPLETED")}
            )

        if request.method == "PATCH" and request.url.path.startswith("/messages/"):
            return httpx.Response(200, json={"status": body.get("status")})

        raise AssertionError(f"unexpected call {request.method} {request.url.path}")


@pytest.fixture
def mock_graph_models(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> GraphModels:
    return GraphModels(
        classification=mock_sync_llm_model("academic_advisory"),
        direct_llm=mock_streaming_llm_model(["42"]),
        query_transformation=mock_sync_llm_model("hyde doc"),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng."]),
        retrieval=FakeRetrievalService(
            [
                RetrievedChunk(
                    chunk_id="c1", content="dummy retrieved content", source="s", score=0.9
                )
            ]
        ),
    )


def _override_java(java: _JavaBackend) -> None:
    def _get_client() -> BackendJavaClient:
        return BackendJavaClient(
            base_url="http://java.test", transport=httpx.MockTransport(java.handler)
        )

    app.dependency_overrides[get_backend_java_client] = _get_client


def _override_models(models: GraphModels) -> None:
    app.dependency_overrides[get_graph_models] = lambda: models


@pytest.fixture(autouse=True)
def _clear_extra_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.pop(get_backend_java_client, None)
    app.dependency_overrides.pop(get_graph_models, None)


def test_missing_conversation_id_is_a_validation_error(client: TestClient) -> None:
    """This project's global RequestValidationError handler maps every
    validation failure to `ErrorCode.VALIDATION_ERROR` (HTTP 400), not
    FastAPI's default 422 - see app/main.py's validation_exception_handler."""

    response = client.post("/api/v1/chat/stream", json={"message": "hi"})

    assert response.status_code == 400


def test_missing_authorization_header_is_treated_as_guest_and_streams(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    java = _JavaBackend()
    _override_java(java)
    _override_models(mock_graph_models)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "Điều kiện học bổng là gì?"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: done" in body
    # Guest -> no Authorization header forwarded to Java on any call.
    assert all(call["authorization"] is None for call in java.calls)


def test_malformed_trusted_header_is_400_not_401(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "hi"},
        headers={"X-User-Id": "u1", "X-User-Department-Access": "not-json"},
    )

    assert response.status_code == 400


def test_java_rejects_user_message_returns_error_without_running_graph(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    java = _JavaBackend(reject_user_message=403)
    _override_java(java)
    _override_models(mock_graph_models)

    response = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": "someone-elses-conv", "message": "hi"},
    )

    assert response.status_code == 403
    # Only the (rejected) USER-message POST happened - no placeholder created.
    post_message_calls = [c for c in java.calls if c["path"] == "/messages"]
    assert len(post_message_calls) == 1


def test_successful_stream_creates_user_then_assistant_then_patches_completed(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    java = _JavaBackend()
    _override_java(java)
    _override_models(mock_graph_models)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "Điều kiện học bổng loại giỏi là gì?"},
        headers={
            "X-User-Id": "u1",
            "X-User-Role": "SINH_VIEN",
            "X-User-Code": "SV001",
            "X-User-Department-Access": "[]",
            "X-User-Permissions": "[]",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_text())  # drain the stream so the background task completes

    message_posts = [c for c in java.calls if c["path"] == "/messages" and c["method"] == "POST"]
    assert len(message_posts) == 2
    assert message_posts[0]["body"]["role"] == "USER"
    assert message_posts[1]["body"]["role"] == "ASSISTANT"
    assert message_posts[1]["body"]["status"] == "STREAMING"

    patches = [c for c in java.calls if c["method"] == "PATCH"]
    assert len(patches) == 1
    assert patches[0]["body"]["status"] == "COMPLETED"


def test_every_java_call_carries_x_internal_secret(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    """Bug 1: this service calls backend-java directly (bypassing the API
    Gateway), so it must send `X-Internal-Secret` itself on every call, not
    rely on `Authorization` forwarding alone."""

    java = _JavaBackend()
    _override_java(java)
    _override_models(mock_graph_models)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "Điều kiện học bổng là gì?"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_text())

    assert len(java.calls) >= 3  # GET history, POST user, POST assistant, PATCH
    assert all(call["x_internal_secret"] == settings.INTERNAL_SECRET_KEY for call in java.calls)


def test_incoming_x_forwarded_for_is_forwarded_to_java(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    """Bug 1: prefer an `X-Forwarded-For` already set on the inbound request
    (e.g. by the API Gateway) over `request.client.host`, and forward its
    first IP to Java as our own outgoing `X-Forwarded-For` - this is how
    Java's guest-conversation ownership `claim()` check can see the real
    browser IP even though Python is the one calling `POST /messages`."""

    java = _JavaBackend()
    _override_java(java)
    _override_models(mock_graph_models)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "Điều kiện học bổng là gì?"},
        headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_text())

    message_posts = [c for c in java.calls if c["path"] == "/messages" and c["method"] == "POST"]
    assert len(message_posts) == 2
    assert all(c["x_forwarded_for"] == "203.0.113.7" for c in message_posts)

    patches = [c for c in java.calls if c["method"] == "PATCH"]
    assert len(patches) == 1
    assert patches[0]["x_forwarded_for"] == "203.0.113.7"


def test_falls_back_to_request_client_host_without_x_forwarded_for(
    client: TestClient, mock_graph_models: GraphModels
) -> None:
    """No inbound `X-Forwarded-For` -> fall back to the request's own peer
    address (`request.client.host`) rather than sending nothing."""

    java = _JavaBackend()
    _override_java(java)
    _override_models(mock_graph_models)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-1", "message": "Điều kiện học bổng là gì?"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_text())

    message_posts = [c for c in java.calls if c["path"] == "/messages" and c["method"] == "POST"]
    assert len(message_posts) == 2
    # Starlette's TestClient sets the synthetic peer address to "testclient".
    assert all(c["x_forwarded_for"] == "testclient" for c in message_posts)
