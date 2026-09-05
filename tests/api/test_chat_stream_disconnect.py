"""T1.13d: client disconnect mid-stream must NOT stop `run_and_persist`.

Simulates "client rớt mạng giữa chừng" by reading only the first SSE chunk
and then closing the response early, never draining the queue to its
end-of-stream sentinel. `run_and_persist` runs in its own
`asyncio.create_task()`, independent of the SSE generator Starlette cancels
on disconnect - so backend-java must still receive exactly one PATCH with
the final status, regardless of whether anything ever read the rest of the
stream.
"""

import json
import time
from collections.abc import Callable, Generator, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel

from app.api.deps import get_backend_java_client, get_graph_models
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app


class _JavaBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next_id = 1

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.content else {}
        self.calls.append({"method": request.method, "path": request.url.path, "body": body})

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


@pytest.fixture(autouse=True)
def _clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.pop(get_backend_java_client, None)
    app.dependency_overrides.pop(get_graph_models, None)


def test_client_disconnect_mid_stream_still_patches_completed(
    client: TestClient,
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    java = _JavaBackend()
    app.dependency_overrides[get_backend_java_client] = lambda: BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(java.handler)
    )
    # Multiple tokens so there is something left in the queue when we bail
    # out after the first one.
    models = GraphModels(
        classification=mock_sync_llm_model("general_knowledge"),
        direct_llm=mock_streaming_llm_model(["token-1 ", "token-2 ", "token-3"]),
        query_transformation=mock_sync_llm_model("hyde"),
        generation=mock_streaming_llm_model(["unused"]),
    )
    app.dependency_overrides[get_graph_models] = lambda: models

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"conversation_id": "conv-disconnect", "message": "1 + 1 bằng mấy?"},
    ) as response:
        assert response.status_code == 200
        iterator = response.iter_text()
        next(iterator)  # read only the first chunk, then abandon the stream

    # The response context manager above has already closed the client-side
    # stream (simulating disconnect) without draining it to "event: done".
    # run_and_persist keeps running regardless - poll briefly for its PATCH,
    # since it finishes asynchronously on the TestClient's own event loop.
    deadline = time.monotonic() + 2.0
    patches = [c for c in java.calls if c["method"] == "PATCH"]
    while not patches and time.monotonic() < deadline:
        time.sleep(0.05)
        patches = [c for c in java.calls if c["method"] == "PATCH"]

    assert len(patches) == 1
    assert patches[0]["body"]["status"] == "COMPLETED"
    assert patches[0]["body"]["content"] == "token-1 token-2 token-3"
