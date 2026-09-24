"""Client disconnect mid-stream must NOT stop `run_and_persist`.

The previous version of this test used `starlette.testclient.TestClient`,
which runs the ENTIRE ASGI app call (including any `asyncio.create_task()`
scheduled inside it) synchronously to completion before `client.stream(...)`
ever hands control back to the test - by the time the test's `with` block's
body ran, the whole request (background task included) had already
finished. "Closing the stream early" in that version did nothing real: there
was no live connection left to cancel, so the test could not fail even if
cancellation-safety were completely broken.

`httpx.ASGITransport` does NOT fix this either - by design, its
`handle_async_request` awaits the whole `app(scope, receive, send)` call to
completion, collecting every `http.response.body` chunk, before it ever
returns an `httpx.Response` back to the caller (see
`httpx._transports.asgi.ASGITransport.handle_async_request`); a
`client.stream(...)` context manager therefore can't hand back control until
the entire SSE stream (background task included) has already finished
either, same problem as `TestClient`.

So this drives the ASGI app directly with a hand-rolled `receive`/`send`
pair, running `app(scope, receive, send)` as its own `asyncio.Task` on the
SAME event loop as the test - real, observable concurrency. Body chunks
arrive on an `asyncio.Queue` as `send()` is actually called from inside the
running app, so the test can read the first SSE chunk the moment it's
produced. A disconnect is then simulated the way Starlette's own
`StreamingResponse` actually detects one in production: `receive()` starts
returning `{"type": "http.disconnect"}`, which is exactly what
`StreamingResponse.listen_for_disconnect` polls for to cancel the response
body generator - not an external `task.cancel()`, which doesn't exercise the
real code path at all.

A gated LLM stream model (`mock_gated_streaming_llm_model`) pauses the graph
after its first token so the test can deterministically catch
`run_and_persist` still mid-flight (not racing wall-clock timing) at the
exact moment it simulates the disconnect - then release the gate and
confirm the background task still runs to completion and PATCHes Java
exactly once with the correct final status.
"""

import asyncio
import json
from collections.abc import Callable, Generator, Sequence
from typing import Any

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel
from starlette.types import Message, Scope

from app.api.deps import get_backend_java_client, get_graph_models
from app.core.config import settings
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.main import app
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService


class _JavaBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next_id = 1
        self.patched = asyncio.Event()

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
            # Signal AFTER recording the call, so a waiter unblocked by this
            # event is guaranteed to see it in `self.calls`.
            self.patched.set()
            return httpx.Response(200, json={"status": body.get("status")})
        raise AssertionError(f"unexpected call {request.method} {request.url.path}")


@pytest.fixture(autouse=True)
def _clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.pop(get_backend_java_client, None)
    app.dependency_overrides.pop(get_graph_models, None)


def _build_scope(*, body: bytes) -> Scope:
    headers = {
        "content-type": "application/json",
        "content-length": str(len(body)),
        "x-internal-secret": settings.INTERNAL_SECRET_KEY,
    }
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "scheme": "http",
        "path": "/api/v1/chat/stream",
        "raw_path": b"/api/v1/chat/stream",
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "root_path": "",
    }


class _StreamingAsgiDriver:
    """Drives one ASGI app call as a real, concurrent `asyncio.Task` -
    unlike `TestClient`/`httpx.ASGITransport`, `send()` messages are
    observable on `messages` as soon as the app actually produces them, and
    `disconnect()` simulates a dropped client connection the same way a real
    ASGI server would (`receive()` starts returning `http.disconnect`).
    """

    def __init__(self, body: bytes) -> None:
        self.messages: asyncio.Queue[Message] = asyncio.Queue()
        self._body = body
        self._request_sent = False
        self._disconnected = asyncio.Event()
        self.task = asyncio.create_task(app(_build_scope(body=body), self._receive, self._send))

    async def _receive(self) -> Message:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self._disconnected.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: Message) -> None:
        await self.messages.put(message)

    def disconnect(self) -> None:
        self._disconnected.set()

    async def next_body_chunk(self) -> bytes:
        while True:
            message = await self.messages.get()
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    return bytes(body)
            elif message["type"] == "http.response.start":
                assert message["status"] == 200, message


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_still_patches_completed(
    client: Any,  # fixture side effect: wires db_session/session_factory overrides
    mock_gated_streaming_llm_model: Callable[[Sequence[str], asyncio.Event], FunctionModel],
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.0)
    java = _JavaBackend()
    app.dependency_overrides[get_backend_java_client] = lambda: BackendJavaClient(
        base_url="http://java.test", transport=httpx.MockTransport(java.handler)
    )

    gate = asyncio.Event()
    dummy_chunk = RetrievedChunk(
        chunk_id="c1", content="dummy retrieved content", source="s", score=0.9
    )
    models = GraphModels(
        classification=mock_sync_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("hyde"),
        generation=mock_gated_streaming_llm_model(["token-1 ", "token-2 ", "token-3"], gate),
        retrieval=FakeRetrievalService([dummy_chunk]),
    )
    app.dependency_overrides[get_graph_models] = lambda: models

    body = json.dumps(
        {"conversation_id": "conv-disconnect", "message": "Điều kiện học bổng là gì?"}
    ).encode()
    driver = _StreamingAsgiDriver(body)

    # Deterministic, not a race: the first SSE chunk can only reach us after
    # the gated model has yielded its first token and is now blocked on
    # `gate.wait()` inside the (still-running) background `run_and_persist`
    # task - so at this point that task is provably still mid-flight.
    first_chunk = await asyncio.wait_for(driver.next_body_chunk(), timeout=5.0)
    assert first_chunk == b'event: token\ndata: "token-1 "\n\n'
    assert not driver.task.done()
    assert all(c["method"] != "PATCH" for c in java.calls)

    # Simulate the client disconnecting mid-stream, the same way a real ASGI
    # server reports one: `receive()` starts returning `http.disconnect`,
    # which is exactly what Starlette's `StreamingResponse.listen_for_disconnect`
    # polls for to cancel the response body generator. The background
    # `run_and_persist` task, scheduled independently via
    # `asyncio.create_task()` inside the endpoint, must NOT be affected by
    # this.
    driver.disconnect()

    # (a) the client-side read is interrupted/incomplete: the ASGI call
    # finishes (the generator gets cancelled) WITHOUT ever sending the
    # `more_body=False` terminator carrying "event: done".
    await asyncio.wait_for(driver.task, timeout=5.0)
    remaining_bodies = []
    while not driver.messages.empty():
        message = driver.messages.get_nowait()
        if message["type"] == "http.response.body" and message.get("body"):
            remaining_bodies.append(bytes(message["body"]))
    assert b"event: done" not in b"".join(remaining_bodies)

    # Let the graph (and so the background task) finish.
    gate.set()

    # (b) the background task still completes and Java still receives
    # exactly one PATCH with the correct final status - waited on via an
    # explicit signal with a real deadline, not a sleep-and-hope poll.
    await asyncio.wait_for(java.patched.wait(), timeout=5.0)

    patches = [c for c in java.calls if c["method"] == "PATCH"]
    assert len(patches) == 1
    assert patches[0]["body"]["status"] == "COMPLETED"
    assert patches[0]["body"]["content"] == "token-1 token-2 token-3"
