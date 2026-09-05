"""T0.4: `BackendJavaClient` against `httpx.MockTransport` — no live Java, no
real network call anywhere in this file."""

from typing import Any

import httpx
import pytest

from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
)


def _client_with(handler: Any) -> BackendJavaClient:
    return BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_create_message_forwards_authorization_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(201, json={"id": "msg-1", "status": "COMPLETED"})

    client = _client_with(handler)

    result = await client.create_message(
        conversation_id="conv-1",
        role="USER",
        content="hello",
        authorization="Bearer abc.def.ghi",
    )

    assert seen["authorization"] == "Bearer abc.def.ghi"
    assert result == {"id": "msg-1", "status": "COMPLETED"}


@pytest.mark.asyncio
async def test_create_message_omits_authorization_header_for_guest() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_auth_header"] = "authorization" in request.headers
        return httpx.Response(201, json={"id": "msg-1"})

    client = _client_with(handler)

    await client.create_message(
        conversation_id="conv-1", role="USER", content="hello", authorization=None
    )

    assert seen["has_auth_header"] is False


@pytest.mark.asyncio
async def test_create_message_raises_http_error_with_status_on_ownership_rejection() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "not your conversation"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaHTTPError) as exc_info:
        await client.create_message(
            conversation_id="someone-elses-conv",
            role="USER",
            content="hello",
            authorization="Bearer token",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_update_message_patches_with_expected_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["method"] = request.method
        seen["url_path"] = request.url.path
        seen["parsed_body"] = _json.loads(request.read())
        return httpx.Response(200, json={"id": "msg-2", "status": "COMPLETED"})

    client = _client_with(handler)

    result = await client.update_message(
        message_id="msg-2",
        conversation_id="conv-1",
        content="full answer text",
        status="COMPLETED",
        citations=[{"chunk_id": "c1"}],
        authorization="Bearer token",
    )

    assert seen["method"] == "PATCH"
    assert seen["url_path"] == "/messages/msg-2"
    assert seen["parsed_body"]["conversationId"] == "conv-1"
    assert seen["parsed_body"]["content"] == "full answer text"
    assert seen["parsed_body"]["status"] == "COMPLETED"
    assert seen["parsed_body"]["citations"] == [{"chunk_id": "c1"}]
    assert result["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_get_conversation_messages_sends_limit_param_when_given() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[{"id": "m1"}, {"id": "m2"}])

    client = _client_with(handler)

    result = await client.get_conversation_messages(
        conversation_id="conv-1", limit=20, authorization="Bearer token"
    )

    assert seen["params"] == {"limit": "20"}
    assert result == [{"id": "m1"}, {"id": "m2"}]


@pytest.mark.asyncio
async def test_get_conversation_messages_empty_history_returns_empty_list() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client_with(handler)

    result = await client.get_conversation_messages(
        conversation_id="new-conv", authorization="Bearer token"
    )

    assert result == []


@pytest.mark.asyncio
async def test_network_failure_raises_connection_error_not_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client_with(handler)

    with pytest.raises(BackendJavaConnectionError):
        await client.get_conversation_messages(conversation_id="conv-1", authorization=None)


@pytest.mark.asyncio
async def test_create_conversation_forwards_authorization() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(201, json={"id": "conv-99"})

    client = _client_with(handler)

    result = await client.create_conversation(title="New chat", authorization="Bearer xyz")

    assert seen["authorization"] == "Bearer xyz"
    assert result == {"id": "conv-99"}
