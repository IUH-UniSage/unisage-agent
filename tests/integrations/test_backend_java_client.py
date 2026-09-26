"""`BackendJavaClient` against `httpx.MockTransport` — no live Java, no
real network call anywhere in this file."""

from typing import Any

import httpx
import pytest

from app.core.config import settings
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
    BackendJavaRedirectError,
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
async def test_every_call_sends_x_internal_secret_header() -> None:
    """Bug 1: Java now requires `X-Internal-Secret` on internal-only calls
    (e.g. `PATCH /messages/{id}`) - this client bypasses the API Gateway, so
    it must send this itself on every call, not just forward the caller's
    `Authorization`."""

    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-internal-secret"))
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"version": 1})
        if request.url.path.endswith("/snapshot"):
            return httpx.Response(200, json={"version": 1, "purposes": {}})
        return httpx.Response(200, json={"id": "msg-1", "status": "COMPLETED"})

    client = _client_with(handler)

    await client.create_message(
        conversation_id="conv-1", role="USER", content="hello", authorization="Bearer token"
    )
    await client.update_message(
        message_id="msg-1", conversation_id="conv-1", content="answer", status="COMPLETED"
    )
    await client.get_conversation_messages(conversation_id="conv-1", authorization=None)
    await client.get_model_registry_version()
    await client.get_model_registry_snapshot()

    assert seen == [settings.APP_INTERNAL_SECRET_KEY] * 5


@pytest.mark.asyncio
async def test_create_message_never_sends_x_forwarded_for() -> None:
    """Guest-conversation ownership moved from IP-matching to
    `X-Guest-Session-Token` - this client no longer accepts or forwards a
    client IP to Java at all."""

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_header"] = "x-forwarded-for" in request.headers
        return httpx.Response(201, json={"id": "msg-1"})

    client = _client_with(handler)

    await client.create_message(conversation_id="conv-1", role="USER", content="hello")

    assert seen["has_header"] is False


@pytest.mark.asyncio
async def test_create_message_forwards_guest_session_token_as_header() -> None:
    """Java's guest-conversation ownership check is keyed on `guest_session_id`
    now (not IP) - forward the raw cookie value this service was given as
    `X-Guest-Session-Token` when it's the one calling `POST /messages`
    instead of the browser."""

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x_guest_session_token"] = request.headers.get("x-guest-session-token")
        return httpx.Response(201, json={"id": "msg-1", "status": "COMPLETED"})

    client = _client_with(handler)

    await client.create_message(
        conversation_id="conv-1",
        role="USER",
        content="hello",
        authorization=None,
        guest_session_token="raw-guest-token",
    )

    assert seen["x_guest_session_token"] == "raw-guest-token"


@pytest.mark.asyncio
async def test_create_message_omits_guest_session_header_when_no_token_given() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_header"] = "x-guest-session-token" in request.headers
        return httpx.Response(201, json={"id": "msg-1"})

    client = _client_with(handler)

    await client.create_message(conversation_id="conv-1", role="USER", content="hello")

    assert seen["has_header"] is False


@pytest.mark.asyncio
async def test_update_message_treats_empty_response_body_as_no_op() -> None:
    """Bug 2: Java may respond `200` with an EMPTY body for an idempotent
    no-op PATCH. `_request` returns `None` for an empty body; `update_message`
    used to unconditionally do `dict(result)`, raising a bare `TypeError` on
    exactly this case. That must not happen - an empty body is a valid
    response, not an error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = _client_with(handler)

    result = await client.update_message(
        message_id="msg-2", conversation_id="conv-1", content="full answer", status="COMPLETED"
    )

    assert result == {}


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


@pytest.mark.asyncio
async def test_unwraps_java_api_response_envelope() -> None:
    """backend-java's `ApiResponse<T>` wraps every 2xx body as
    `{code, message, data}` - real responses look like this, not the flat
    dicts other tests in this file use for brevity."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={"code": 1000, "message": "Successful", "data": {"id": "msg-1"}},
        )

    client = _client_with(handler)

    result = await client.create_message(conversation_id="conv-1", role="USER", content="hi")

    assert result == {"id": "msg-1"}


@pytest.mark.asyncio
async def test_get_model_registry_version_returns_int() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/internal/model-registry/version")
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"version": 42})

    client = _client_with(handler)

    assert await client.get_model_registry_version() == 42


@pytest.mark.asyncio
async def test_get_model_registry_snapshot_returns_raw_payload() -> None:
    payload = {
        "version": 7,
        "generatedAt": "2026-09-25T03:00:00Z",
        "purposes": {"CHAT": [{"id": "c1", "apiKey": "sk-secret"}], "EMBEDDING": [], "EXTRACTION": []},
        "embeddingIndexIdentity": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/internal/model-registry/snapshot")
        assert "authorization" not in request.headers
        return httpx.Response(200, json=payload)

    client = _client_with(handler)

    assert await client.get_model_registry_snapshot() == payload


# ── never follow a redirect from Java — every method that hits it ──────────

_REDIRECT_STATUSES = [301, 302, 307, 308]


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_create_message_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.create_message(conversation_id="conv-1", role="USER", content="hi")

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_update_message_rejects_redirect(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.update_message(
            message_id="msg-1", conversation_id="conv-1", content="x", status="COMPLETED"
        )


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_create_conversation_rejects_redirect(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.create_conversation()


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_get_conversation_messages_rejects_redirect(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.get_conversation_messages(conversation_id="conv-1")


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_get_model_registry_snapshot_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.get_model_registry_snapshot()

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_get_model_registry_version_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.get_model_registry_version()

    assert call_count == 1


def test_client_never_follows_redirects_by_default() -> None:
    client = BackendJavaClient(base_url="http://java.test")
    assert client._client().follow_redirects is False
