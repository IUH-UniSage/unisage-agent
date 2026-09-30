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
async def test_get_conversation_messages_context_sends_context_flag_only() -> None:
    """Prompt history asks Java for `context=true` and no `limit`, so the admin
    setting `chat.max_history_messages` alone decides the window (UNISAGE-94)."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = _client_with(handler)

    await client.get_conversation_messages(conversation_id="conv-1", context=True)

    assert seen["params"] == {"context": "true"}


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
        if request.url.path.endswith("/verifications/claim"):
            return httpx.Response(200, json=[])
        if "/verifications/" in request.url.path and request.url.path.endswith("/result"):
            return httpx.Response(200, json={"applied": True, "duplicate": False})
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
    await client.report_health(
        credential_id="cred-1",
        credential_revision=1,
        snapshot_version=1,
        error_type="TRANSIENT",
        error_code="x",
        message="x",
        occurred_at="2026-09-26T00:00:00+00:00",
    )
    await client.get_embedding_index_identity(collection="unisage_chunks")
    await client.put_embedding_index_identity(
        collection="unisage_chunks",
        provider="openai",
        model_name="text-embedding-3-small",
        model_source_ref=None,
        api_base_url="https://api.openai.com/v1",
        dimension=2,
        fingerprint=[0.1, 0.2, 0.3, 0.4],
        established_by="bootstrap-cli",
    )
    await client.claim_verifications(limit=5)
    await client.post_verification_result(job_id="job-1", lease_token="lease-1", result_type="OK")

    assert seen == [settings.APP_INTERNAL_SECRET_KEY] * 10


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
async def test_report_health_posts_expected_body_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["method"] = request.method
        seen["url_path"] = request.url.path
        seen["authorization"] = request.headers.get("authorization")
        seen["parsed_body"] = _json.loads(request.read())
        return httpx.Response(200, json={"applied": True})

    client = _client_with(handler)

    result = await client.report_health(
        credential_id="cred-1",
        credential_revision=3,
        snapshot_version=42,
        error_type="TRANSIENT",
        error_code="RateLimitError:429",
        message="rate limited",
        occurred_at="2026-09-26T00:00:00+00:00",
    )

    assert seen["method"] == "POST"
    assert seen["url_path"] == "/internal/model-registry/credentials/cred-1/health"
    assert seen["authorization"] is None
    assert seen["parsed_body"] == {
        "credentialRevision": 3,
        "snapshotVersion": 42,
        "errorType": "TRANSIENT",
        "errorCode": "RateLimitError:429",
        "message": "rate limited",
        "occurredAt": "2026-09-26T00:00:00+00:00",
    }
    assert result == {"applied": True}


@pytest.mark.asyncio
async def test_get_embedding_index_identity_returns_none_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _client_with(handler)

    assert await client.get_embedding_index_identity(collection="unisage_chunks") is None


@pytest.mark.asyncio
async def test_get_embedding_index_identity_returns_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            request.url.path == "/internal/model-registry/embedding-index/unisage_chunks/identity"
        )
        assert request.headers.get("authorization") is None
        return httpx.Response(
            200,
            json={
                "provider": "openai",
                "modelName": "text-embedding-3-small",
                "modelSourceRef": None,
                "apiBaseUrl": "https://api.openai.com/v1",
                "dimension": 2,
                "fingerprint": [0.1, 0.2, 0.3, 0.4],
            },
        )

    client = _client_with(handler)

    result = await client.get_embedding_index_identity(collection="unisage_chunks")

    assert result == {
        "provider": "openai",
        "modelName": "text-embedding-3-small",
        "modelSourceRef": None,
        "apiBaseUrl": "https://api.openai.com/v1",
        "dimension": 2,
        "fingerprint": [0.1, 0.2, 0.3, 0.4],
    }


@pytest.mark.asyncio
async def test_put_embedding_index_identity_posts_expected_body_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["method"] = request.method
        seen["url_path"] = request.url.path
        seen["parsed_body"] = _json.loads(request.read())
        return httpx.Response(201, json={"established": True})

    client = _client_with(handler)

    result = await client.put_embedding_index_identity(
        collection="unisage_chunks",
        provider="openai",
        model_name="text-embedding-3-small",
        model_source_ref=None,
        api_base_url="https://api.openai.com/v1",
        dimension=2,
        fingerprint=[0.1, 0.2, 0.3, 0.4],
        established_by="first-upsert",
    )

    assert seen["method"] == "PUT"
    assert seen["url_path"] == "/internal/model-registry/embedding-index/unisage_chunks/identity"
    assert seen["parsed_body"] == {
        "provider": "openai",
        "modelName": "text-embedding-3-small",
        "modelSourceRef": None,
        "apiBaseUrl": "https://api.openai.com/v1",
        "dimension": 2,
        "fingerprint": [0.1, 0.2, 0.3, 0.4],
        "establishedBy": "first-upsert",
    }
    assert result == {"established": True}


@pytest.mark.asyncio
async def test_put_embedding_index_identity_raises_http_error_on_409() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "EMBEDDING_INDEX_IDENTITY_EXISTS"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaHTTPError) as exc_info:
        await client.put_embedding_index_identity(
            collection="unisage_chunks",
            provider="openai",
            model_name="text-embedding-3-small",
            model_source_ref=None,
            api_base_url="https://api.openai.com/v1",
            dimension=2,
            fingerprint=[0.1, 0.2, 0.3, 0.4],
            established_by="bootstrap-cli",
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_get_model_registry_snapshot_returns_raw_payload() -> None:
    payload = {
        "version": 7,
        "generatedAt": "2026-09-25T03:00:00Z",
        "purposes": {
            "CHAT": [{"id": "c1", "apiKey": "sk-secret"}],
            "EMBEDDING": [],
            "EXTRACTION": [],
        },
        "embeddingIndexIdentity": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/internal/model-registry/snapshot")
        assert "authorization" not in request.headers
        return httpx.Response(200, json=payload)

    client = _client_with(handler)

    assert await client.get_model_registry_snapshot() == payload


@pytest.mark.asyncio
async def test_claim_verifications_sends_limit_param_and_returns_list() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url_path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json=[
                {
                    "jobId": "job-1",
                    "leaseToken": "lease-1",
                    "attempt": 1,
                    "leaseUntil": "2026-09-26T00:01:00",
                    "credential": {
                        "chatModelId": "cm-1",
                        "modelPurpose": "CHAT",
                        "sourceType": "CLOUD_API",
                        "provider": "openai",
                        "modelName": "gpt-4o-mini",
                        "modelSourceRef": None,
                        "apiBaseUrl": "https://api.openai.com/v1",
                        "apiKey": "sk-secret",
                        "maxRpm": 500,
                    },
                }
            ],
        )

    client = _client_with(handler)

    result = await client.claim_verifications(limit=5)

    assert seen["method"] == "POST"
    assert seen["url_path"] == "/internal/model-registry/verifications/claim"
    assert seen["params"] == {"limit": "5"}
    assert result == [
        {
            "jobId": "job-1",
            "leaseToken": "lease-1",
            "attempt": 1,
            "leaseUntil": "2026-09-26T00:01:00",
            "credential": {
                "chatModelId": "cm-1",
                "modelPurpose": "CHAT",
                "sourceType": "CLOUD_API",
                "provider": "openai",
                "modelName": "gpt-4o-mini",
                "modelSourceRef": None,
                "apiBaseUrl": "https://api.openai.com/v1",
                "apiKey": "sk-secret",
                "maxRpm": 500,
            },
        }
    ]


@pytest.mark.asyncio
async def test_claim_verifications_returns_empty_list_when_nothing_to_claim() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client_with(handler)

    assert await client.claim_verifications(limit=5) == []


@pytest.mark.asyncio
async def test_post_verification_result_posts_expected_body_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["method"] = request.method
        seen["url_path"] = request.url.path
        seen["parsed_body"] = _json.loads(request.read())
        return httpx.Response(200, json={"applied": True, "duplicate": False})

    client = _client_with(handler)

    result = await client.post_verification_result(
        job_id="job-1",
        lease_token="lease-1",
        result_type="PERMANENT",
        error_code="AuthenticationError:401",
        message="invalid api key",
    )

    assert seen["method"] == "POST"
    assert seen["url_path"] == "/internal/model-registry/verifications/job-1/result"
    assert seen["parsed_body"] == {
        "leaseToken": "lease-1",
        "resultType": "PERMANENT",
        "errorCode": "AuthenticationError:401",
        "message": "invalid api key",
    }
    assert result == {"applied": True, "duplicate": False}


@pytest.mark.asyncio
async def test_post_verification_result_includes_embedding_fields_when_given() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["parsed_body"] = _json.loads(request.read())
        return httpx.Response(200, json={"applied": True, "duplicate": False})

    client = _client_with(handler)

    await client.post_verification_result(
        job_id="job-2",
        lease_token="lease-2",
        result_type="OK",
        embedding_dimension=3,
        embedding_fingerprint=[0.1, 0.2, 0.3],
    )

    assert seen["parsed_body"] == {
        "leaseToken": "lease-2",
        "resultType": "OK",
        "embeddingDimension": 3,
        "embeddingFingerprint": [0.1, 0.2, 0.3],
    }


@pytest.mark.asyncio
async def test_post_verification_result_raises_http_error_on_409_lease_lost() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "VERIFICATION_LEASE_LOST"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaHTTPError) as exc_info:
        await client.post_verification_result(
            job_id="job-1", lease_token="stale-token", result_type="OK"
        )

    assert exc_info.value.status_code == 409


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


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_report_health_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.report_health(
            credential_id="cred-1",
            credential_revision=1,
            snapshot_version=1,
            error_type="PERMANENT",
            error_code="x",
            message="x",
            occurred_at="2026-09-26T00:00:00+00:00",
        )

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_get_embedding_index_identity_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.get_embedding_index_identity(collection="unisage_chunks")

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_put_embedding_index_identity_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.put_embedding_index_identity(
            collection="unisage_chunks",
            provider="openai",
            model_name="text-embedding-3-small",
            model_source_ref=None,
            api_base_url="https://api.openai.com/v1",
            dimension=2,
            fingerprint=[0.1, 0.2, 0.3, 0.4],
            established_by="bootstrap-cli",
        )

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_claim_verifications_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.claim_verifications(limit=5)

    assert call_count == 1


@pytest.mark.parametrize("status_code", _REDIRECT_STATUSES)
@pytest.mark.asyncio
async def test_post_verification_result_rejects_redirect(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, headers={"Location": "https://evil.test/steal"})

    client = _client_with(handler)

    with pytest.raises(BackendJavaRedirectError):
        await client.post_verification_result(
            job_id="job-1", lease_token="lease-1", result_type="OK"
        )

    assert call_count == 1


def test_client_never_follows_redirects_by_default() -> None:
    client = BackendJavaClient(base_url="http://java.test")
    assert client._client().follow_redirects is False


@pytest.mark.asyncio
async def test_get_period_totals_sends_period_and_period_key_params() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"period": "DAILY", "periodKey": "2026-09-28", "totals": {"SYSTEM": 1500}}
        )

    client = _client_with(handler)

    result = await client.get_period_totals(period="DAILY", period_key="2026-09-28")

    assert seen["params"] == {"period": "DAILY", "periodKey": "2026-09-28"}
    assert result["totals"] == {"SYSTEM": 1500}
