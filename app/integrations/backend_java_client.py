"""HTTP client for `backend-java`'s Conversation/Message API.

`unisage-agent` does not own conversation/message persistence — `backend-java`
does. Every call here forwards the caller's original `Authorization` header
verbatim (or omits it for a guest/`KHACH` caller) so Java's
`GatewayHeaderFilter` can re-verify the JWT and enforce ownership itself;
this client never sends a separate service secret.

This client bypasses the API Gateway and talks to `backend-java` directly
(`settings.BACKEND_JAVA_BASE_URL`), so it also always sends
`X-Internal-Secret` (the same shared-secret gate this service itself
enforces on its own inbound endpoints via `app/core/security.py`) - Java
uses this to (a) authenticate the caller as `unisage-agent` itself, and (b)
decide whether to trust an accompanying `X-Guest-Session-Token` header for
the guest-conversation ownership check, since Python is the one calling
Java here instead of the browser directly (see `guest_session_token` on
`create_message`).

This client is tested entirely with `httpx.MockTransport`, never a live
Java instance.
"""

from typing import Any, Literal

import httpx

from app.core.config import settings

MessageRole = Literal["USER", "ASSISTANT"]
MessageStatus = Literal["PENDING", "STREAMING", "COMPLETED", "ERROR"]


class BackendJavaError(Exception):
    """Base class for all `backend-java` integration failures."""


class BackendJavaHTTPError(BackendJavaError):
    """Java responded with a non-2xx status.

    Carries the raw status code and parsed (or raw text) body so callers can
    make routing decisions on it — e.g. the streaming endpoint must treat
    404/403 from `POST /messages` as "do not run the graph", not as a
    generic 500.
    """

    def __init__(self, method: str, url: str, status_code: int, body: Any) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body
        super().__init__(f"backend-java {method} {url} -> HTTP {status_code}: {body!r}")


class BackendJavaConnectionError(BackendJavaError):
    """Network-level failure calling `backend-java` (timeout, connection refused, DNS, ...)."""

    def __init__(self, method: str, url: str, cause: Exception) -> None:
        self.method = method
        self.url = url
        self.cause = cause
        super().__init__(f"backend-java {method} {url} -> network error: {cause}")


class BackendJavaRedirectError(BackendJavaError):
    """Java answered with a 3xx — never followed, these requests carry secrets."""

    def __init__(self, method: str, url: str, status_code: int) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        super().__init__(f"backend-java {method} {url} -> unexpected redirect (HTTP {status_code})")


def _auth_headers(
    authorization: str | None,
    guest_session_token: str | None = None,
) -> dict[str, str]:
    """Build the header dict to forward for one call.

    Absent/empty `authorization` means the caller is a guest (`KHACH`) —
    deliberately sends no `Authorization` header at all rather than an empty
    one, matching how the gateway itself behaves for unauthenticated
    requests.

    `X-Internal-Secret` is always sent (this client talks to backend-java
    directly, bypassing the API Gateway). `X-Guest-Session-Token` is sent
    only when `guest_session_token` is given - this is how Java's
    guest-conversation ownership check works when this Python service is the
    caller instead of the browser directly: the browser's `guest_session_id`
    httpOnly cookie never reaches us as a forwardable cookie jar entry, so
    the caller reads it off the inbound request and passes it through
    explicitly (see `_resolve_guest_session_token` in `app/api/v1/chat.py`).
    Only honored by Java together with a valid `X-Internal-Secret`, so it's
    safe to always include once we're already sending the secret.
    """

    headers: dict[str, str] = {"X-Internal-Secret": settings.APP_INTERNAL_SECRET_KEY}
    if authorization:
        headers["Authorization"] = authorization
    if guest_session_token:
        headers["X-Guest-Session-Token"] = guest_session_token
    return headers


class BackendJavaClient:
    """Thin async wrapper around `backend-java`'s Conversation/Message REST API.

    Pass `transport=httpx.MockTransport(...)` in tests to avoid any real
    network call; production code leaves it `None` and gets a real
    `httpx.AsyncClient` pointed at `settings.BACKEND_JAVA_BASE_URL`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = (base_url or settings.BACKEND_JAVA_BASE_URL).rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            transport=self._transport,
            timeout=self._timeout,
            follow_redirects=False,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        authorization: str | None,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        guest_session_token: str | None = None,
    ) -> Any:
        headers = _auth_headers(authorization, guest_session_token)
        try:
            async with self._client() as client:
                response = await client.request(
                    method, path, headers=headers, json=json_body, params=params
                )
        except httpx.RequestError as exc:
            raise BackendJavaConnectionError(method, path, exc) from exc

        if response.is_redirect:
            raise BackendJavaRedirectError(method, path, response.status_code)

        if response.status_code >= 400:
            try:
                body: Any = response.json()
            except ValueError:
                body = response.text
            raise BackendJavaHTTPError(method, path, response.status_code, body)

        # An empty 200 body is a valid response, not an error - Java may
        # return one for an idempotent no-op (e.g. a PATCH that matches the
        # message's already-final state). Callers must not blindly
        # `dict(...)` this without checking for `None` first (see
        # `update_message`, which used to do exactly that and raised a bare
        # `TypeError` on this exact case).
        if not response.content:
            return None
        # Java wraps every response as {code, message, data} (see
        # backend-java's ApiResponse<T>) - unwrap to the payload callers
        # actually want.
        payload = response.json()
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    async def create_conversation(
        self,
        *,
        title: str | None = None,
        authorization: str | None = None,
    ) -> dict[str, Any]:
        """`POST /conversations`."""

        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        result = await self._request(
            "POST", "/conversations", authorization=authorization, json_body=body
        )
        return dict(result) if result is not None else {}

    async def create_message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
        status: MessageStatus = "COMPLETED",
        authorization: str | None = None,
        guest_session_token: str | None = None,
    ) -> dict[str, Any]:
        """`POST /messages`.

        Java validates `conversation_id` ownership on this call — a 404/403
        here means "do not run the graph, do not create a placeholder",
        which callers detect via `BackendJavaHTTPError.status_code`.

        `guest_session_token`, when given, is forwarded as
        `X-Guest-Session-Token` - this is how Java's guest-conversation
        ownership check works when this Python service is the caller instead
        of the browser directly (see `_auth_headers`). Only meaningful
        together with a valid `X-Internal-Secret`, which `_request` always
        sends.
        """

        body = {
            "conversationId": conversation_id,
            "role": role,
            "content": content,
            "status": status,
        }
        result = await self._request(
            "POST",
            "/messages",
            authorization=authorization,
            json_body=body,
            guest_session_token=guest_session_token,
        )
        return dict(result) if result is not None else {}

    async def update_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        content: str,
        status: Literal["COMPLETED", "ERROR"],
        citations: list[dict[str, Any]] | None = None,
        retrieval_score: float | None = None,
        metadata: dict[str, Any] | None = None,
        authorization: str | None = None,
    ) -> dict[str, Any]:
        """`PATCH /messages/{id}` — finalizes a `STREAMING` assistant message.

        Contract: only the message's own `conversation_id`, only
        `role=ASSISTANT`, only `STREAMING -> COMPLETED|ERROR`, idempotent on
        identical payload. Enforced entirely by Java; this client just
        shapes the request.

        Java requires `X-Internal-Secret` on this call (`_request` always
        sends it). An idempotent no-op PATCH (identical payload to the
        message's current state) may come back as `200` with an EMPTY body -
        that's a valid response, not an error, so this returns `{}` for it
        rather than raising (previously did a bare `dict(result)` here,
        which raised `TypeError: 'NoneType' object is not a mapping` on
        exactly this case since `_request` returns `None` for an empty
        body).
        """

        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "content": content,
            "status": status,
        }
        if citations is not None:
            body["citations"] = citations
        if retrieval_score is not None:
            body["retrievalScore"] = retrieval_score
        if metadata is not None:
            body["metadata"] = metadata

        result = await self._request(
            "PATCH",
            f"/messages/{message_id}",
            authorization=authorization,
            json_body=body,
        )
        return dict(result) if result is not None else {}

    async def get_model_registry_version(self) -> int:
        """`GET /internal/model-registry/version` (plan.md "Internal API contract" endpoint #2).

        No secret in the response - used for the periodic hot-reload poll (Task 7, out of
        scope here). No `Authorization` is sent: `/internal/**` grants on
        `X-Internal-Secret` + caller IP alone, never on a JWT.
        """

        result = await self._request("GET", "/internal/model-registry/version", authorization=None)
        return int(result["version"]) if result else 0

    async def get_model_registry_snapshot(self) -> dict[str, Any]:
        """`GET /internal/model-registry/snapshot` (plan.md "Internal API contract" endpoint #1) -
        the only source of provider credentials for this service once the registry is enabled
        (plan.md "Cutover khỏi cấu hình .env tĩnh"). The response carries plaintext API keys -
        callers must parse it into `app.core.model_registry.ModelRegistrySnapshot` immediately
        and never log or repr the raw dict this returns.
        """

        result = await self._request("GET", "/internal/model-registry/snapshot", authorization=None)
        return dict(result) if result else {}

    async def report_health(
        self,
        *,
        credential_id: str,
        credential_revision: int,
        snapshot_version: int,
        error_type: Literal["TRANSIENT", "PERMANENT"],
        error_code: str,
        message: str,
        occurred_at: str,
    ) -> dict[str, Any]:
        """`POST /internal/model-registry/credentials/{id}/health` (plan.md "Internal
        API contract" endpoint #3) - called by `app.core.model_router` after a
        provider-call failure. `message` must already be redacted
        (`app.core.redaction.safe_error_message`) before it reaches this method; this
        client does not redact anything itself.

        No secret in the response. No `Authorization` is sent, matching every other
        `/internal/**` call this client makes.
        """

        body = {
            "credentialRevision": credential_revision,
            "snapshotVersion": snapshot_version,
            "errorType": error_type,
            "errorCode": error_code,
            "message": message,
            "occurredAt": occurred_at,
        }
        result = await self._request(
            "POST",
            f"/internal/model-registry/credentials/{credential_id}/health",
            authorization=None,
            json_body=body,
        )
        return dict(result) if result else {}

    async def get_embedding_index_identity(self, *, collection: str) -> dict[str, Any] | None:
        """`GET /internal/model-registry/embedding-index/{collection}/identity` (plan.md
        "Internal API contract" endpoint #6) - the identity currently registered for `collection`,
        or `None` if none has been established yet (Java answers with an empty 404 body for that
        case, which this unwraps into `None` rather than raising).

        No secret in the response - no `Authorization` is sent, matching every other
        `/internal/**` call this client makes.
        """

        try:
            result = await self._request(
                "GET",
                f"/internal/model-registry/embedding-index/{collection}/identity",
                authorization=None,
            )
        except BackendJavaHTTPError as exc:
            if exc.status_code == 404:
                return None
            raise
        return dict(result) if result else None

    async def put_embedding_index_identity(
        self,
        *,
        collection: str,
        provider: str,
        model_name: str,
        model_source_ref: str | None,
        api_base_url: str | None,
        dimension: int,
        fingerprint: list[float],
        established_by: str,
    ) -> dict[str, Any]:
        """`PUT /internal/model-registry/embedding-index/{collection}/identity` (plan.md
        "Internal API contract" endpoint #7) - only-if-absent, called by the bootstrap CLI
        (`established_by="bootstrap-cli"`) or by the first ingest batch into an empty collection
        (`established_by="first-upsert"`).

        Raises `BackendJavaHTTPError` with `status_code == 409` when an identity already exists
        for `collection` - callers must then `get_embedding_index_identity` and compare rather
        than treat this as a generic failure (plan.md "Embedding identity guard": "đã có → 409
        `EMBEDDING_INDEX_IDENTITY_EXISTS`, không bao giờ ghi đè").
        """

        body = {
            "provider": provider,
            "modelName": model_name,
            "modelSourceRef": model_source_ref,
            "apiBaseUrl": api_base_url,
            "dimension": dimension,
            "fingerprint": fingerprint,
            "establishedBy": established_by,
        }
        result = await self._request(
            "PUT",
            f"/internal/model-registry/embedding-index/{collection}/identity",
            authorization=None,
            json_body=body,
        )
        return dict(result) if result else {}

    async def claim_verifications(self, *, limit: int) -> list[dict[str, Any]]:
        """`POST /internal/model-registry/verifications/claim?limit=N` (plan.md "Internal API
        contract" endpoint #4) - claims up to `limit` queued/lease-expired verification jobs,
        each with a freshly-minted `leaseToken`. The response carries each job's candidate
        credential in plaintext - callers must not log or repr the raw list this returns.

        No `Authorization` is sent, matching every other `/internal/**` call this client makes.
        """

        result = await self._request(
            "POST",
            "/internal/model-registry/verifications/claim",
            authorization=None,
            params={"limit": limit},
        )
        return list(result) if result else []

    async def post_verification_result(
        self,
        *,
        job_id: str,
        lease_token: str,
        result_type: Literal["OK", "TRANSIENT", "PERMANENT"],
        error_code: str | None = None,
        message: str | None = None,
        embedding_dimension: int | None = None,
        embedding_fingerprint: list[float] | None = None,
    ) -> dict[str, Any]:
        """`POST /internal/model-registry/verifications/{jobId}/result` (plan.md "Internal API
        contract" endpoint #5) - reports the outcome of trying `job_id`'s candidate credential.

        `lease_token` must be the exact token from the matching `claim_verifications` entry -
        Java rejects a stale/mismatched one with `409` (raised here as `BackendJavaHTTPError`
        with `status_code == 409`; callers must not retry that case, see plan.md "Verification
        lifecycle"). `message` must already be redacted (`app.core.redaction.safe_error_message`)
        before it reaches this method - this client does not redact anything itself.
        `embedding_dimension`/`embedding_fingerprint` are only meaningful for an EMBEDDING
        candidate's `OK` result.

        No `Authorization` is sent, matching every other `/internal/**` call this client makes.
        """

        body: dict[str, Any] = {"leaseToken": lease_token, "resultType": result_type}
        if error_code is not None:
            body["errorCode"] = error_code
        if message is not None:
            body["message"] = message
        if embedding_dimension is not None:
            body["embeddingDimension"] = embedding_dimension
        if embedding_fingerprint is not None:
            body["embeddingFingerprint"] = embedding_fingerprint

        result = await self._request(
            "POST",
            f"/internal/model-registry/verifications/{job_id}/result",
            authorization=None,
            json_body=body,
        )
        return dict(result) if result else {}

    async def get_conversation_messages(
        self,
        *,
        conversation_id: str,
        limit: int | None = None,
        context: bool = False,
        authorization: str | None = None,
    ) -> list[dict[str, Any]]:
        """`GET /messages/conversation/{id}?limit=N&context=true`.

        `context=True` asks for prompt context: Java caps it at the admin
        setting `chat.max_history_messages` (System Settings), so that
        setting - not a local env var - decides how much history the model
        sees. Without it Java returns the whole history (or the last `limit`).
        """

        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if context:
            params["context"] = "true"
        result = await self._request(
            "GET",
            f"/messages/conversation/{conversation_id}",
            authorization=authorization,
            params=params or None,
        )
        return list(result) if result is not None else []
