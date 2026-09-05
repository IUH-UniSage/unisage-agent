"""HTTP client for `backend-java`'s Conversation/Message API (T0.4).

`unisage-agent` does not own conversation/message persistence — `backend-java`
does (see tasks/plan.md). Every call here forwards the caller's original
`Authorization` header verbatim (or omits it for a guest/`KHACH` caller) so
Java's `GatewayHeaderFilter` can re-verify the JWT and enforce ownership
itself; this client never sends a separate service secret.

`PATCH /messages/{id}` and the `limit` param on
`GET /messages/conversation/{id}` are being added to `backend-java` in
parallel (see tasks/plan.md checkpoint notes) — this client is written
against the contract plan.md describes and is tested entirely with
`httpx.MockTransport`, never a live Java instance.
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
    make routing decisions on it — e.g. T1.13c must treat 404/403 from
    `POST /messages` as "do not run the graph", not as a generic 500.
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


def _auth_headers(authorization: str | None) -> dict[str, str]:
    """Build the header dict to forward for one call.

    Absent/empty `authorization` means the caller is a guest (`KHACH`) —
    deliberately sends no `Authorization` header at all rather than an empty
    one, matching how the gateway itself behaves for unauthenticated
    requests (see tasks/plan.md "Auth" section).
    """

    return {"Authorization": authorization} if authorization else {}


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
            base_url=self._base_url, transport=self._transport, timeout=self._timeout
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        authorization: str | None,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        headers = _auth_headers(authorization)
        try:
            async with self._client() as client:
                response = await client.request(
                    method, path, headers=headers, json=json_body, params=params
                )
        except httpx.RequestError as exc:
            raise BackendJavaConnectionError(method, path, exc) from exc

        if response.status_code >= 400:
            try:
                body: Any = response.json()
            except ValueError:
                body = response.text
            raise BackendJavaHTTPError(method, path, response.status_code, body)

        if not response.content:
            return None
        return response.json()

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
        return dict(result)

    async def create_message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
        status: MessageStatus = "COMPLETED",
        authorization: str | None = None,
    ) -> dict[str, Any]:
        """`POST /messages`.

        Java validates `conversation_id` ownership on this call (see
        tasks/plan.md invariant) — a 404/403 here means "do not run the
        graph, do not create a placeholder", which callers detect via
        `BackendJavaHTTPError.status_code`.
        """

        body = {
            "conversationId": conversation_id,
            "role": role,
            "content": content,
            "status": status,
        }
        result = await self._request(
            "POST", "/messages", authorization=authorization, json_body=body
        )
        return dict(result)

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

        Contract (see tasks/plan.md): only the message's own
        `conversation_id`, only `role=ASSISTANT`, only
        `STREAMING -> COMPLETED|ERROR`, idempotent on identical payload.
        Enforced entirely by Java; this client just shapes the request.
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
            "PATCH", f"/messages/{message_id}", authorization=authorization, json_body=body
        )
        return dict(result)

    async def get_conversation_messages(
        self,
        *,
        conversation_id: str,
        limit: int | None = None,
        authorization: str | None = None,
    ) -> list[dict[str, Any]]:
        """`GET /messages/conversation/{id}?limit=N`.

        `limit` is optional and, per plan.md, always capped server-side by
        Java's own `MAX_MESSAGE_HISTORY` regardless of what's requested here.
        """

        params = {"limit": limit} if limit is not None else None
        result = await self._request(
            "GET",
            f"/messages/conversation/{conversation_id}",
            authorization=authorization,
            params=params,
        )
        return list(result) if result is not None else []
