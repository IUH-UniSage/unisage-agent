"""`POST /chat/stream` controller: read the HTTP request, hand it to
`ChatStreamService`, return its SSE stream. No business logic lives here."""

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_backend_java_client,
    get_graph_models,
    get_session_factory,
)
from app.core.security.security import verify_internal_secret
from app.database.session import get_db_session
from app.graph.nodes.security_context import parse_security_headers
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.schemas.chat import ChatStreamRequest
from app.schemas.security import AcademicSecurityContext
from app.services.chat_stream_service import ChatCaller, ChatStreamService

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_internal_secret)])


def _resolve_client_ip(http_request: Request, x_forwarded_for: str | None) -> str | None:
    """Best client IP available for this request, for trace/log correlation only.

    Threaded into `GraphTrace` (see `app/core/observability/graph_trace.py`) so each log
    line can be tied back to a caller - it is NOT sent to backend-java.
    Guest-conversation ownership there is now checked via
    `X-Guest-Session-Token` (see `_resolve_guest_session_token` below), not
    IP matching.

    Prefers an `X-Forwarded-For` the API Gateway may already have set on
    this inbound request (first IP in the list is the original client, per
    the header's usual left-to-right convention) over
    `request.client.host`, since the latter would just be the gateway's own
    address once traffic passes through it - not the real caller. Falls
    back to `request.client.host` when no such header is present (e.g. the
    gateway is not between the caller and this service in some deployment).
    Appending our own hop is unnecessary here - we're just forwarding the
    best client IP we already have, not extending the chain.
    """

    if x_forwarded_for:
        first_ip = x_forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    if http_request.client is not None:
        return http_request.client.host
    return None


GUEST_SESSION_COOKIE_NAME = "guest_session_id"


def _resolve_guest_session_token(http_request: Request) -> str | None:
    """The guest's `guest_session_id` httpOnly cookie, read off our own inbound request.

    Java trusts an `X-Guest-Session-Token` header from us only together with
    a valid `X-Internal-Secret` (see `BackendJavaClient._auth_headers`) -
    this is how the guest-conversation ownership check works when this
    service calls Java directly instead of the browser doing it. We have no
    cookie jar of our own to forward the cookie as-is, so we read it off the
    request we received (the frontend calls this endpoint with
    `credentials: include`, same as it does for backend-java, so the cookie
    reaches us intact) and pass the raw value through explicitly.
    """

    return http_request.cookies.get(GUEST_SESSION_COOKIE_NAME)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatStreamRequest,
    http_request: Request,
    security: AcademicSecurityContext = Depends(parse_security_headers),
    authorization: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
    db_session: AsyncSession = Depends(get_db_session),
    java_client: BackendJavaClient = Depends(get_backend_java_client),
    models: GraphModels = Depends(get_graph_models),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> StreamingResponse:
    """SSE streaming chat turn - see `ChatStreamService` for the ordering guarantees."""

    service = ChatStreamService(
        db_session=db_session,
        java_client=java_client,
        models=models,
        session_factory=session_factory,
    )
    caller = ChatCaller(
        security=security,
        authorization=authorization,
        client_ip=_resolve_client_ip(http_request, x_forwarded_for),
        guest_session_token=_resolve_guest_session_token(http_request),
    )
    stream = await service.open_stream(request, caller, body_size=len(await http_request.body()))
    return StreamingResponse(stream, media_type="text/event-stream")
