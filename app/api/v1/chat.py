import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_backend_java_client,
    get_graph_models,
    get_session_factory,
)
from app.core.config import settings
from app.core.exceptions import (
    BackendJavaUnavailableException,
    ConversationRejectedException,
    InvalidQueryException,
)
from app.core.sanitizer import sanitize_input_text
from app.core.security import verify_internal_secret
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import get_db_session
from app.graph.nodes.greeting import is_first_turn
from app.graph.nodes.security_context import parse_security_headers
from app.graph.streaming_session import run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
)
from app.schemas.chat import ChatStreamRequest
from app.schemas.chat_history import HistoryMessage
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_internal_secret)])

# Capped independently of backend-java's own app.message.max-history (default
# 20, used by GET /messages/conversation/{id} for the frontend's history
# list) - this is specifically how many prior turns get folded into the
# generation prompt as raw <history_message> context, not how many the UI
# shows. Configurable via HISTORY_MESSAGE_LIMIT (see app/core/config.py).
_HISTORY_MESSAGE_LIMIT = settings.HISTORY_MESSAGE_LIMIT


async def _load_history(
    java_client: BackendJavaClient,
    *,
    conversation_id: str,
    authorization: str | None,
) -> list[HistoryMessage]:
    """The last `_HISTORY_MESSAGE_LIMIT` messages BEFORE this turn (called
    prior to persisting this turn's own USER message, so the list never
    includes it - `user_query` already carries that separately). Drops any
    row that isn't a real, finished message (e.g. a STREAMING placeholder
    left behind by a previous turn's error) - raw conversational context is
    only useful if it reads as something the student or the assistant
    actually said."""

    raw_messages = await java_client.get_conversation_messages(
        conversation_id=conversation_id, limit=_HISTORY_MESSAGE_LIMIT, authorization=authorization
    )
    return [
        HistoryMessage(role=message["role"], content=message["content"])
        for message in raw_messages
        if message.get("status") == "COMPLETED" and message.get("content")
    ]


# asyncio.create_task() only holds a WEAK reference to the task it schedules
# per the stdlib's own docs - without keeping a strong reference somewhere,
# the task can be garbage-collected mid-run. This set is that reference; the
# done-callback discards it once the task (run_and_persist) finishes, so the
# set doesn't grow unbounded.
_background_tasks: set[asyncio.Task[None]] = set()


def _resolve_client_ip(http_request: Request, x_forwarded_for: str | None) -> str | None:
    """Best client IP available for this request, for trace/log correlation only.

    Threaded into `GraphTrace` (see `app/core/graph_trace.py`) so each log
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


async def _sse_token_generator(queue: "asyncio.Queue[str | None]") -> AsyncGenerator[str, None]:
    """Reads tokens from `queue` until the end-of-stream sentinel (`None`).

    Deliberately does nothing else - no graph execution, no Java calls. This
    is the piece Starlette cancels on client disconnect; `run_and_persist`,
    which does the real work, runs in an independent `asyncio.create_task()`
    and is never awaited here.
    """

    while True:
        token = await queue.get()
        if token is None:
            break
        yield f"event: token\ndata: {json.dumps(token, ensure_ascii=False)}\n\n"
    yield "event: done\ndata: {}\n\n"


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
    """SSE streaming chat endpoint.

    Ordering matters and is the whole point of this endpoint (conversation_id
    ownership must be verified before any state is created, and persistence
    must survive a client disconnect):

    1. Ask Java for this conversation's message count (first-turn detection)
       and its last `_HISTORY_MESSAGE_LIMIT` messages (raw `<history_message>`
       context for the prompt - see `_load_history`), both BEFORE this turn's
       own USER message exists.
    2. Call Java `POST /messages` (role=USER) FIRST, synchronously, still
       inside this request/response cycle. If Java rejects it (404/403 -
       conversation doesn't exist or belongs to someone else), this raises
       straight back to the client as an HTTP error - no graph run, no
       assistant placeholder, ever. Forwards the guest's `guest_session_id`
       cookie value we read off our own inbound request
       (`_resolve_guest_session_token`) as `X-Guest-Session-Token` alongside
       our `X-Internal-Secret` on every `BackendJavaClient` call in this
       request, so Java can run its guest-conversation ownership check even
       though it's us calling, not the browser directly.
    3. Only once that succeeds: create the ASSISTANT `STREAMING` placeholder.
    4. Load this conversation's clarification state.
    5. Schedule `run_and_persist` as an independent `asyncio.create_task()` -
       NOT awaited here - and return a `StreamingResponse` whose generator
       only reads the queue that task writes to.
    """

    clean_message = sanitize_input_text(request.message)
    if not clean_message:
        raise InvalidQueryException("Câu hỏi không được để trống hoặc không hợp lệ.")

    client_ip = _resolve_client_ip(http_request, x_forwarded_for)
    guest_session_token = _resolve_guest_session_token(http_request)

    first_turn = await is_first_turn(
        java_client, conversation_id=request.conversation_id, authorization=authorization
    )
    # Fetched BEFORE this turn's own USER message is persisted below, so it
    # never includes it (see `_load_history`).
    history = await _load_history(
        java_client, conversation_id=request.conversation_id, authorization=authorization
    )

    try:
        await java_client.create_message(
            conversation_id=request.conversation_id,
            role="USER",
            content=clean_message,
            status="COMPLETED",
            authorization=authorization,
            guest_session_token=guest_session_token,
        )
    except BackendJavaHTTPError as exc:
        raise ConversationRejectedException(exc.status_code) from exc
    except BackendJavaConnectionError as exc:
        raise BackendJavaUnavailableException() from exc

    try:
        assistant_message = await java_client.create_message(
            conversation_id=request.conversation_id,
            role="ASSISTANT",
            content="",
            status="STREAMING",
            authorization=authorization,
            guest_session_token=guest_session_token,
        )
    except (BackendJavaHTTPError, BackendJavaConnectionError) as exc:
        raise BackendJavaUnavailableException() from exc

    assistant_message_id = str(assistant_message["id"])

    clarification_repo = ClarificationStateRepository(db_session)
    pending_clarification = await clarification_repo.get_pending_clarification(
        request.conversation_id
    )
    confirmed_metadata = await clarification_repo.get_confirmed_metadata(request.conversation_id)

    graph_input = GraphInput(
        conversation_id=request.conversation_id,
        user_message=clean_message,
        is_first_turn=first_turn,
        security=security,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending_clarification,
        clarification_max_retry=settings.CLARIFICATION_MAX_RETRY,
        history=history,
    )

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(
        run_and_persist(
            java_client=java_client,
            conversation_id=request.conversation_id,
            assistant_message_id=assistant_message_id,
            authorization=authorization,
            client_ip=client_ip,
            graph_input=graph_input,
            models=models,
            queue=queue,
            session_factory=session_factory,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(_sse_token_generator(queue), media_type="text/event-stream")
