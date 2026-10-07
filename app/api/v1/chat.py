import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_backend_java_client,
    get_graph_models,
    get_session_factory,
)
from app.core.budget.tracker import get_default_tracker
from app.core.config import settings
from app.core.errors.exceptions import (
    BackendJavaUnavailableException,
    ConversationRejectedException,
    InvalidQueryException,
    UsageLimitExceededException,
)
from app.core.security.sanitizer import detect_prompt_injection, sanitize_input_text
from app.core.security.security import verify_internal_secret
from app.core.usage.cost_calculator import estimate as estimate_cost
from app.core.usage.usage_recorder import UsageRecorder
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import get_db_session
from app.graph.nodes.security_context import parse_security_headers
from app.graph.queue_items import DoneItem, ErrorItem, QueueItem, TokenItem, WarningItem
from app.graph.streaming import BudgetContext
from app.graph.streaming_session import run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaError,
    BackendJavaHTTPError,
)
from app.schemas.chat import ChatStreamRequest
from app.schemas.chat_history import HistoryMessage
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_internal_secret)])


async def _start_turn(
    java_client: BackendJavaClient,
    *,
    conversation_id: str,
    content: str,
    authorization: str | None,
    guest_session_token: str | None,
) -> dict[str, Any]:
    try:
        return await java_client.start_turn(
            conversation_id=conversation_id,
            content=content,
            authorization=authorization,
            guest_session_token=guest_session_token,
        )
    except BackendJavaHTTPError as exc:
        if exc.status_code == 429:
            raise UsageLimitExceededException(_usage_limit_errors(exc.body)) from exc
        if exc.status_code in _CONVERSATION_REJECTED_STATUSES:
            raise ConversationRejectedException(exc.status_code) from exc
        # A Java 5xx/400 is not "this conversation isn't yours" - don't say it is.
        raise BackendJavaUnavailableException() from exc
    except BackendJavaError as exc:
        raise BackendJavaUnavailableException() from exc


def _history_from_context(raw_messages: list[dict[str, Any]]) -> list[HistoryMessage]:
    """Keeps only finished messages (drops e.g. a STREAMING placeholder left by a failed turn)."""

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

# Java statuses meaning "this conversation doesn't exist / isn't the caller's" - every other
# error status is Java failing, reported as such.
_CONVERSATION_REJECTED_STATUSES = (401, 403, 404)


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


def _estimate_chat_call_cost_usd(message: str, models: GraphModels) -> Decimal:
    """Upper-bound reservation estimate for ONE LLM call (used for every
    PROVIDER-scope acquire; `streaming_session.py` multiplies this up for the
    request-level SYSTEM/PURPOSE reservation, which must cover every attempt
    across every node, not just one).

    A rough char/4 heuristic stands in for a real tokenizer here - this is only
    ever an upper bound for a Redis reservation that gets settled to the real
    cost afterward, not a billing figure, so exactness doesn't matter as much as
    never under-reserving.
    """

    if models.generation_credential is None:
        return Decimal("0")
    input_tokens_estimate = max(1, len(message) // 4)
    return estimate_cost(
        provider=models.generation_credential.provider,
        model_name=models.generation_credential.model_name,
        source_type=models.generation_credential.source_type,
        input_tokens=input_tokens_estimate,
        max_output_tokens=settings.BUDGET_ESTIMATE_MAX_OUTPUT_TOKENS,
    )


def _usage_limit_errors(body: object) -> dict[str, str]:
    """The `window` / `resetAt` detail of Java's 429, or {} when the body has none."""

    errors = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(errors, dict):
        return {}
    return {
        key: value
        for key, value in errors.items()
        if key in ("window", "resetAt") and isinstance(value, str)
    }


async def _sse_token_generator(queue: "asyncio.Queue[QueueItem]") -> AsyncGenerator[str, None]:
    """Reads items from `queue` until the end-of-stream sentinel (`DoneItem`).

    Deliberately does nothing else - no graph execution, no Java calls. This
    is the piece Starlette cancels on client disconnect; `run_and_persist`,
    which does the real work, runs in an independent `asyncio.create_task()`
    and is never awaited here.

    `event: done` is always the last event emitted, no matter what was queued
    before it - the loop below only ever
    breaks on `DoneItem`, so every `TokenItem`/`ErrorItem` queued ahead of it
    is drained and emitted first. `run_and_persist` only ever queues at most
    one `ErrorItem`, immediately before its own `DoneItem` put, so `event:
    error` (when present) always immediately precedes `event: done` and no
    `event: token` can ever follow it.
    """

    while True:
        item = await queue.get()
        if isinstance(item, DoneItem):
            break
        if isinstance(item, TokenItem):
            yield f"event: token\ndata: {json.dumps(item.text, ensure_ascii=False)}\n\n"
        elif isinstance(item, ErrorItem):
            payload = {"code": item.code, "message": item.message, "retryable": item.retryable}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif isinstance(item, WarningItem):
            payload = {"code": item.code, "message": item.message}
            yield f"event: warning\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "event: done\ndata: {}\n\n"


def _log_suspected_prompt_injection(message: str, *, role: str, conversation_id: str) -> None:
    """Log-only: a match never blocks or changes the turn. The record carries
    the pattern name, role, conversation id and message length - never the
    message, the matched text, or the user id - so the false-positive rate can
    be measured before anyone decides to block."""

    pattern = detect_prompt_injection(message)
    if pattern is None:
        return
    logger.warning(
        "Prompt injection suspected: pattern=%s role=%s conversation_id=%s message_length=%d",
        pattern,
        role,
        conversation_id,
        len(message),
        extra={
            "pattern": pattern,
            "role": role,
            "conversation_id": conversation_id,
            "message_length": len(message),
        },
    )


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

    1. `POST /messages/turn`: Java checks ownership, returns first-turn +
       history, then stores the USER message and the ASSISTANT placeholder
       in one transaction. A 404/403/429 raises straight back to the client -
       no graph run, nothing persisted.
    2. Concurrently, load this conversation's clarification state.
    3. Schedule `run_and_persist` as an independent `asyncio.create_task()` -
       NOT awaited here - and return a `StreamingResponse` whose generator
       only reads the queue that task writes to.
    """

    clean_message = sanitize_input_text(request.message)
    if not clean_message:
        raise InvalidQueryException("Câu hỏi không được để trống hoặc không hợp lệ.")
    _log_suspected_prompt_injection(
        clean_message, role=security.role, conversation_id=request.conversation_id
    )

    client_ip = _resolve_client_ip(http_request, x_forwarded_for)
    guest_session_token = _resolve_guest_session_token(http_request)

    clarification_read = asyncio.create_task(
        ClarificationStateRepository(db_session).get_clarification(request.conversation_id)
    )
    try:
        turn = await _start_turn(
            java_client,
            conversation_id=request.conversation_id,
            content=clean_message,
            authorization=authorization,
            guest_session_token=guest_session_token,
        )
    except BaseException:
        # Don't close the request's DB session under an in-flight query.
        with contextlib.suppress(Exception):
            await clarification_read
        raise
    pending_clarification, confirmed_metadata = await clarification_read

    first_turn = bool(turn.get("firstTurn"))
    history = _history_from_context(turn.get("context") or [])
    assistant_message_id = str(turn["assistantMessage"]["id"])
    user_message_id = str(turn["userMessage"]["id"])

    request_id = str(uuid.uuid4())
    budget_tracker = get_default_tracker()
    usage_recorder = UsageRecorder(
        request_id=request_id,
        purpose="CHAT",
        conversation_id=request.conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        user_id=security.user_id,
        guest_ip=client_ip if security.is_guest else None,
        budget_tracker=budget_tracker,
    )
    budget = BudgetContext(
        tracker=budget_tracker,
        request_id=request_id,
        reserve_seq=usage_recorder.reserve_budget_seq,
        per_attempt_estimate_usd=_estimate_chat_call_cost_usd(clean_message, models),
    )

    graph_input = GraphInput(
        conversation_id=request.conversation_id,
        user_message=clean_message,
        is_first_turn=first_turn,
        security=security,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending_clarification,
        clarification_max_retry=settings.CHAT_CLARIFICATION_MAX_RETRY,
        history=history,
    )

    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    task = asyncio.create_task(
        run_and_persist(
            java_client=java_client,
            conversation_id=request.conversation_id,
            assistant_message_id=assistant_message_id,
            authorization=authorization,
            client_ip=client_ip,
            graph_input=graph_input,
            models=models,
            usage_recorder=usage_recorder,
            queue=queue,
            session_factory=session_factory,
            budget=budget,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(_sse_token_generator(queue), media_type="text/event-stream")
