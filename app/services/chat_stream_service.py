"""Chat stream service: everything one `POST /chat/stream` turn does.

The controller (app/api/v1/chat.py) only reads the HTTP request and returns
what this service yields. Ordering matters (conversation ownership is verified
before any state is created, and persistence must survive a client
disconnect):

1. Body size, then the clarification panel gate (app/services/
   clarification_service.py) - every refusal happens before Java is called.
2. `POST /messages/turn`: Java checks ownership, returns first-turn + history,
   then stores the USER message and the ASSISTANT placeholder in one
   transaction. A 404/403/429 raises straight back to the client.
3. `run_and_persist` is scheduled as an independent `asyncio.create_task()` -
   NOT awaited here - and the returned generator only reads the queue that
   task writes to.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.budget.tracker import get_default_tracker
from app.core.config import settings
from app.core.errors.exceptions import (
    BackendJavaUnavailableException,
    ConversationRejectedException,
    InvalidQueryException,
    RequestTooLargeException,
    UsageLimitExceededException,
)
from app.core.security.sanitizer import detect_prompt_injection, sanitize_input_text
from app.core.usage.cost_calculator import estimate as estimate_cost
from app.core.usage.usage_recorder import UsageRecorder
from app.database.repositories.clarification_state import ClarificationRoundRepository
from app.graph.queue_items import (
    ClarificationClosedItem,
    ClarificationItem,
    DoneItem,
    ErrorItem,
    QueueItem,
    TokenItem,
    WarningItem,
)
from app.graph.streaming import BudgetContext
from app.graph.streaming_session import ClaimContext, run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels, ResumeInput
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaError,
    BackendJavaHTTPError,
)
from app.schemas.chat import CHAT_REQUEST_MAX_BYTES, ChatStreamRequest
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import ClarificationCancel, ClarificationSubmit, LastCalculation
from app.schemas.security import AcademicSecurityContext
from app.services.clarification_service import (
    cancel_panel,
    gate_clarification,
    prepare_submit,
    release_claim,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatCaller:
    """Who is asking, as read off the HTTP request by the controller."""

    security: AcademicSecurityContext
    authorization: str | None
    client_ip: str | None
    guest_session_token: str | None


_CONVERSATION_REJECTED_STATUSES = (401, 403, 404)


# asyncio.create_task() only holds a WEAK reference to the task it schedules
# per the stdlib's own docs - without keeping a strong reference somewhere,
# the task can be garbage-collected mid-run. This set is that reference; the
# done-callback discards it once the task (run_and_persist) finishes, so the
# set doesn't grow unbounded.
_background_tasks: set[asyncio.Task[None]] = set()


async def _start_turn(
    java_client: BackendJavaClient,
    *,
    conversation_id: str,
    content: str,
    authorization: str | None,
    guest_session_token: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return await java_client.start_turn(
            conversation_id=conversation_id,
            content=content,
            authorization=authorization,
            guest_session_token=guest_session_token,
            metadata=metadata,
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
        elif isinstance(item, ClarificationItem):
            yield f"event: clarification\ndata: {json.dumps(item.panel, ensure_ascii=False)}\n\n"
        elif isinstance(item, ClarificationClosedItem):
            payload = {"panel_id": item.panel_id, "status": item.status}
            yield f"event: clarification_closed\ndata: {json.dumps(payload)}\n\n"
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


class ChatStreamService:
    def __init__(
        self,
        *,
        db_session: AsyncSession,
        java_client: BackendJavaClient,
        models: GraphModels,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._db = db_session
        self._java = java_client
        self._models = models
        self._session_factory = session_factory

    async def open_stream(
        self, request: ChatStreamRequest, caller: ChatCaller, *, body_size: int
    ) -> AsyncGenerator[str, None]:
        """Run every check, start the turn, and return the SSE event stream."""

        if body_size > CHAT_REQUEST_MAX_BYTES:
            raise RequestTooLargeException()

        conversation_id = request.conversation_id
        round_state = await ClarificationRoundRepository(self._db).get_round(conversation_id)
        await self._db.commit()  # get_round may have consumed an expired lease
        pending_round = gate_clarification(round_state, request)

        resume: ResumeInput | None = None
        claim: ClaimContext | None = None
        start_turn_metadata: dict[str, Any] | None = None
        if pending_round is not None:
            action = request.clarification
            if isinstance(action, ClarificationCancel):
                return await cancel_panel(
                    db_session=self._db,
                    java_client=self._java,
                    conversation_id=conversation_id,
                    pending=pending_round,
                    action=action,
                    confirmed_metadata=round_state.confirmed_metadata,
                )
            assert isinstance(action, ClarificationSubmit)
            prepared = await prepare_submit(
                db_session=self._db,
                conversation_id=conversation_id,
                pending=pending_round,
                action=action,
            )
            claim, resume = prepared.claim, prepared.resume
            message = prepared.summary
            start_turn_metadata = prepared.start_turn_metadata
        else:
            assert request.message is not None
            message = sanitize_input_text(request.message)
            if not message:
                raise InvalidQueryException("Câu hỏi không được để trống hoặc không hợp lệ.")
            _log_suspected_prompt_injection(
                message, role=caller.security.role, conversation_id=conversation_id
            )

        try:
            turn = await _start_turn(
                self._java,
                conversation_id=conversation_id,
                content=message,
                authorization=caller.authorization,
                guest_session_token=caller.guest_session_token,
                metadata=start_turn_metadata,
            )
        except BaseException:
            if claim is not None:
                await release_claim(self._db, conversation_id, claim)
            raise

        return self._run_turn(
            turn,
            conversation_id=conversation_id,
            message=message,
            caller=caller,
            confirmed_metadata=round_state.confirmed_metadata,
            last_calculation=round_state.last_calculation,
            resume=resume,
            claim=claim,
        )

    def _run_turn(
        self,
        turn: dict[str, Any],
        *,
        conversation_id: str,
        message: str,
        caller: ChatCaller,
        confirmed_metadata: dict[str, str],
        last_calculation: LastCalculation | None,
        resume: ResumeInput | None,
        claim: ClaimContext | None,
    ) -> AsyncGenerator[str, None]:
        assistant_message_id = str(turn["assistantMessage"]["id"])
        user_message_id = str(turn["userMessage"]["id"])
        security = caller.security

        request_id = str(uuid.uuid4())
        budget_tracker = get_default_tracker()
        usage_recorder = UsageRecorder(
            request_id=request_id,
            purpose="CHAT",
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            user_id=security.user_id,
            guest_ip=caller.client_ip if security.is_guest else None,
            budget_tracker=budget_tracker,
        )
        budget = BudgetContext(
            tracker=budget_tracker,
            request_id=request_id,
            reserve_seq=usage_recorder.reserve_budget_seq,
            per_attempt_estimate_usd=_estimate_chat_call_cost_usd(message, self._models),
        )
        graph_input = GraphInput(
            conversation_id=conversation_id,
            user_message=message,
            is_first_turn=bool(turn.get("firstTurn")),
            security=security,
            confirmed_metadata=confirmed_metadata,
            history=_history_from_context(turn.get("context") or []),
            resume=resume,
            last_calculation=last_calculation,
        )

        queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        task = asyncio.create_task(
            run_and_persist(
                java_client=self._java,
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                authorization=caller.authorization,
                client_ip=caller.client_ip,
                graph_input=graph_input,
                models=self._models,
                usage_recorder=usage_recorder,
                queue=queue,
                session_factory=self._session_factory,
                budget=budget,
                claim=claim,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return _sse_token_generator(queue)
