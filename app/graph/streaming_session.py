"""Cancellation-safe graph execution + Java persistence.

`run_and_persist` MUST be scheduled with `asyncio.create_task()` by its
caller and never awaited inline inside the SSE response cycle -
`ChatStreamService` (app/services/chat_stream_service.py) creates the task and the
controller returns a
`StreamingResponse` whose generator only reads `queue.get()`. If the client
disconnects, Starlette cancels the SSE generator (and stops iterating the
queue) but this task, running independently, keeps going to completion and
still PATCHes Java in `finally`. This is why streaming must NOT be
implemented as `async for token in graph: yield token` directly inside the
response generator - that would tie graph execution to the response's own
cancel scope.

Every finalization step below (the Java PATCH, the clarification-state
write) is wrapped in its own `except Exception` so that a bug in one of
them can never prevent the others from running - and the whole function
body is wrapped in one outer `try`/`finally` whose `finally` puts the
end-of-stream sentinel onto `queue`. That outer `finally` is the actual
fix for the bug this module previously had: the sentinel used to be a
plain statement *after* a sequence of independent `try/except` blocks, not
inside any `finally` - so an exception type not anticipated by one of
those inner `except` clauses (e.g. `BackendJavaClient.update_message`
raising a bare `TypeError` on an empty response body, since fixed at the
source in `app/integrations/backend_java_client.py`) would propagate
straight out of `run_and_persist`, skip the sentinel put, and leave the SSE
generator's `while True: token = await queue.get()` waiting forever - the
client's connection would then never see `event: done` and never close.

The SSE error contract adds one more thing to that same
outer `finally`'s neighborhood: when `run_graph(...)` raises, an `ErrorItem`
describing it is put onto `queue` immediately BEFORE the `DoneItem`
sentinel - still inside the same outer `try`, so it's put exactly once, and
still before the unconditional `finally` puts `DoneItem` - `event: error`
never has a chance to arrive after `event: done`.
"""

import asyncio
import logging
import uuid
from asyncio import Queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.budget.tracker import RequestBudgetRejectedError
from app.core.config import settings
from app.core.errors.llm_failure import LLMFailure, describe_llm_failure, is_model_failure
from app.core.errors.public_errors import (
    can_see_ai_details,
    public_chat_message,
    short_reference,
)
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.database.repositories.clarification_state import ClarificationRoundRepository
from app.database.session import async_session_factory
from app.graph.queue_items import (
    ClarificationItem,
    DoneItem,
    ErrorItem,
    QueueItem,
    TokenItem,
    WarningItem,
)
from app.graph.stream_error_codes import (
    LLM_STREAM_INTERRUPTED,
    MESSAGES,
    RETRYABLE,
    VECTOR_STORE_ERROR,
)
from app.graph.streaming import BudgetContext
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput
from app.integrations.backend_java_client import BackendJavaClient, with_java_retries
from app.schemas.clarification import PendingRound

logger = logging.getLogger(__name__)


def _error_item_for(
    exc: Exception,
    *,
    streamed_any: bool,
    detailed: bool = True,
    reference: str | None = None,
) -> ErrorItem:
    """Maps a graph-execution failure to the `event: error` payload.

    `code` is always the precise cause - `describe_llm_failure`'s reason for an AI-model
    failure (CHAT in any node, EMBEDDING during retrieval: "LLM_AUTH_FAILED", ...),
    `VECTOR_STORE_ERROR` for Qdrant, `LLM_STREAM_INTERRUPTED` for anything else (a bug).

    `message` depends on `detailed`: the full technical explanation for a caller who can
    fix it (an AI admin - see `can_see_ai_details`), otherwise one of
    `public_chat_message`'s plain categories (fix your question / try again later /
    system problem) - a student or guest can't act on "API key không hợp lệ" and must not
    learn the infrastructure from it. `reference` (the request's short id, also in the
    server log line) is appended either way so a reported error can be traced.
    """

    failure: LLMFailure | None = None
    code: str
    if is_model_failure(exc):
        failure = describe_llm_failure(exc, purpose="CHAT")
        code, message, retryable = failure.reason, failure.message, failure.retryable
    elif type(exc).__module__.startswith("qdrant_client"):
        code = VECTOR_STORE_ERROR
        message, retryable = MESSAGES[code], RETRYABLE[code]
    else:
        code = LLM_STREAM_INTERRUPTED
        message = f"{MESSAGES[code]} (lỗi nội bộ: {type(exc).__name__})"
        retryable = RETRYABLE[code]

    if detailed:
        if reference:
            message = f"{message} (Mã tham chiếu: {reference})"
    else:
        message, retryable = public_chat_message(failure, reference=reference)
        if code == VECTOR_STORE_ERROR:
            retryable = True
    if streamed_any and not (detailed and code == LLM_STREAM_INTERRUPTED):
        message = f"Câu trả lời bị gián đoạn giữa chừng. {message}"
    return ErrorItem(code=code, message=message, retryable=retryable)


def _admin_warnings(graph_output: GraphOutput, graph_input: GraphInput) -> list[WarningItem]:
    """`event: warning`s for an AI admin (web search or the LLM rerank failed,
    the Extraction model is missing, ...) - everyone else just gets the answer,
    with no mention of what was skipped."""

    if not can_see_ai_details(graph_input.security.permissions):
        return []
    return [
        WarningItem(code=warning.code, message=warning.message)
        for warning in graph_output.admin_warnings
    ]


_LOST_CLAIM = object()


async def _finalize_safely(call: Callable[[], Awaitable[object]], *, what: str) -> bool:
    """Retried PATCH; any non-Java exception (a client bug) also counts as a failure
    instead of escaping - the end-of-stream sentinel must always be queued."""

    try:
        return await with_java_retries(call, what=what)
    except Exception:
        logger.exception("%s raised unexpectedly", what)
        return False


@dataclass(frozen=True)
class ClaimContext:
    """A submit turn holds the conversation's panel claim (PROCESSING) until it ends."""

    token: uuid.UUID
    # `loop.time()` deadline for everything after the claim (CHAT_CLAIMED_TURN_DEADLINE_SECONDS).
    deadline: float | None = None


async def _store_round_state(
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: str,
    claim: ClaimContext | None,
    pending_round: PendingRound | None,
    confirmed_metadata: dict[str, str],
) -> object:
    """True when written, False when the write failed or was refused, `_LOST_CLAIM`
    when a submit turn no longer owns its claim."""

    try:
        async with session_factory() as session:
            repo = ClarificationRoundRepository(session)
            if claim is not None:
                done = await repo.complete(
                    conversation_id, claim.token, pending_round, confirmed_metadata
                )
                await session.commit()
                return True if done else _LOST_CLAIM
            if pending_round is not None:
                opened = await repo.upsert_open(conversation_id, pending_round, confirmed_metadata)
            else:
                await repo.save_confirmed_metadata(conversation_id, confirmed_metadata)
                opened = True
            await session.commit()
            return opened
    except Exception:
        logger.exception(
            "failed to persist clarification state for conversation_id=%s", conversation_id
        )
        return False


async def _revoke_round(
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: str,
    pending_round: PendingRound,
) -> None:
    """The panel could not be projected to Java: take it back so no panel exists on
    one side only."""

    try:
        async with session_factory() as session:
            await ClarificationRoundRepository(session).revoke_open(
                conversation_id, pending_round.panel.panel_id
            )
            await session.commit()
    except Exception:
        logger.exception(
            "failed to revoke clarification round for conversation_id=%s", conversation_id
        )


async def run_and_persist(
    *,
    java_client: BackendJavaClient,
    conversation_id: str,
    assistant_message_id: str,
    authorization: str | None,
    client_ip: str | None = None,
    graph_input: GraphInput,
    models: GraphModels,
    usage_recorder: UsageRecorder,
    queue: "Queue[QueueItem]",
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
    budget: BudgetContext | None = None,
    claim: ClaimContext | None = None,
) -> None:
    """Run the graph, stream tokens into `queue`, always finalize.

    Uses its own database session (via `session_factory`, defaulting to the
    module-level `async_session_factory` - NOT the request's session)
    because this task must keep running after the HTTP request that started
    it may have already finished or been cancelled.

    Always ends by putting a `DoneItem` (end-of-stream sentinel) onto
    `queue` in an outer `finally` wrapping the entire function body, and
    always attempts Java's `PATCH /messages/{id}` with the final status - so
    neither a graph exception, an unexpected exception from the Java call,
    nor a cancelled SSE consumer can skip the sentinel.
    """

    accumulated: list[str] = []

    async def sink(token: str) -> None:
        trace.first_token()
        accumulated.append(token)
        await queue.put(TokenItem(token))

    trace = GraphTrace(
        conversation_id=conversation_id,
        message_id=assistant_message_id,
        user_id=graph_input.security.user_id,
        client_ip=client_ip,
    )

    # Bound up front (not just declared) so `finally` below can always read it, even
    # if `run_graph` raises something the inner `except Exception` doesn't catch
    # (e.g. `asyncio.CancelledError`, a `BaseException`) - matches this module's own
    # past bug (see module docstring) of "never skip the finally".
    status: Literal["COMPLETED", "ERROR"] = "ERROR"
    response_text: str
    graph_output = None
    error_item: ErrorItem | None = None
    try:
        # A submit turn holds the panel claim: past its deadline it is cancelled
        # before any further Java or state write (the lease outlives the deadline).
        async with asyncio.timeout_at(claim.deadline if claim is not None else None):
            try:
                if budget is not None:
                    # Request-level SYSTEM/PURPOSE reservation covers every attempt across
                    # every node in this request - multiplied up from one call's estimate to
                    # account for the 2-3 secondary LLM calls (classification, query
                    # transformation) beyond the primary generation call.
                    request_estimate_usd = budget.per_attempt_estimate_usd * Decimal(
                        str(settings.BUDGET_RESERVATION_MULTIPLIER_CHAT)
                    )
                    reserve_result = await budget.tracker.reserve_request(
                        request_id=budget.request_id,
                        purpose=usage_recorder.purpose,
                        estimate_usd=request_estimate_usd,
                    )
                    if reserve_result != "OK":
                        raise RequestBudgetRejectedError(usage_recorder.purpose, reserve_result)
                try:
                    graph_output = await run_graph(
                        graph_input, models, sink, trace, usage_recorder, budget=budget
                    )
                finally:
                    trace.finish()
                response_text = graph_output.response_text
                status = "COMPLETED"
                for warning in _admin_warnings(graph_output, graph_input):
                    await queue.put(warning)
            except Exception as exc:
                reference = short_reference(usage_recorder.request_id)
                logger.exception(
                    "graph execution failed for conversation_id=%s, message_id=%s, ref=%s",
                    conversation_id,
                    assistant_message_id,
                    reference,
                )
                response_text = "".join(accumulated)
                status = "ERROR"
                error_item = _error_item_for(
                    exc,
                    streamed_any=bool(accumulated),
                    detailed=can_see_ai_details(graph_input.security.permissions),
                    reference=reference,
                )

            pending_round: PendingRound | None = None
            if status == "COMPLETED" and graph_output is not None and graph_output.pending_round:
                pending_round = graph_output.pending_round.model_copy(
                    update={"assistant_message_id": uuid.UUID(assistant_message_id)}
                )
            confirmed = (
                graph_output.confirmed_metadata
                if graph_output is not None
                else graph_input.confirmed_metadata
            )
            # State first, projection second, event last: a client that receives
            # `event: clarification` can always reload and submit that panel.
            persisted = await _store_round_state(
                session_factory, conversation_id, claim, pending_round, confirmed
            )
            if persisted is _LOST_CLAIM:
                # Our claim was taken over (lease expired): another request owns the
                # conversation now - write nothing more to Java.
                return
            round_stored = persisted is True and pending_round is not None

            metadata: dict[str, Any] | None = None
            if round_stored and pending_round is not None:
                metadata = {
                    "clarification": {
                        "schema_version": 1,
                        "status": "open",
                        "panel": pending_round.panel.public().model_dump(mode="json"),
                    }
                }
            finalized = await _finalize_safely(
                lambda: java_client.update_message(
                    message_id=assistant_message_id,
                    conversation_id=conversation_id,
                    content=response_text,
                    status=status,
                    citations=(graph_output.citations or None)
                    if graph_output is not None
                    else None,
                    metadata=metadata,
                    authorization=authorization,
                ),
                what=f"finalize message {assistant_message_id}",
            )
            if not finalized:
                logger.error(
                    "failed to PATCH backend-java final message state for conversation_id=%s, "
                    "message_id=%s (status=%s) - message stays STREAMING in Java's DB",
                    conversation_id,
                    assistant_message_id,
                    status,
                )
            if round_stored and pending_round is not None:
                if finalized:
                    await queue.put(
                        ClarificationItem(pending_round.panel.public().model_dump(mode="json"))
                    )
                else:
                    await _revoke_round(session_factory, conversation_id, pending_round)
    except TimeoutError:
        logger.error(
            "claimed turn passed its deadline for conversation_id=%s, message_id=%s",
            conversation_id,
            assistant_message_id,
        )
        error_item = _error_item_for(
            TimeoutError("claimed turn deadline"),
            streamed_any=bool(accumulated),
            detailed=can_see_ai_details(graph_input.security.permissions),
            reference=short_reference(usage_recorder.request_id),
        )
    finally:
        # Closes exactly once here regardless of which path above ran - success,
        # graph exception, or (since this whole function keeps running independently
        # of the SSE response per the module docstring) a client disconnect.
        # UsageRecorder's status vocabulary (SUCCESS/ERROR/PARTIAL) isn't Java
        # message status (COMPLETED/ERROR/STREAMING) - map explicitly, don't pass
        # `status` through as-is.
        await usage_recorder.close(status="SUCCESS" if status == "COMPLETED" else "ERROR")
        if error_item is not None:
            await queue.put(error_item)
        await queue.put(DoneItem())
