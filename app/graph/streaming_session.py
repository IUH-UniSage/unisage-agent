"""Cancellation-safe graph execution + Java persistence.

`run_and_persist` MUST be scheduled with `asyncio.create_task()` by its
caller and never awaited inline inside the SSE response cycle -
`app/api/v1/chat.py`'s endpoint creates the task and returns a
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

import logging
from asyncio import Queue
from decimal import Decimal
from typing import Literal

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
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import async_session_factory
from app.graph.queue_items import DoneItem, ErrorItem, QueueItem, TokenItem
from app.graph.stream_error_codes import (
    LLM_STREAM_INTERRUPTED,
    MESSAGES,
    RETRYABLE,
    VECTOR_STORE_ERROR,
)
from app.graph.streaming import BudgetContext
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import BackendJavaClient

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
            graph_output = await run_graph(
                graph_input, models, sink, trace, usage_recorder, budget=budget
            )
            response_text = graph_output.response_text
            status = "COMPLETED"
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

        try:
            await java_client.update_message(
                message_id=assistant_message_id,
                conversation_id=conversation_id,
                content=response_text,
                status=status,
                citations=(graph_output.citations or None) if graph_output is not None else None,
                authorization=authorization,
            )
        except Exception:
            # Catches `BackendJavaError` (Java rejected/couldn't be reached)
            # AND any other unexpected exception - a bug in the client must
            # not prevent the clarification-state write below or the
            # sentinel put in `finally` from happening.
            logger.exception(
                "failed to PATCH backend-java final message state for conversation_id=%s, "
                "message_id=%s (status=%s) - message stays STREAMING in Java's DB",
                conversation_id,
                assistant_message_id,
                status,
            )

        if graph_output is not None:
            try:
                async with session_factory() as session:
                    repo = ClarificationStateRepository(session)
                    await repo.upsert(
                        conversation_id,
                        pending_clarification=graph_output.pending_clarification,
                        confirmed_metadata=graph_output.confirmed_metadata,
                    )
                    await session.commit()
            except Exception:
                logger.exception(
                    "failed to persist clarification state for conversation_id=%s",
                    conversation_id,
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
