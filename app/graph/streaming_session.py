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

Task 11 (plan.md "SSE error contract") adds one more thing to that same
outer `finally`'s neighborhood: when `run_graph(...)` raises, an `ErrorItem`
describing it is put onto `queue` immediately BEFORE the `DoneItem`
sentinel - still inside the same outer `try`, so it's put exactly once, and
still before the unconditional `finally` puts `DoneItem` - `event: error`
never has a chance to arrive after `event: done`.
"""

import logging
from asyncio import Queue
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.graph_trace import GraphTrace
from app.core.model_router import NoAvailableCredentialError
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import async_session_factory
from app.graph.queue_items import DoneItem, ErrorItem, QueueItem, TokenItem
from app.graph.stream_error_codes import (
    LLM_STREAM_INTERRUPTED,
    LLM_UNAVAILABLE,
    MESSAGES,
    RETRYABLE,
)
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import BackendJavaClient

logger = logging.getLogger(__name__)


def _error_item_for(exc: Exception, *, streamed_any: bool) -> ErrorItem:
    """Maps a graph-execution failure to the `event: error` payload.

    `NoAvailableCredentialError` only ever surfaces before any chunk of the
    generation/ticket-fallback response streamed (`stream_agent_text()`
    raises it uncaught, and it can only originate there) - so it always maps
    to `LLM_UNAVAILABLE` regardless of `streamed_any`. Every other exception
    that reaches here either happened after a chunk already streamed (the
    "no retry past this point" boundary in `stream_agent_text()`) or before
    any chunk streamed for a non-credential-exhaustion reason - both map to
    `LLM_STREAM_INTERRUPTED`, the only other non-reserved code the wire
    contract defines.
    """

    if isinstance(exc, NoAvailableCredentialError):
        code = LLM_UNAVAILABLE
    else:
        code = LLM_STREAM_INTERRUPTED
    return ErrorItem(code=code, message=MESSAGES[code], retryable=RETRYABLE[code])


async def run_and_persist(
    *,
    java_client: BackendJavaClient,
    conversation_id: str,
    assistant_message_id: str,
    authorization: str | None,
    client_ip: str | None = None,
    graph_input: GraphInput,
    models: GraphModels,
    queue: "Queue[QueueItem]",
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
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

    status: Literal["COMPLETED", "ERROR"]
    response_text: str
    graph_output = None
    error_item: ErrorItem | None = None
    try:
        try:
            graph_output = await run_graph(graph_input, models, sink, trace)
            response_text = graph_output.response_text
            status = "COMPLETED"
        except Exception as exc:
            logger.exception(
                "graph execution failed for conversation_id=%s, message_id=%s",
                conversation_id,
                assistant_message_id,
            )
            response_text = "".join(accumulated)
            status = "ERROR"
            error_item = _error_item_for(exc, streamed_any=bool(accumulated))

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
        if error_item is not None:
            await queue.put(error_item)
        await queue.put(DoneItem())
