"""T1.13c/d — cancellation-safe graph execution + Java persistence.

Non-negotiable requirement from tasks/plan.md: `run_and_persist` MUST be
scheduled with `asyncio.create_task()` by its caller and never awaited
inline inside the SSE response cycle - `app/api/v1/chat.py`'s endpoint
creates the task and returns a `StreamingResponse` whose generator only
reads `queue.get()`. If the client disconnects, Starlette cancels the SSE
generator (and stops iterating the queue) but this task, running
independently, keeps going to completion and still PATCHes Java in
`finally`. This is why streaming must NOT be implemented as
`async for token in graph: yield token` directly inside the response
generator (see plan.md) - that would tie graph execution to the response's
own cancel scope.
"""

import logging
from asyncio import Queue
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import async_session_factory
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import BackendJavaClient, BackendJavaError

logger = logging.getLogger(__name__)


async def run_and_persist(
    *,
    java_client: BackendJavaClient,
    conversation_id: str,
    assistant_message_id: str,
    authorization: str | None,
    graph_input: GraphInput,
    models: GraphModels,
    queue: "Queue[str | None]",
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
) -> None:
    """Run the graph, stream tokens into `queue`, always finalize.

    Uses its own database session (via `session_factory`, defaulting to the
    module-level `async_session_factory` - NOT the request's session)
    because this task must keep running after the HTTP request that started
    it may have already finished or been cancelled.

    Always ends by putting `None` (end-of-stream sentinel) onto `queue`,
    and always calls Java's `PATCH /messages/{id}` with the final status -
    both in `finally`, so neither a graph exception nor a cancelled SSE
    consumer can skip them.
    """

    accumulated: list[str] = []

    async def sink(token: str) -> None:
        accumulated.append(token)
        await queue.put(token)

    status: Literal["COMPLETED", "ERROR"]
    response_text: str
    graph_output = None
    try:
        graph_output = await run_graph(graph_input, models, sink)
        response_text = graph_output.response_text
        status = "COMPLETED"
    except Exception:
        logger.exception(
            "graph execution failed for conversation_id=%s, message_id=%s",
            conversation_id,
            assistant_message_id,
        )
        response_text = "".join(accumulated)
        status = "ERROR"

    try:
        await java_client.update_message(
            message_id=assistant_message_id,
            conversation_id=conversation_id,
            content=response_text,
            status=status,
            authorization=authorization,
        )
    except BackendJavaError:
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
                "failed to persist clarification state for conversation_id=%s", conversation_id
            )

    await queue.put(None)
