import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_backend_java_client,
    get_chat_deps,
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
from app.core.trace import TraceLogger
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.database.session import get_db_session
from app.graph.deps import ChatDeps
from app.graph.graph import chat_graph
from app.graph.nodes.greeting import is_first_turn
from app.graph.nodes.security_context import parse_security_headers
from app.graph.state import ChatState
from app.graph.streaming_session import run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
)
from app.rag.generation.suggestions import SuggestionService
from app.schemas.chat import ChatRequest, ChatResponse, ChatStreamRequest, Citation
from app.schemas.common import ApiResponse
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_internal_secret)])
suggestion_service = SuggestionService()

# asyncio.create_task() only holds a WEAK reference to the task it schedules
# per the stdlib's own docs - without keeping a strong reference somewhere,
# the task can be garbage-collected mid-run. This set is that reference; the
# done-callback discards it once the task (run_and_persist) finishes, so the
# set doesn't grow unbounded.
_background_tasks: set[asyncio.Task[None]] = set()


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat_endpoint(
    request: ChatRequest,
    deps: ChatDeps = Depends(get_chat_deps),
) -> ApiResponse[ChatResponse]:
    clean_query = sanitize_input_text(request.query)
    if not clean_query:
        raise InvalidQueryException("Câu hỏi không được để trống hoặc không hợp lệ.")

    trace_logger = TraceLogger(query=clean_query, user_faculty=request.user_faculty)
    state = ChatState(
        query=clean_query,
        user_faculty=request.user_faculty,
        user_level=request.user_level,
        trace_id=trace_logger.trace.trace_id,
    )

    response_text = await chat_graph.run(inputs=clean_query, state=state, deps=deps)
    trace_logger.finalize(
        final_response=state.final_response or response_text,
        retrieved_chunk_ids=[
            str(chunk["chunk_id"]) for chunk in state.retrieved_chunks if "chunk_id" in chunk
        ],
    )

    return ApiResponse.success(
        ChatResponse(
            trace_id=trace_logger.trace.trace_id,
            query=state.query,
            response=state.final_response or response_text,
            intent=state.intent,
            citations=[Citation.model_validate(citation) for citation in state.citations],
            suggestions=suggestion_service.generate_suggestions(
                query=clean_query,
                intent=state.intent,
            ),
        )
    )


async def _sse_token_generator(queue: "asyncio.Queue[str | None]") -> AsyncGenerator[str, None]:
    """Reads tokens from `queue` until the end-of-stream sentinel (`None`).

    Deliberately does nothing else - no graph execution, no Java calls. This
    is the piece Starlette cancels on client disconnect; `run_and_persist`
    (T1.13c/d), which does the real work, runs in an independent
    `asyncio.create_task()` and is never awaited here.
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
    security: AcademicSecurityContext = Depends(parse_security_headers),
    authorization: str | None = Header(default=None),
    db_session: AsyncSession = Depends(get_db_session),
    java_client: BackendJavaClient = Depends(get_backend_java_client),
    models: GraphModels = Depends(get_graph_models),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> StreamingResponse:
    """T1.13b/c/d/e — SSE streaming chat endpoint.

    Ordering matters and is the whole point of this endpoint (see
    tasks/plan.md's conversation_id ownership invariant and cancellation-safe
    persistence requirement):

    1. Ask Java for this conversation's message count (first-turn detection).
    2. Call Java `POST /messages` (role=USER) FIRST, synchronously, still
       inside this request/response cycle. If Java rejects it (404/403 -
       conversation doesn't exist or belongs to someone else), this raises
       straight back to the client as an HTTP error - no graph run, no
       assistant placeholder, ever.
    3. Only once that succeeds: create the ASSISTANT `STREAMING` placeholder.
    4. Load this conversation's clarification state (T1.1).
    5. Schedule `run_and_persist` as an independent `asyncio.create_task()`
       (T1.13c/d) - NOT awaited here - and return a `StreamingResponse`
       whose generator only reads the queue that task writes to.
    """

    clean_message = sanitize_input_text(request.message)
    if not clean_message:
        raise InvalidQueryException("Câu hỏi không được để trống hoặc không hợp lệ.")

    first_turn = await is_first_turn(
        java_client, conversation_id=request.conversation_id, authorization=authorization
    )

    try:
        await java_client.create_message(
            conversation_id=request.conversation_id,
            role="USER",
            content=clean_message,
            status="COMPLETED",
            authorization=authorization,
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
    )

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(
        run_and_persist(
            java_client=java_client,
            conversation_id=request.conversation_id,
            assistant_message_id=assistant_message_id,
            authorization=authorization,
            graph_input=graph_input,
            models=models,
            queue=queue,
            session_factory=session_factory,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(_sse_token_generator(queue), media_type="text/event-stream")
