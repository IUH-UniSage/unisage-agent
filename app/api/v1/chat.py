from fastapi import APIRouter, Depends

from app.api.deps import get_chat_deps
from app.core.exceptions import InvalidQueryException
from app.core.sanitizer import sanitize_input_text
from app.core.security import verify_internal_secret
from app.core.trace import TraceLogger
from app.graph.deps import ChatDeps
from app.graph.graph import chat_graph
from app.graph.state import ChatState
from app.rag.generation.suggestions import SuggestionService
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_internal_secret)])
suggestion_service = SuggestionService()


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
