from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.config import settings
from app.core.graph_trace import GraphTrace
from app.graph.nodes.greeting import GREETING_TEMPLATE
from app.graph.nodes.intent_routing import SOCIAL_CHAT_TEMPLATE
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATE
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)
_DUMMY_CHUNK = RetrievedChunk(
    chunk_id="c1", content="dummy retrieved content", source="s", score=0.9
)


def _models(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    *,
    classification: str = "academic_advisory",
) -> GraphModels:
    return GraphModels(
        classification=mock_sync_llm_model(classification),
        direct_llm=mock_streaming_llm_model(["42"]),
        query_transformation=mock_sync_llm_model("HyDE doc giả định"),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )


def _sink(target: list[str]) -> TokenSink:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


@pytest.mark.asyncio
async def test_greeting_fast_path_on_first_turn_no_llm_needed(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Chào bạn",
        is_first_turn=True,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input, _models(mock_sync_llm_model, mock_streaming_llm_model), _sink(tokens), _TRACE
    )

    assert result.response_text == GREETING_TEMPLATE
    assert tokens == [GREETING_TEMPLATE]


@pytest.mark.asyncio
async def test_social_chat_routes_to_static_template(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Cảm ơn bạn nhiều",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="social_chat"),
        _sink(tokens),
        _TRACE,
    )

    assert result.response_text == SOCIAL_CHAT_TEMPLATE


@pytest.mark.asyncio
async def test_off_topic_routes_to_static_template(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Cho mình công thức nấu phở",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="off_topic"),
        _sink(tokens),
        _TRACE,
    )

    assert result.response_text == OFF_TOPIC_TEMPLATE


@pytest.mark.asyncio
async def test_general_knowledge_routes_to_direct_llm(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="1 + 1 bằng mấy?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="general_knowledge"),
        _sink(tokens),
        _TRACE,
    )

    assert result.response_text == "42"


@pytest.mark.asyncio
async def test_academic_advisory_routes_through_full_rag_pipeline(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test proves routing/wiring, not the crude demo-corpus scoring
    # heuristic - keep the threshold permissive so it isn't coupled to that.
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.0)
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Điều kiện học bổng loại giỏi là gì?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="academic_advisory"),
        _sink(tokens),
        _TRACE,
    )

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert result.used_ticket_fallback is False


@pytest.mark.asyncio
async def test_no_valid_context_falls_back_to_ticket(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Điều kiện học bổng loại giỏi là gì?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []

    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="academic_advisory"),
        _sink(tokens),
        _TRACE,
    )

    assert result.used_ticket_fallback is True
    assert "chưa tìm thấy" in result.response_text.lower()


@pytest.mark.asyncio
async def test_clarification_guard_match_skips_classification_and_resumes_advisory(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.0)
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Chính quy ạ",
        is_first_turn=False,
        security=AcademicSecurityContext(),
        pending_clarification=pending,
    )
    tokens: list[str] = []

    # classification model returns something that, if it were reached, would
    # be a dead giveaway (off_topic) - proves classification was skipped.
    result = await run_graph(
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model, classification="off_topic"),
        _sink(tokens),
        _TRACE,
    )

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert result.confirmed_metadata == {"training_type": "chinh_quy"}
