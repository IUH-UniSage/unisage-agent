import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.clarification_answers import validate_answers
from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.graph.nodes.greeting import GREETING_TEMPLATE
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATES
from app.graph.nodes.web_search import WebSearchOutcome, WebSearchRequest
from app.graph.progress import status_reporter
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import _web_request, run_graph
from app.graph.streaming_state import GraphInput, GraphModels, ResumeInput
from app.schemas.clarification import Answer, ClarificationSubmit, PendingRound
from app.schemas.intent import ClassifiedTask
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult, WebSlots
from tests.llm_mocks import FakeRetrievalService, RetrieveManyMixin, make_classification_llm_model

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)


def _usage_recorder() -> UsageRecorder:
    # A fresh instance per call - see the identical helper's docstring in
    # tests/graph/test_calculation_node.py.
    return UsageRecorder(request_id="test-request", purpose="CHAT")


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
        classification=make_classification_llm_model(classification),
        query_transformation=mock_sync_llm_model("HyDE doc giả định"),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )


def _sink(target: list[str]) -> TokenSink:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


def _echo_query_transformation_model() -> FunctionModel:
    """A query-transformation model double that returns its input VERBATIM
    (instead of a fixed string) - lets a test assert exactly which text was
    used to build the retrieval query, without depending on pydantic_ai's
    internal message shape beyond reading the last part's `.content`."""

    def echo(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        text = ""
        for part in messages[-1].parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                text = content
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=echo)


@dataclass
class _RecordingRetrievalService(RetrieveManyMixin):
    """Like `FakeRetrievalService`, but remembers every query it was asked to
    retrieve for - so a test can assert which text actually reached
    retrieval, not just what the graph returned."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        del security
        self.queries.append(query)
        return self.chunks if limit is None else self.chunks[:limit]


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
        graph_input,
        _models(mock_sync_llm_model, mock_streaming_llm_model),
        _sink(tokens),
        _TRACE,
        _usage_recorder(),
    )

    assert result.response_text == GREETING_TEMPLATE
    assert tokens == [GREETING_TEMPLATE]


@pytest.mark.asyncio
async def test_social_chat_streams_the_llm_reply(
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
        _usage_recorder(),
    )

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert "".join(tokens) == result.response_text


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
        _usage_recorder(),
    )

    assert result.response_text in OFF_TOPIC_TEMPLATES


@pytest.mark.asyncio
async def test_academic_advisory_routes_through_full_rag_pipeline(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test proves routing/wiring, not the crude demo-corpus scoring
    # heuristic - keep the threshold permissive so it isn't coupled to that.
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
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
        _usage_recorder(),
    )

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert result.used_ticket_fallback is False


@pytest.mark.asyncio
async def test_no_valid_context_falls_back_to_ticket(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Điều kiện học bổng loại giỏi là gì?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )
    tokens: list[str] = []
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("HyDE doc giả định"),
        generation=mock_streaming_llm_model(["Chưa tìm thấy quy định phù hợp."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    result = await run_graph(graph_input, models, _sink(tokens), _TRACE, _usage_recorder())

    assert result.used_ticket_fallback is True
    assert "chưa tìm thấy" in result.response_text.lower()


def _traced_nodes(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage().split(" ", 1)[0].removeprefix("node=")
        for record in caplog.records
        if record.name == "unisage.graph" and record.getMessage().startswith("node=")
    ]


@pytest.mark.asyncio
async def test_advisory_turn_traces_nodes_with_flow_design_numbering(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Điều kiện học bổng là gì?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        await run_graph(
            graph_input,
            _models(mock_sync_llm_model, mock_streaming_llm_model),
            _sink([]),
            _TRACE,
            _usage_recorder(),
        )

    assert _traced_nodes(caplog) == [
        "01_GreetingDetectionNode",
        "03_MessageClassificationNode",
        "04_IntentRoutingNode",
        "06_QueryTransformationNode",
        "08_RetrievalFilteringNode",
        "09_PostRetrievalRerankNode",
        "10_GenerationSynthesisNode",
    ]


@pytest.mark.asyncio
async def test_off_topic_turn_traces_node_05(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Giá vàng hôm nay?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        await run_graph(
            graph_input,
            _models(mock_sync_llm_model, mock_streaming_llm_model, classification="off_topic"),
            _sink([]),
            _TRACE,
            _usage_recorder(),
        )

    assert _traced_nodes(caplog)[-1] == "05_OffTopicRejectNode"


def _capturing_generation_model(seen: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        for part in messages[-1].parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                seen.append(content)
        yield "Câu trả lời cuối cùng [1]."

    return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
async def test_one_multi_advisory_task_decomposed_into_sub_queries_uses_the_multi_intent_frame(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typical shape post-merge: MessageClassificationNode keeps 2+
    same-intent questions as ONE academic_advisory task with routing_mode
    MULTI; splitting into sub-queries is the decomposer's job, not
    classification's - the two nodes no longer duplicate that work."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    message = "Học phí ngành CNTT bao nhiêu, với lại điều kiện học bổng là gì?"
    payload = {
        "tasks": [
            {"intent": "academic_advisory", "query": message, "routing_mode": "MULTI"},
        ],
        "confidence": 0.9,
    }
    sub_queries_payload = {
        "sub_queries": ["Học phí ngành CNTT bao nhiêu?", "Điều kiện học bổng là gì?"]
    }
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=mock_sync_llm_model(json.dumps(payload, ensure_ascii=False)),
        query_transformation=mock_sync_llm_model(
            json.dumps(sub_queries_payload, ensure_ascii=False)
        ),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message=message,
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    (prompt,) = seen_prompts
    assert "SQ1. Học phí ngành CNTT bao nhiêu?" in prompt
    assert "SQ2. Điều kiện học bổng là gì?" in prompt


@pytest.mark.asyncio
async def test_two_advisory_questions_use_the_multi_intent_frame(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: even if classification ever emits 2 separate SINGLE tasks
    for the same intent instead of 1 MULTI task, the graph still answers
    both via the multi-intent frame."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    message = "Học phí ngành CNTT bao nhiêu, với lại điều kiện học bổng là gì?"
    payload = {
        "tasks": [
            {
                "intent": "academic_advisory",
                "query": "Học phí ngành CNTT bao nhiêu?",
                "routing_mode": "SINGLE",
            },
            {
                "intent": "academic_advisory",
                "query": "Điều kiện học bổng là gì?",
                "routing_mode": "SINGLE",
            },
        ],
        "confidence": 0.9,
    }
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=mock_sync_llm_model(json.dumps(payload, ensure_ascii=False)),
        query_transformation=_echo_query_transformation_model(),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message=message,
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    (prompt,) = seen_prompts
    assert "SQ1. Học phí ngành CNTT bao nhiêu?" in prompt
    assert "SQ2. Điều kiện học bổng là gì?" in prompt


# ── WebSearchNode (between rerank and ticket fallback) ─────────────────────

_WEB_PAGE = WebSearchResult(
    title="Lịch thi HK1",
    url="https://pdt.iuh.edu.vn/lich-thi",
    content="Lịch thi học kỳ 1 bắt đầu ngày 05/01.",
    score=0.8,
)


@dataclass
class _PerQueryRetrieval(RetrieveManyMixin):
    results: dict[str, list[RetrievedChunk]]

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del security, limit
        return self.results[query]


def _fake_web_search(
    monkeypatch: pytest.MonkeyPatch, pages: list[WebSearchResult]
) -> list[list[str]]:
    calls: list[list[str]] = []

    async def search_web(requests: Sequence[WebSearchRequest]) -> WebSearchOutcome:
        # By query text: the sub-queries without a chunk are searched by their question.
        calls.append([request.query for request in requests])
        return WebSearchOutcome(results=pages, per_request=[pages for _ in requests])

    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr("app.graph.streaming_graph.search_web", search_web)
    return calls


def _single_question_input() -> GraphInput:
    return GraphInput(
        conversation_id="c1",
        user_message="Lịch thi học kỳ 1 khi nào?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )


@pytest.mark.asyncio
async def test_single_query_without_chunks_answers_from_the_web_instead_of_a_ticket(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    calls = _fake_web_search(monkeypatch, [_WEB_PAGE])
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("HyDE lịch thi"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    assert calls == [["HyDE lịch thi"]]
    assert result.used_ticket_fallback is False
    assert result.used_web_search is True
    (prompt,) = seen_prompts
    assert "(không có tài liệu liên quan)" in prompt
    assert "[1] (Lịch thi HK1 — https://pdt.iuh.edu.vn/lich-thi)" in prompt
    assert result.citations == [
        {
            "index": 1,
            "documentId": None,
            "title": "Lịch thi HK1",
            "section": None,
            "pageStart": None,
            "pageEnd": None,
            "sourceType": "WEB",
            "url": "https://pdt.iuh.edu.vn/lich-thi",
        }
    ]
    assert "09b_WebSearchNode" in _traced_nodes(caplog)
    assert "11_TicketFallbackNode" not in _traced_nodes(caplog)


@pytest.mark.asyncio
async def test_only_the_sub_query_without_chunks_is_searched_and_both_blocks_reach_the_prompt(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.5)
    calls = _fake_web_search(monkeypatch, [_WEB_PAGE])
    message = "Học phí ngành CNTT bao nhiêu, với lại lịch thi học kỳ 1 khi nào?"
    payload = {
        "tasks": [{"intent": "academic_advisory", "query": message, "routing_mode": "MULTI"}],
        "confidence": 0.9,
    }
    sub_queries = {"sub_queries": ["Học phí ngành CNTT?", "Lịch thi học kỳ 1?"]}
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=mock_sync_llm_model(json.dumps(payload, ensure_ascii=False)),
        query_transformation=mock_sync_llm_model(json.dumps(sub_queries, ensure_ascii=False)),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=_PerQueryRetrieval(
            {
                "Học phí ngành CNTT?": [
                    RetrievedChunk(chunk_id="hp", content="Học phí...", source="hp.pdf", score=0.9)
                ],
                "Lịch thi học kỳ 1?": [
                    RetrievedChunk(chunk_id="lt", content="x", source="lt.pdf", score=0.1)
                ],
            }
        ),
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message=message,
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    result = await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    assert calls == [["Lịch thi học kỳ 1?"]]
    (prompt,) = seen_prompts
    assert "[1] (hp.pdf) Học phí..." in prompt
    assert "[2] (Lịch thi HK1 — https://pdt.iuh.edu.vn/lich-thi)" in prompt
    assert result.used_web_search is True


@pytest.mark.asyncio
async def test_no_chunk_and_no_web_page_still_falls_back_to_ticket(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)
    # Hybrid: with an LLM rerank ahead the graph applies the dense floor instead, so
    # raise it too - these turns must reach rerank with no usable chunk.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_DENSE_FLOOR", 1.1)
    calls = _fake_web_search(monkeypatch, [])
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("HyDE lịch thi"),
        generation=mock_streaming_llm_model(["Chưa tìm thấy quy định phù hợp."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    result = await run_graph(_single_question_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert calls == [["HyDE lịch thi"]]
    assert result.used_ticket_fallback is True
    assert result.used_web_search is False


@pytest.mark.asyncio
async def test_all_sub_queries_with_chunks_never_search_the_web(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    calls = _fake_web_search(monkeypatch, [_WEB_PAGE])

    result = await run_graph(
        _single_question_input(),
        _models(mock_sync_llm_model, mock_streaming_llm_model),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert calls == []
    assert result.used_web_search is False


# ── Speculative web search + WebJudgeNode (09c) ───────────────────────────

_FEE_DECISION = WebSearchResult(
    title="1035-QD-DHCN",
    url="https://ptckt.iuh.edu.vn/upload/files/bieumau/1035-QD-DHCN-11-4-2025.pdf",
    content="Quy định mức học phí năm học 2025-2026 (chi tiết theo phụ lục đính kèm).",
    score=0.08,
    academic_year="2025-2026",
)


def _web_slot_classification(
    mock_sync_llm_model: Callable[[str], FunctionModel], topic: str = "hoc_phi"
) -> FunctionModel:
    payload = {
        "tasks": [
            {
                "intent": "academic_advisory",
                "query": "học phí ngành cntt",
                "routing_mode": "SINGLE",
                "web": {
                    "topics": [topic],
                    "subject": "mức thu học phí",
                    "nganh_hoc": "Công nghệ Thông tin",
                },
            }
        ],
        "confidence": 0.95,
    }
    return mock_sync_llm_model(json.dumps(payload, ensure_ascii=False))


def _fake_slot_web_search(
    monkeypatch: pytest.MonkeyPatch, pages: list[WebSearchResult]
) -> list[list[object]]:
    calls: list[list[object]] = []

    async def search_web(requests: Sequence[object]) -> WebSearchOutcome:
        calls.append(list(requests))
        return WebSearchOutcome(results=pages, per_request=[pages for _ in requests])

    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MODE", "speculative")
    monkeypatch.setattr("app.graph.streaming_graph.search_web", search_web)
    return calls


def _fee_input() -> GraphInput:
    return GraphInput(
        conversation_id="c1",
        user_message="học phí ngành cntt",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )


@pytest.mark.asyncio
async def test_fee_question_searches_the_slot_query_and_answers_from_the_web(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    calls = _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    ((request,),) = calls
    assert isinstance(request, WebSearchRequest)
    assert request.topic == "hoc_phi"
    # Fee schedules are split by Khối: the major stays out of the query; the user named no
    # year, so none is added either.
    assert request.query == "mức thu học phí IUH"
    assert result.used_web_search is True
    codes = [warning.code for warning in result.admin_warnings]
    # No RERANK model here: the page is kept by its score, and the admin is told.
    assert "WEB_JUDGE_FAILED" in codes
    assert "WEB_SOURCE_USED" in codes
    assert "09b_WebSearchNode" in _traced_nodes(caplog)


@pytest.mark.parametrize("with_chunks", [False, True])
@pytest.mark.asyncio
async def test_speculative_search_runs_but_is_not_used_when_the_knowledge_base_answers(
    with_chunks: bool,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Judging alongside the chunks needs the RERANK model: without one, no page picked
    by its score alone goes next to them (and LLM_RERANK_UNAVAILABLE already tells)."""

    monkeypatch.setattr(settings, "CHAT_WEB_JUDGE_WITH_CHUNKS", with_chunks)
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    calls = _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=mock_streaming_llm_model(["Học phí ... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert len(calls) == 1  # started with the advisory branch, for the cache
    assert result.used_web_search is False
    assert "09b_WebSearchNode" not in _traced_nodes(caplog)
    assert not [w for w in result.admin_warnings if w.code.startswith("WEB_")]


def _status_trace(seen: list[tuple[str, str | None]]) -> GraphTrace:
    return GraphTrace(
        conversation_id="c1",
        message_id="m1",
        user_id=None,
        client_ip=None,
        on_status=status_reporter(lambda stage, url: seen.append((stage, url))),
    )


@pytest.mark.asyncio
async def test_web_branch_reports_each_page_it_reads(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_JUDGE_WITH_CHUNKS", False)
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)
    # Hybrid: with an LLM rerank ahead the graph applies the dense floor instead, so
    # raise it too - these turns must reach rerank with no usable chunk.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_DENSE_FLOOR", 1.1)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE, _FEE_DECISION])
    seen: list[tuple[str, str | None]] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=mock_streaming_llm_model(["Học phí ... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    await run_graph(_fee_input(), models, _sink([]), _status_trace(seen), _usage_recorder())

    assert seen == [
        ("thinking", None),
        ("analyzing", None),
        ("retrieving", None),
        ("web_searching", None),
        ("web_searching", _WEB_PAGE.url),
        ("web_searching", _FEE_DECISION.url),
        ("generating", None),
    ]


@pytest.mark.asyncio
async def test_pages_judged_alongside_rerank_are_reported_once_rerank_is_done(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judged while retrieval runs: a status then would interleave with retrieval's, so
    the pages the answer uses are shown after rerank (no RERANK model here: the score
    rule keeps the page above CHAT_WEB_SEARCH_MIN_SCORE only)."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)
    # Hybrid: with an LLM rerank ahead the graph applies the dense floor instead, so
    # raise it too - these turns must reach rerank with no usable chunk.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_DENSE_FLOOR", 1.1)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE, _FEE_DECISION])
    seen: list[tuple[str, str | None]] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=mock_streaming_llm_model(["Học phí ... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    await run_graph(_fee_input(), models, _sink([]), _status_trace(seen), _usage_recorder())

    assert seen == [
        ("thinking", None),
        ("analyzing", None),
        ("retrieving", None),
        ("web_searching", None),
        ("web_searching", _WEB_PAGE.url),
        ("generating", None),
    ]


_JUDGE_KEEPS_THE_PAGE = '{"SQ1": {"W1": {"reason": "lịch thi ngày 05,06/11/2026", "answer": true}}}'


@pytest.mark.asyncio
async def test_a_page_is_judged_and_used_next_to_the_kept_chunks(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 14-10-2026 turn: the 2022 exam results were kept for "the latest exam date",
    so the page with the 2026 schedule was never judged."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", False)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model, "lich_thong_bao"),
        query_transformation=mock_sync_llm_model("HyDE lịch thi"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model(_JUDGE_KEEPS_THE_PAGE),
    )

    with caplog.at_level("INFO"):
        result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert result.used_ticket_fallback is False
    assert result.used_web_search is True
    (prompt,) = seen_prompts
    # The chunk is [1], the page numbered after it.
    assert _DUMMY_CHUNK.content in prompt
    assert f"[2] (Lịch thi HK1 — {_WEB_PAGE.url})" in prompt
    assert "mode=parallel" in caplog.text
    assert "sub_queries_with_chunks=1/1" in caplog.text
    assert [w.code for w in result.admin_warnings] == ["WEB_SOURCE_USED"]


@pytest.mark.asyncio
async def test_the_web_judge_runs_while_retrieval_is_still_running(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval here only returns once the judge has been called: run one after the
    other, the turn would time out."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", False)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    judged = asyncio.Event()

    def judge(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        judged.set()
        return ModelResponse(parts=[TextPart(content=_JUDGE_KEEPS_THE_PAGE)])

    @dataclass(frozen=True)
    class WaitsForTheJudge(FakeRetrievalService):
        async def retrieve_many_async(
            self,
            queries: list[str],
            *,
            security: AcademicSecurityContext,
            limit: int | None = None,
            lexical_queries: list[str] | None = None,
        ) -> list[list[RetrievedChunk]]:
            await asyncio.wait_for(judged.wait(), timeout=2)
            return self.retrieve_many(queries, security=security, limit=limit)

    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model, "lich_thong_bao"),
        query_transformation=mock_sync_llm_model("HyDE lịch thi"),
        generation=mock_streaming_llm_model(["Lịch thi ... [2]."]),
        retrieval=WaitsForTheJudge([_DUMMY_CHUNK]),
        rerank=FunctionModel(function=judge),
    )

    result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert [citation["url"] for citation in result.citations] == [_WEB_PAGE.url]


@pytest.mark.asyncio
async def test_a_failed_judge_adds_no_page_next_to_the_chunks(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", False)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model, "lich_thong_bao"),
        query_transformation=mock_sync_llm_model("HyDE lịch thi"),
        generation=mock_streaming_llm_model(["Lịch thi ... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model("không phải JSON"),
    )

    result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert result.used_web_search is False
    assert [w.code for w in result.admin_warnings] == ["WEB_JUDGE_FAILED"]


@pytest.mark.asyncio
async def test_unused_speculative_search_reports_no_web_status(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    seen: list[tuple[str, str | None]] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=mock_streaming_llm_model(["Học phí ... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    await run_graph(_fee_input(), models, _sink([]), _status_trace(seen), _usage_recorder())

    assert [stage for stage, _url in seen] == ["thinking", "analyzing", "retrieving", "generating"]


@pytest.mark.parametrize(
    ("topic", "mode"), [("quy_che_thu_tuc", "speculative"), ("hoc_phi", "fallback")]
)
@pytest.mark.asyncio
async def test_no_speculative_search_for_a_stable_topic_or_in_fallback_mode(
    topic: str,
    mode: str,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    calls = _fake_slot_web_search(monkeypatch, [_WEB_PAGE])
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MODE", mode)
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model, topic),
        query_transformation=mock_sync_llm_model("HyDE"),
        generation=mock_streaming_llm_model(["... [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
    )

    await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert calls == []


@pytest.mark.asyncio
async def test_official_page_without_the_figures_is_linked_by_the_ticket_fallback(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)
    # Hybrid: with an LLM rerank ahead the graph applies the dense floor instead, so
    # raise it too - these turns must reach rerank with no usable chunk.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_DENSE_FLOOR", 1.1)
    _fake_slot_web_search(monkeypatch, [_FEE_DECISION])
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model(
            '{"SQ1": {"W1": {"reason": "QĐ mức học phí 2025-2026, chỉ có tiêu đề",'
            ' "answer": false}}}'
        ),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert result.used_ticket_fallback is True
    assert result.used_web_search is False
    (prompt,) = seen_prompts
    assert "<web_sources>" in prompt
    assert f"{_FEE_DECISION.url}; năm học 2025-2026" in prompt
    assert [w.code for w in result.admin_warnings] == ["WEB_SOURCE_USED"]
    assert "09c_WebJudgeNode" in _traced_nodes(caplog)


@pytest.mark.asyncio
async def test_pages_kept_only_as_links_are_listed_after_an_answer(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One page answers, another is an official document without the figures: the
    second one is linked after the answer rather than lost."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)
    # Hybrid: with an LLM rerank ahead the graph applies the dense floor instead, so
    # raise it too - these turns must reach rerank with no usable chunk.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_DENSE_FLOOR", 1.1)
    _fake_slot_web_search(monkeypatch, [_WEB_PAGE, _FEE_DECISION])
    streamed: list[str] = []
    models = GraphModels(
        classification=_web_slot_classification(mock_sync_llm_model),
        query_transformation=mock_sync_llm_model("HyDE học phí"),
        generation=mock_streaming_llm_model(["Lịch thi bắt đầu 05/01 [1]."]),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model(
            '{"SQ1": {"W1": {"reason": "lịch thi 05/01", "answer": true},'
            ' "W2": {"reason": "QĐ học phí, chỉ có tiêu đề", "answer": false}}}'
        ),
    )

    result = await run_graph(_fee_input(), models, _sink(streamed), _TRACE, _usage_recorder())

    assert result.response_text == (
        "Lịch thi bắt đầu 05/01 [1].\n\nVăn bản liên quan trên website của Trường:\n"
        f"- [1035-QD-DHCN]({_FEE_DECISION.url}) · ptckt.iuh.edu.vn · năm học 2025-2026"
    )
    assert "".join(streamed) == result.response_text
    assert [c["url"] for c in result.citations] == [_WEB_PAGE.url]


_FEE_Q = "Học phí ngành CNTT năm học 2026-2027 là bao nhiêu?"
_EXAM_Q = "Lịch thi chứng chỉ ứng dụng CNTT đợt gần nhất năm 2026 là khi nào?"
_FEE_PAGE = WebSearchResult(
    title="Quyết định mức thu học phí năm học 2026-2027",
    url="https://ptckt.iuh.edu.vn/hoc-phi-2026-2027.pdf",
    content="Khối Công nghệ: 1.050.000 đồng/tín chỉ",
    score=0.3,
)
_EXAM_PAGE = WebSearchResult(
    title="Thông báo lịch thi chứng chỉ",
    url="https://itc.iuh.edu.vn/lich-thi-05-06-11-2026",
    content="Lịch thi và danh sách thi ngày 05,06/11/2026",
    score=0.5,
)


@pytest.mark.asyncio
async def test_each_multi_sub_query_is_searched_and_judged_on_its_own_pages(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 14-10-2026 turn: "học phí ... và lịch thi chứng chỉ" searched as one keyword
    query found only fee pages; the exam question had only 2022 results in the knowledge
    base. Each sub-query now has its own search and its own judge call."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MODE", "speculative")
    searches: list[list[str]] = []

    async def search_web(requests: Sequence[WebSearchRequest]) -> WebSearchOutcome:
        searches.append([request.query for request in requests])
        per_request = [
            [_EXAM_PAGE] if "lịch thi" in request.query else [_FEE_PAGE] for request in requests
        ]
        return WebSearchOutcome(
            results=[page for pages in per_request for page in pages], per_request=per_request
        )

    monkeypatch.setattr("app.graph.streaming_graph.search_web", search_web)
    judged: list[tuple[str, str]] = []

    def judge(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompt = str(getattr(messages[-1].parts[-1], "content", ""))
        question = _EXAM_Q if _EXAM_Q in prompt else _FEE_Q
        page = _EXAM_PAGE if question == _EXAM_Q else _FEE_PAGE
        judged.append((question, page.url if page.url in prompt else "-"))
        return ModelResponse(
            parts=[TextPart(content='{"SQ1": {"W1": {"reason": "có số liệu", "answer": true}}}')]
        )

    classification = {
        "tasks": [
            {
                "intent": "academic_advisory",
                "query": "học phí cntt năm nay và lịch thi chứng chỉ cntt gần nhất?",
                "routing_mode": "MULTI",
                "sub_queries": [_FEE_Q, _EXAM_Q],
                "web": [
                    {"topics": ["hoc_phi"], "subject": "mức thu học phí", "nam_hoc": "2026-2027"},
                    {
                        "topics": ["lich_thong_bao"],
                        "subject": "lịch thi chứng chỉ ứng dụng CNTT",
                        "nam_hoc": "2026",
                    },
                ],
            }
        ],
        "confidence": 0.9,
    }
    seen_prompts: list[str] = []
    old_results = RetrievedChunk(
        chunk_id="exam-2022",
        content="Kết quả thi chứng chỉ ứng dụng CNTT ngày 12/3/2022",
        source="ket-qua-thi-2022.pdf",
        score=0.9,
    )
    models = GraphModels(
        classification=mock_sync_llm_model(json.dumps(classification, ensure_ascii=False)),
        query_transformation=mock_sync_llm_model("unused"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=_PerQueryRetrieval({_FEE_Q: [], _EXAM_Q: [old_results]}),
        rerank=FunctionModel(function=judge),
    )

    result = await run_graph(_fee_input(), models, _sink([]), _TRACE, _usage_recorder())

    # One speculative search, one request per sub-query, by its own slots.
    assert searches == [
        ["mức thu học phí năm học 2026-2027 IUH", "lịch thi chứng chỉ ứng dụng CNTT năm 2026 IUH"]
    ]
    # One judge call per sub-query, each on its own page.
    assert sorted(judged) == sorted([(_FEE_Q, _FEE_PAGE.url), (_EXAM_Q, _EXAM_PAGE.url)])
    assert result.used_ticket_fallback is False
    (prompt,) = seen_prompts
    assert "ket-qua-thi-2022.pdf" in prompt
    assert _FEE_PAGE.url in prompt and _EXAM_PAGE.url in prompt


def test_a_postgraduate_question_keeps_postgraduate_pages() -> None:
    masters = WebSlots(topics=["hoc_phi"], subject="mức thu học phí", bac_dao_tao="Thạc sĩ")
    undergraduate = WebSlots(topics=["hoc_phi"], subject="mức thu học phí")

    assert _web_request(masters, "Học phí thạc sĩ?").postgraduate is True
    assert _web_request(undergraduate, "Học phí ngành CNTT?").postgraduate is False
    assert _web_request(None, "Học phí cao học bao nhiêu?").postgraduate is True


# ── LLMRerankNode (09a, RERANK model) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_keyword_only_chunk_is_judged_irrelevant_so_the_web_is_searched(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    # The dummy chunk scores 0.9, which the rescue would restore.
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_RESCUE_KEEP", 0)
    calls = _fake_web_search(monkeypatch, [_WEB_PAGE])
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("Lịch thi học kỳ 1"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model('{"SQ1": []}'),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    assert "09a_LLMRerankNode" in _traced_nodes(caplog)
    assert calls == [["Lịch thi học kỳ 1"]]
    (prompt,) = seen_prompts
    assert "dummy retrieved content" not in prompt
    assert result.used_web_search is True


@pytest.mark.asyncio
async def test_missing_extraction_model_skips_the_llm_rerank_and_warns_admins(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    models = _models(mock_sync_llm_model, mock_streaming_llm_model)
    models.rerank_unavailable = "Mô hình Extraction chưa được cấu hình."

    result = await run_graph(_single_question_input(), models, _sink([]), _TRACE, _usage_recorder())

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    (warning,) = result.admin_warnings
    assert warning.code == "LLM_RERANK_UNAVAILABLE"
    assert warning.message.endswith("Mô hình Extraction chưa được cấu hình.")


# ── CrossEncoderRerankNode (09a, CHAT_RERANK_MODE) ─────────────────────────


class _FakeCrossEncoderClient:
    """Stands in for `CrossEncoderClient()` in the graph: scores every passage 1.0, or
    raises when the service is "down"."""

    down = False
    calls: ClassVar[list[tuple[str, int]]] = []

    async def rerank(self, query: str, passages: list[str], *, top_k: int | None = None):
        from app.integrations.cross_encoder_client import (
            CrossEncoderScores,
            CrossEncoderUnavailableError,
        )

        _FakeCrossEncoderClient.calls.append((query, len(passages)))
        if _FakeCrossEncoderClient.down:
            raise CrossEncoderUnavailableError("Không kết nối được tới service cross-encoder.")
        return CrossEncoderScores(
            scores=[1.0] * len(passages),
            ranking=list(range(len(passages))),
            truncated=0,
            took_ms=1,
            model="fake",
        )


def _cross_encoder_mode(monkeypatch: pytest.MonkeyPatch, mode: str, *, down: bool = False) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_RERANK_MODE", mode)
    monkeypatch.setattr("app.graph.streaming_graph.CrossEncoderClient", _FakeCrossEncoderClient)
    _FakeCrossEncoderClient.down = down
    _FakeCrossEncoderClient.calls = []


@pytest.mark.asyncio
async def test_cross_encoder_mode_ranks_with_the_service_and_skips_the_llm_rerank(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _cross_encoder_mode(monkeypatch, "cross_encoder")
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("Lịch thi học kỳ 1"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        # Would drop the chunk if it ran.
        rerank=mock_sync_llm_model('{"SQ1": []}'),
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    nodes = _traced_nodes(caplog)
    assert "09a_CrossEncoderRerankNode" in nodes and "09a_LLMRerankNode" not in nodes
    assert _FakeCrossEncoderClient.calls == [("Lịch thi học kỳ 1", 1)]
    (prompt,) = seen_prompts
    assert "dummy retrieved content" in prompt
    assert result.used_ticket_fallback is False
    assert [w.code for w in result.admin_warnings] == []


@pytest.mark.asyncio
async def test_two_stage_mode_runs_the_llm_rerank_after_the_cross_encoder(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _cross_encoder_mode(monkeypatch, "cross_encoder_then_llm")
    models = _models(mock_sync_llm_model, mock_streaming_llm_model)
    models.rerank = mock_sync_llm_model('{"SQ1": {"C1": "đúng lịch thi"}}')

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    nodes = _traced_nodes(caplog)
    assert nodes.index("09a_CrossEncoderRerankNode") < nodes.index("09a_LLMRerankNode")
    assert result.response_text == "Câu trả lời cuối cùng [1]."


@pytest.mark.asyncio
async def test_a_dead_service_falls_over_to_the_llm_rerank_and_warns_admins(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _cross_encoder_mode(monkeypatch, "cross_encoder", down=True)
    models = _models(mock_sync_llm_model, mock_streaming_llm_model)
    models.rerank = mock_sync_llm_model('{"SQ1": {"C1": "đúng lịch thi"}}')

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    nodes = _traced_nodes(caplog)
    assert "09a_CrossEncoderRerankNode" in nodes and "09a_LLMRerankNode" in nodes
    assert result.response_text == "Câu trả lời cuối cùng [1]."
    (warning,) = result.admin_warnings
    assert warning.code == "CROSS_ENCODER_FAILED"
    assert warning.message.startswith("Bỏ qua bước xếp hạng cross-encoder")


@pytest.mark.asyncio
async def test_none_mode_runs_neither_rerank_and_keeps_the_retrieval_order(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _cross_encoder_mode(monkeypatch, "none")
    seen_prompts: list[str] = []
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("Lịch thi học kỳ 1"),
        generation=_capturing_generation_model(seen_prompts),
        retrieval=FakeRetrievalService([_DUMMY_CHUNK]),
        rerank=mock_sync_llm_model('{"SQ1": []}'),  # would drop the chunk if it ran
    )

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _single_question_input(), models, _sink([]), _TRACE, _usage_recorder()
        )

    nodes = _traced_nodes(caplog)
    assert "09_PostRetrievalRerankNode" in nodes
    assert "09a_CrossEncoderRerankNode" not in nodes and "09a_LLMRerankNode" not in nodes
    assert _FakeCrossEncoderClient.calls == []
    (prompt,) = seen_prompts
    assert "dummy retrieved content" in prompt
    assert result.admin_warnings == []


# --- clarification panel (UNISAGE-99) ----------------------------------------------

_FORM = {
    "type": "ask_user_form",
    "fields": [
        {
            "field": "training_type",
            "label": "Hệ đào tạo",
            "options": [
                {"id": "chinh_quy", "label": "Chính quy"},
                {"id": "lien_thong", "label": "Liên thông"},
            ],
        }
    ],
}
_FORM_JSON = f"```json\n{json.dumps(_FORM, ensure_ascii=False)}\n```"


def _advisory_round(origin_tasks: list[ClassifiedTask], *, chain_depth: int = 1) -> PendingRound:
    pending = build_round(
        [
            TaskQuestions(
                task=advisory_task("T1", origin_tasks),
                questions=advisory_questions(
                    [
                        {
                            "type": "ask_user_form",
                            "fields": [
                                {
                                    "field": "training_type",
                                    "label": "Hệ đào tạo",
                                    "options": [
                                        {"id": "chinh_quy", "label": "Chính quy"},
                                        {"id": "lien_thong", "label": "Liên thông"},
                                    ],
                                }
                            ],
                        }
                    ],
                    confirmed_metadata={},
                ),
            )
        ],
        original_query="học phí của ngành tôi học",
        chain_depth=chain_depth,
    )
    assert pending is not None
    return pending


def _resume(pending: PendingRound, option_id: str = "chinh_quy") -> ResumeInput:
    submit = ClarificationSubmit(
        action="submit",
        panel_id=pending.panel.panel_id,
        answers=[Answer(question_id="q1", option_id=option_id)],
    )
    return ResumeInput(pending_round=pending, answers=validate_answers(pending.panel, submit))


@pytest.mark.asyncio
async def test_advisory_answer_asking_for_more_raises_a_panel(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    models = _models(mock_sync_llm_model, mock_streaming_llm_model)
    models.generation = mock_streaming_llm_model(
        ["Học phí tuỳ hệ đào tạo. Bạn học hệ nào?\n\n", _FORM_JSON]
    )
    tokens: list[str] = []
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="học phí của ngành tôi học",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    result = await run_graph(graph_input, models, _sink(tokens), _TRACE, _usage_recorder())

    assert "ask_user_form" not in "".join(tokens)
    assert result.pending_round is not None
    question = result.pending_round.panel.questions[0]
    assert (question.id, question.kind, question.field, question.task_id) == (
        "q1",
        "choice",
        "he_dao_tao",  # the form's "training_type", under its registry name
        "T1",
    )
    assert result.pending_round.original_query == "học phí của ngành tôi học"


@pytest.mark.asyncio
async def test_resume_skips_classification_and_retrieves_on_the_original_question(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrievalService([_DUMMY_CHUNK])
    models = GraphModels(
        classification=mock_sync_llm_model("off_topic"),  # must never be reached
        query_transformation=_echo_query_transformation_model(),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=retrieval,
    )
    pending = _advisory_round(
        [
            ClassifiedTask(
                intent="academic_advisory", query="học phí của ngành tôi học", routing_mode="SINGLE"
            )
        ]
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Hệ đào tạo: Chính quy",
        is_first_turn=False,
        security=AcademicSecurityContext(),
        resume=_resume(pending),
    )

    result = await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert result.confirmed_metadata == {"he_dao_tao": "chinh_quy"}
    assert len(retrieval.queries) == 1
    assert retrieval.queries[0].startswith("học phí của ngành tôi học")
    assert "Hệ đào tạo: Chính quy" not in retrieval.queries[0]
    assert result.pending_round is None


@pytest.mark.asyncio
async def test_resume_reruns_every_origin_task(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrievalService([_DUMMY_CHUNK])
    models = GraphModels(
        classification=mock_sync_llm_model("off_topic"),
        query_transformation=_echo_query_transformation_model(),
        generation=mock_streaming_llm_model(["Trả lời [1]."]),
        retrieval=retrieval,
    )
    tasks = [
        ClassifiedTask(
            intent="academic_advisory", query="học phí ngành CNTT", routing_mode="SINGLE"
        ),
        ClassifiedTask(
            intent="academic_advisory", query="điều kiện học bổng", routing_mode="SINGLE"
        ),
    ]
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="x",
        is_first_turn=False,
        security=AcademicSecurityContext(),
        resume=_resume(_advisory_round(tasks)),
    )

    await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    assert len(retrieval.queries) == 2
    assert retrieval.queries[0].startswith("học phí ngành CNTT")
    assert retrieval.queries[1].startswith("điều kiện học bổng")


@pytest.mark.asyncio
async def test_resume_keeps_asking_new_fields_but_never_an_answered_one(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    task = [ClassifiedTask(intent="academic_advisory", query="học phí", routing_mode="SINGLE")]

    def run(form: str, depth: int) -> GraphInput:
        return GraphInput(
            conversation_id="c1",
            user_message="x",
            is_first_turn=False,
            security=AcademicSecurityContext(),
            resume=_resume(_advisory_round(task, chain_depth=depth)),
        )

    def models(form: str) -> GraphModels:
        built = _models(mock_sync_llm_model, mock_streaming_llm_model)
        built.generation = mock_streaming_llm_model(["Bạn cho mình biết thêm nhé?\n", form])
        return built

    # A new field after the 5th panel in a row is still asked - there is no chain limit.
    new_field = _FORM_JSON.replace("training_type", "cohort").replace("Hệ đào tạo", "Khoá")
    later = await run_graph(
        run(new_field, 5), models(new_field), _sink([]), _TRACE, _usage_recorder()
    )
    assert later.pending_round is not None and later.pending_round.chain_depth == 6
    assert [q.field for q in later.pending_round.panel.questions] == ["khoa_nhap_hoc"]

    # Asking again for the field just answered (training_type = he_dao_tao) never opens a panel.
    again = await run_graph(
        run(_FORM_JSON, 1), models(_FORM_JSON), _sink([]), _TRACE, _usage_recorder()
    )
    assert again.pending_round is None
    assert again.confirmed_metadata == {"he_dao_tao": "chinh_quy"}


def _capturing_generation(prompts: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        prompts.append(
            "\n".join(
                str(getattr(part, "content", "")) for message in messages for part in message.parts
            )
        )
        yield "Trả lời [1]."

    return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
async def test_a_panel_about_someone_else_answers_that_question_and_saves_nothing(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lecturer asked about a student's fees and answered "Liên thông" for the student:
    asked even though the lecturer's own profile has a mode, kept out of that profile,
    and given to generation as part of the question."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    own_profile = {"he_dao_tao": "chinh_quy", "hoc_vi": "Tiến sĩ"}
    question = "học phí của sinh viên lớp tôi chủ nhiệm"
    pending = build_round(
        [
            TaskQuestions(
                task=advisory_task(
                    "T1",
                    [
                        ClassifiedTask(
                            intent="academic_advisory", query=question, routing_mode="SINGLE"
                        )
                    ],
                ),
                questions=advisory_questions(
                    [{**_FORM, "about": "other"}], confirmed_metadata=own_profile
                ),
            )
        ],
        original_query=question,
        chain_depth=1,
    )
    assert pending is not None and pending.panel.questions[0].about == "other"
    assert "about" not in pending.panel.questions[0].public().model_dump()
    prompts: list[str] = []
    models = GraphModels(
        classification=mock_sync_llm_model("off_topic"),
        query_transformation=_echo_query_transformation_model(),
        generation=_capturing_generation(prompts),
        retrieval=_RecordingRetrievalService([_DUMMY_CHUNK]),
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Hệ đào tạo: Liên thông",
        is_first_turn=False,
        security=AcademicSecurityContext(user_id="u1", role="GIANG_VIEN"),
        confirmed_metadata=dict(own_profile),
        resume=_resume(pending, "lien_thong"),
    )

    result = await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    assert result.confirmed_metadata == own_profile
    assert "Về trường hợp được hỏi, người dùng vừa cho biết: Hệ đào tạo: Liên thông" in prompts[-1]


def test_a_form_about_the_caller_skips_what_their_profile_has() -> None:
    profile = {"he_dao_tao": "chinh_quy"}

    assert advisory_questions([_FORM], confirmed_metadata=profile) == []
    (asked,) = advisory_questions([{**_FORM, "about": "other"}], confirmed_metadata=profile)
    assert (asked["field"], asked["about"]) == ("he_dao_tao", "other")
