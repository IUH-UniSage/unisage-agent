import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.nodes.greeting import GREETING_TEMPLATE
from app.graph.nodes.intent_routing import SOCIAL_CHAT_TEMPLATE
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATE
from app.graph.nodes.web_search import WebSearchOutcome
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.schemas.clarification import PendingClarification
from app.schemas.intent import ClassifiedTask
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model

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
class _RecordingRetrievalService:
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
        _usage_recorder(),
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
        _usage_recorder(),
    )

    assert result.response_text == OFF_TOPIC_TEMPLATE


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


@pytest.mark.asyncio
async def test_clarification_guard_match_skips_classification_and_resumes_advisory(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
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
        _usage_recorder(),
    )

    assert result.response_text == "Câu trả lời cuối cùng [1]."
    assert result.confirmed_metadata == {"training_type": "chinh_quy"}


@pytest.mark.asyncio
async def test_resuming_clarification_retrieves_using_original_query_not_the_reply(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real bug found via live E2E testing: turn 1 asks
    "học phí của ngành tôi học" -> gets an ask_user_form; turn 2 replies
    "Chính quy ạ" (just data, not a question). Retrieval on turn 2 must still
    search for the ORIGINAL question ("học phí..."), not the reply text -
    otherwise the topic is lost and the model answers nothing relevant (see
    the tasks/report.md write-up of this bug)."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
        original_query="học phí của ngành tôi học",
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Chính quy ạ",  # answers the form - NOT a question itself
        is_first_turn=False,
        security=AcademicSecurityContext(),
        pending_clarification=pending,
    )
    retrieval = _RecordingRetrievalService([_DUMMY_CHUNK])
    models = GraphModels(
        classification=mock_sync_llm_model("off_topic"),  # must never be reached
        query_transformation=_echo_query_transformation_model(),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=retrieval,
    )

    result = await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    # `transform_query` folds `confirmed_metadata` onto the query before
    # retrieval (see `_fold_confirmed_metadata_into_query`), so assert on
    # what matters here: the original topic is the base, and the reply text
    # never appears anywhere in it.
    assert len(retrieval.queries) == 1
    assert retrieval.queries[0].startswith("học phí của ngành tôi học")
    assert "Chính quy ạ" not in retrieval.queries[0]
    assert result.confirmed_metadata == {"training_type": "chinh_quy"}
    # This field was the last one pending, so the round is now fully
    # resolved - exactly the turn `original_query` is easiest to lose (see
    # ClarificationGuardResult.original_query's docstring).
    assert result.pending_clarification is None


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
        "02_SecurityContextExtractionNode_ClarificationGuard",
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


@pytest.mark.asyncio
async def test_resuming_a_single_origin_task_folds_the_reply_into_its_query(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    origin_task = ClassifiedTask(
        intent="academic_advisory", query="học phí của ngành tôi học", routing_mode="SINGLE"
    )
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        original_query="học phí của ngành tôi học",
        origin_tasks=[origin_task],
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Chính quy ạ",
        is_first_turn=False,
        security=AcademicSecurityContext(),
        pending_clarification=pending,
    )
    retrieval = _RecordingRetrievalService([_DUMMY_CHUNK])
    models = GraphModels(
        classification=mock_streaming_llm_model(["unused"]),  # must never be reached
        query_transformation=_echo_query_transformation_model(),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=retrieval,
    )

    await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    (query,) = retrieval.queries
    assert query.startswith("học phí của ngành tôi học")


@pytest.mark.asyncio
async def test_resuming_several_origin_tasks_reruns_each_on_its_own_query(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    origin_tasks = [
        ClassifiedTask(
            intent="academic_advisory", query="Học phí ngành CNTT bao nhiêu?", routing_mode="SINGLE"
        ),
        ClassifiedTask(
            intent="academic_advisory", query="Điều kiện học bổng là gì?", routing_mode="SINGLE"
        ),
    ]
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        original_query="Học phí ngành CNTT bao nhiêu, với lại điều kiện học bổng là gì?",
        origin_tasks=origin_tasks,
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="Chính quy ạ",
        is_first_turn=False,
        security=AcademicSecurityContext(),
        pending_clarification=pending,
    )
    retrieval = _RecordingRetrievalService([_DUMMY_CHUNK])
    models = GraphModels(
        classification=mock_sync_llm_model("off_topic"),  # must never be reached
        query_transformation=_echo_query_transformation_model(),
        generation=mock_streaming_llm_model(["Câu trả lời cuối cùng [1]."]),
        retrieval=retrieval,
    )

    await run_graph(graph_input, models, _sink([]), _TRACE, _usage_recorder())

    assert len(retrieval.queries) == len(origin_tasks)
    for query, task in zip(retrieval.queries, origin_tasks, strict=True):
        assert query.startswith(task.query)


# ── WebSearchNode (between rerank and ticket fallback) ─────────────────────

_WEB_PAGE = WebSearchResult(
    title="Lịch thi HK1",
    url="https://pdt.iuh.edu.vn/lich-thi",
    content="Lịch thi học kỳ 1 bắt đầu ngày 05/01.",
    score=0.8,
)


@dataclass
class _PerQueryRetrieval:
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

    async def search_web(queries: Sequence[str]) -> WebSearchOutcome:
        calls.append(list(queries))
        return WebSearchOutcome(results=pages)

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
