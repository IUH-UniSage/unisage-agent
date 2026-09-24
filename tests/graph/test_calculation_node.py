"""CalculationNode placeholder, alone and merged with the advisory branch."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.core.graph_trace import GraphTrace
from app.graph.nodes.calculation import CALCULATION_PLACEHOLDER_TEMPLATE
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import make_classification_llm_model

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)
_CHUNK = RetrievedChunk(
    chunk_id="c1", content="Quy trình đăng ký tốt nghiệp", source="s", score=0.9
)

_MIXED_MESSAGE = "Tính giúp điểm GPA cho mình và cho mình biết thủ tục đăng ký tốt nghiệp"
_CALCULATION_QUERY = "Tính giúp điểm GPA cho mình"
_ADVISORY_QUERY = "Thủ tục đăng ký tốt nghiệp là gì?"

_ASK_FORM_ANSWER = (
    "Thủ tục đăng ký tốt nghiệp khác nhau theo hệ đào tạo [1].\n\n"
    "Bạn đang học theo hệ đào tạo nào ạ?\n\n"
    "```json\n"
    '{"type": "ask_user_form", "fields": [{"field": "training_type", "options": '
    '[{"id": "chinh_quy"}, {"id": "lien_thong"}]}]}\n'
    "```"
)


def _mixed_classification_model() -> FunctionModel:
    payload = {
        "tasks": [
            {"intent": "academic_calculation", "query": _CALCULATION_QUERY, "routing_mode": None},
            {"intent": "academic_advisory", "query": _ADVISORY_QUERY, "routing_mode": "SINGLE"},
        ],
        "confidence": 0.9,
    }

    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload, ensure_ascii=False))])

    return FunctionModel(function=function)


def _echo_model() -> FunctionModel:
    """Returns its input verbatim, so the retrieval query shows what
    QueryTransformationNode was given."""

    def echo(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        text = ""
        for part in messages[-1].parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                text = content
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=echo)


@dataclass
class _RecordingRetrieval:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del security, limit
        self.queries.append(query)
        return self.chunks


def _input(message: str, **kwargs: object) -> GraphInput:
    return GraphInput(
        conversation_id="c1",
        user_message=message,
        is_first_turn=False,
        security=AcademicSecurityContext(),
        **kwargs,  # type: ignore[arg-type]
    )


def _sink(target: list[str]) -> TokenSink:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


def _traced_nodes(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage().split(" ", 1)[0].removeprefix("node=")
        for record in caplog.records
        if record.name == "unisage.graph" and record.getMessage().startswith("node=")
    ]


@pytest.mark.asyncio
async def test_calculation_only_turn_returns_placeholder_without_retrieval(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=make_classification_llm_model("academic_calculation"),
        query_transformation=mock_sync_llm_model("unused"),
        generation=mock_streaming_llm_model(["unused"]),
        retrieval=retrieval,
    )
    tokens: list[str] = []

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _input("Tính GPA giúp em", confirmed_metadata={"he_dao_tao": "chinh_quy"}),
            models,
            _sink(tokens),
            _TRACE,
        )

    assert result.response_text == CALCULATION_PLACEHOLDER_TEMPLATE
    assert "".join(tokens) == CALCULATION_PLACEHOLDER_TEMPLATE
    assert retrieval.queries == []
    assert result.confirmed_metadata == {"he_dao_tao": "chinh_quy"}
    assert result.citations == []
    nodes = _traced_nodes(caplog)
    assert nodes[-1] == "07_CalculationNode"
    assert "06_QueryTransformationNode" not in nodes


@pytest.mark.asyncio
async def test_calculation_plus_procedure_answers_both_parts(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["Thủ tục gồm 3 bước ", "[1]."]),
        retrieval=retrieval,
    )
    tokens: list[str] = []

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(_input(_MIXED_MESSAGE), models, _sink(tokens), _TRACE)

    expected = f"Thủ tục gồm 3 bước [1].\n\n{CALCULATION_PLACEHOLDER_TEMPLATE}"
    assert result.response_text == expected
    assert "".join(tokens) == expected
    # Citations come from the advisory answer only - the placeholder has no [n].
    assert len(result.citations) == 1
    assert _traced_nodes(caplog)[-5:] == [
        "06_QueryTransformationNode",
        "08_RetrievalFilteringNode",
        "09_PostRetrievalRerankNode",
        "10_GenerationSynthesisNode",
        "07_CalculationNode",
    ]


@pytest.mark.asyncio
async def test_mixed_turn_retrieves_on_the_advisory_question_only(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["ok"]),
        retrieval=retrieval,
    )

    await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE)

    (query,) = retrieval.queries
    assert query.startswith(_ADVISORY_QUERY)
    assert "GPA" not in query


@pytest.mark.asyncio
async def test_mixed_turn_keeps_the_clarification_raised_by_the_advisory_part(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model([_ASK_FORM_ANSWER]),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )

    result = await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE)

    assert result.pending_clarification is not None
    assert result.pending_clarification.missing_fields == ["training_type"]
    # Resume must retrieve on the advisory question, not the GPA part.
    assert result.pending_clarification.original_query == _ADVISORY_QUERY
    assert result.response_text.endswith(CALCULATION_PLACEHOLDER_TEMPLATE)


@pytest.mark.asyncio
async def test_mixed_turn_without_context_falls_back_then_appends_the_placeholder(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["unused"]),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )

    result = await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE)

    assert result.used_ticket_fallback is True
    assert result.response_text.endswith(f"\n\n{CALCULATION_PLACEHOLDER_TEMPLATE}")
    assert "chưa tìm thấy" in result.response_text.lower()


def _two_advisory_classification_model() -> FunctionModel:
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

    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload, ensure_ascii=False))])

    return FunctionModel(function=function)


def _fixed_hyde_model() -> FunctionModel:
    """HyDE double whose standalone-question line differs from the student's
    message, so a single-task turn visibly carries a `resolved_query`."""

    def function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompt = str(getattr(messages[-1].parts[-1], "content", ""))
        first_line = prompt.splitlines()[0]
        return ModelResponse(
            parts=[TextPart(content=f"Câu hỏi độc lập: {first_line}\n\nVăn bản HyDE.")]
        )

    return FunctionModel(function=function)


def _prompt_capturing_generation(seen: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        seen.append(str(getattr(messages[-1].parts[-1], "content", "")))
        yield "ok [1]."

    return FunctionModel(stream_function=stream)


_RESOLVED_MARKER = "Nguyên văn người dùng vừa nhắn ở lượt này"


@pytest.mark.asyncio
async def test_two_advisory_questions_run_node_06_per_task_and_search_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrieval([_CHUNK])
    generation_prompts: list[str] = []
    message = "Học phí ngành CNTT bao nhiêu, với lại điều kiện học bổng là gì?"
    models = GraphModels(
        classification=_two_advisory_classification_model(),
        query_transformation=_fixed_hyde_model(),
        generation=_prompt_capturing_generation(generation_prompts),
        retrieval=retrieval,
    )

    await run_graph(_input(message), models, _sink([]), _TRACE)

    assert retrieval.queries == [
        "Câu hỏi độc lập: Học phí ngành CNTT bao nhiêu?\n\nVăn bản HyDE.",
        "Câu hỏi độc lập: Điều kiện học bổng là gì?\n\nVăn bản HyDE.",
    ]
    # Two questions → no standalone rewrite; the student's message is answered.
    (prompt,) = generation_prompts
    assert _RESOLVED_MARKER not in prompt
    assert message in prompt


@pytest.mark.asyncio
async def test_single_advisory_question_still_passes_its_resolved_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    generation_prompts: list[str] = []
    message = "còn khóa 2024 thì sao?"
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=_fixed_hyde_model(),
        generation=_prompt_capturing_generation(generation_prompts),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )

    await run_graph(_input(message), models, _sink([]), _TRACE)

    (prompt,) = generation_prompts
    assert f"Câu hỏi độc lập: {message}" in prompt
    assert _RESOLVED_MARKER in prompt
