"""CalculationNode placeholder, alone and merged with the advisory branch."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.calculation.formulas import param_spec
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.calculation_turn import NEEDS_INPUT_LEAD
from app.graph.clarification_answers import validate_answers
from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.graph.nodes.calculation import question_for
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels, ResumeInput
from app.schemas.clarification import (
    Answer,
    CalculationPlan,
    ClarificationSubmit,
    PendingCalculationTask,
)
from app.schemas.intent import ClassifiedTask
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import RetrieveManyMixin, make_classification_llm_model

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)


def _usage_recorder() -> UsageRecorder:
    # A fresh instance per call, not a shared module-level constant like _TRACE -
    # UsageRecorder is stateful (closes exactly once), so sharing one across tests
    # would leak `_closed`/`_lines` state between them.
    return UsageRecorder(request_id="test-request", purpose="CHAT")


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
class _RecordingRetrieval(RetrieveManyMixin):
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

    await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE, _usage_recorder())

    (query,) = retrieval.queries
    assert query.startswith(_ADVISORY_QUERY)
    assert "GPA" not in query


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

    await run_graph(_input(message), models, _sink([]), _TRACE, _usage_recorder())

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

    await run_graph(_input(message), models, _sink([]), _TRACE, _usage_recorder())

    (prompt,) = generation_prompts
    assert f"Câu hỏi độc lập: {message}" in prompt
    assert _RESOLVED_MARKER in prompt


def _scripted(*outputs: object) -> FunctionModel:
    """Non-streamed calls answered in order (classification, extractor, ...)."""

    remaining = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs]

    def respond(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=remaining.pop(0))])

    return FunctionModel(function=respond)


def _note_model(note: str) -> FunctionModel:
    def respond(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=note)])

    return FunctionModel(function=respond)


_COURSE_QUERY = (
    "Môn 2 tín lý thuyết 1 tín thực hành, TX 8 GK 7 CK 6.5, TH 9 và 8 thì tổng kết bao nhiêu?"
)
_COURSE_TASK = {
    "tasks": [{"intent": "academic_calculation", "query": _COURSE_QUERY}],
    "confidence": 0.9,
}
_COURSE_PARAMS = {
    "formula_id": "course_score",
    "params": {"tclt": 2, "tcth": 1, "tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8]},
}


@pytest.mark.asyncio
async def test_calculation_only_turn_with_every_number_shows_steps_and_a_checked_note(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=_note_model("Điểm chữ B nghĩa là bạn đã qua học phần."),
        retrieval=retrieval,
    )
    tokens: list[str] = []

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _input(_COURSE_QUERY), models, _sink(tokens), _TRACE, _usage_recorder()
        )

    assert "".join(tokens) == result.response_text
    assert "Kết quả: ĐTKHP **7.5** · Điểm chữ **B** · Thang 4 **3.0**" in result.response_text
    assert result.response_text.endswith("Điểm chữ B nghĩa là bạn đã qua học phần.")
    assert result.pending_round is None
    assert retrieval.queries == []
    assert result.calculation_items[0]["status"] == "computed"
    assert result.calculation_items[0]["result_summary"] == "ĐTKHP: 7.5; Điểm chữ: B; Thang 4: 3.0"
    assert "question_raw" not in result.calculation_items[0]
    assert result.calculation_traces[0]["trace"]["question_raw"] == _COURSE_QUERY
    assert "06_QueryTransformationNode" not in _traced_nodes(caplog)


@pytest.mark.asyncio
async def test_note_with_an_invented_number_is_replaced() -> None:
    models = GraphModels(
        classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=_note_model("Bạn cần thêm 0.8 điểm để lên B+."),
        retrieval=_RecordingRetrieval([]),
    )
    result = await run_graph(_input(_COURSE_QUERY), models, _sink([]), _TRACE, _usage_recorder())
    assert "0.8" not in result.response_text
    assert result.response_text.endswith("Kết quả: ĐTKHP 7.5, Điểm chữ B, Thang 4 3.0.")


@pytest.mark.asyncio
async def test_calculation_only_turn_missing_numbers_asks_on_the_panel_without_any_llm_answer(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=make_classification_llm_model("academic_calculation"),
        query_transformation=mock_sync_llm_model("unused"),
        generation=_note_model("must not be called"),
        retrieval=retrieval,
    )
    tokens: list[str] = []

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _input("Tính GPA giúp em", confirmed_metadata={"he_dao_tao": "chinh_quy"}),
            models,
            _sink(tokens),
            _TRACE,
            _usage_recorder(),
        )

    assert result.response_text == NEEDS_INPUT_LEAD == "".join(tokens)
    assert result.pending_round is not None
    (question,) = result.pending_round.panel.questions
    assert (question.kind, question.field, question.task_id) == ("course_table", "courses", "T1")
    assert retrieval.queries == []
    assert result.confirmed_metadata == {"he_dao_tao": "chinh_quy"}
    assert result.calculation_items[0]["status"] == "needs_input"
    assert "07_CalculationNode" in _traced_nodes(caplog)


@pytest.mark.asyncio
async def test_mixed_turn_streams_the_calculation_before_the_advisory_answer(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["Thủ tục gồm 3 bước ", "[1]."]),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )
    tokens: list[str] = []

    result = await run_graph(
        _input(_MIXED_MESSAGE), models, _sink(tokens), _TRACE, _usage_recorder()
    )

    assert tokens[0] == f"{NEEDS_INPUT_LEAD}\n\n"
    assert result.response_text == f"{NEEDS_INPUT_LEAD}\n\nThủ tục gồm 3 bước [1]."
    assert "".join(tokens) == result.response_text
    assert len(result.citations) == 1


@pytest.mark.asyncio
async def test_mixed_turn_asks_both_parts_on_one_panel(
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

    result = await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE, _usage_recorder())

    assert result.pending_round is not None
    questions = result.pending_round.panel.questions
    assert [(q.id, q.field, q.origin, q.task_id) for q in questions] == [
        ("q1", "courses", "calculation", "T1"),
        ("q2", "training_type", "advisory", "T2"),
    ]
    # Resume must retrieve on the advisory question, not the GPA part.
    assert result.pending_round.original_query == _ADVISORY_QUERY
    assert "ask_user_form" not in result.response_text


@pytest.mark.asyncio
async def test_mixed_turn_without_context_still_shows_the_calculation_first(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 1.1)  # nothing can pass
    models = GraphModels(
        classification=_mixed_classification_model(),
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["Chưa tìm thấy quy định phù hợp."]),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )

    result = await run_graph(_input(_MIXED_MESSAGE), models, _sink([]), _TRACE, _usage_recorder())

    assert result.used_ticket_fallback is True
    assert result.response_text.startswith(NEEDS_INPUT_LEAD)
    assert "chưa tìm thấy" in result.response_text.lower()


def _resume_round(*, chain_depth: int = 1, with_advisory: bool = False) -> ResumeInput:
    calc_task = PendingCalculationTask(
        task_id="T1",
        query=_COURSE_QUERY,
        plan=CalculationPlan(formula_id="course_score"),
        known_params={"tclt": 2, "tcth": 1, "tbtx": 8, "gk": 7, "ck": 6.5},
    )
    parts = [
        TaskQuestions(
            task=calc_task,
            questions=[question_for(param_spec("course_score", "th"))],  # type: ignore[arg-type]
        )
    ]
    if with_advisory:
        parts.append(
            TaskQuestions(
                task=advisory_task(
                    "T2",
                    [
                        ClassifiedTask(
                            intent="academic_advisory", query=_ADVISORY_QUERY, routing_mode="SINGLE"
                        )
                    ],
                ),
                questions=advisory_questions(
                    [json.loads(_ASK_FORM_ANSWER.split("```json\n")[1].split("\n```")[0])],
                    confirmed_metadata={},
                ),
            )
        )
    pending = build_round(parts, original_query=_ADVISORY_QUERY, chain_depth=chain_depth)
    assert pending is not None
    answers = [Answer(question_id="q1", numbers=[Decimal("9"), Decimal("8")])]
    if with_advisory:
        answers.append(Answer(question_id="q2", option_id="chinh_quy"))
    submit = ClarificationSubmit(action="submit", panel_id=pending.panel.panel_id, answers=answers)
    return ResumeInput(pending_round=pending, answers=validate_answers(pending.panel, submit))


@pytest.mark.asyncio
async def test_resume_mixed_round_computes_then_answers_the_advisory_question(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=_scripted(),  # any classification/extractor call would fail
        query_transformation=_echo_model(),
        generation=mock_streaming_llm_model(["Thủ tục hệ chính quy gồm 3 bước [1]."]),
        retrieval=retrieval,
    )
    tokens: list[str] = []
    result = await run_graph(
        _input("x", resume=_resume_round(with_advisory=True)),
        models,
        _sink(tokens),
        _TRACE,
        _usage_recorder(),
    )
    assert "ĐTKHP **7.5**" in tokens[0]  # calculation block first
    assert result.response_text.endswith("Thủ tục hệ chính quy gồm 3 bước [1].")
    assert result.confirmed_metadata == {"training_type": "chinh_quy"}
    (query,) = retrieval.queries
    assert query.startswith(_ADVISORY_QUERY)
    assert result.pending_round is None


@pytest.mark.asyncio
async def test_resume_with_a_still_invalid_answer_asks_again_without_limit() -> None:
    models = GraphModels(
        classification=_scripted(),
        query_transformation=_echo_model(),
        generation=_note_model("unused"),
        retrieval=_RecordingRetrieval([]),
    )
    # TCLT + TCTH = 0 is only caught when computing: asked again, at any depth.
    resume = _resume_round()
    task = resume.pending_round.tasks[0]
    assert isinstance(task, PendingCalculationTask)
    broken = task.model_copy(update={"known_params": {**task.known_params, "tclt": 0, "tcth": 0}})
    for depth in (1, 4):
        pending = resume.pending_round.model_copy(update={"tasks": [broken], "chain_depth": depth})
        result = await run_graph(
            _input("x", resume=ResumeInput(pending_round=pending, answers=resume.answers)),
            models,
            _sink([]),
            _TRACE,
            _usage_recorder(),
        )
        assert result.pending_round is not None
        assert result.pending_round.chain_depth == depth + 1
        assert result.pending_round.panel.questions[0].field == "tclt"
