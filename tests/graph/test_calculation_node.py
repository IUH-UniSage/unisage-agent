"""CalculationNode placeholder, alone and merged with the advisory branch."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.calculation.formulas import calculate, param_spec
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.calculation_turn import (
    NEEDS_INPUT_LEAD,
    number_citations,
)
from app.graph.clarification_answers import validate_answers
from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.graph.nodes.calculation import (
    Computed,
    TaskOutcome,
    question_for,
)
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput, ResumeInput
from app.schemas.chat_history import HistoryMessage
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


NEW_RESULT_LINE = (
    "Kết quả: ĐTKHP **7.5** thuộc khoảng [7.0; 8.0) → điểm chữ **B** → thang 4 **3.0**"
)


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


def _review(fits: bool, answer: str, issue: str | None = None) -> str:
    return json.dumps({"fits": fits, "issue": issue, "answer": answer}, ensure_ascii=False)


def _failing_model() -> FunctionModel:
    def respond(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model down")

    return FunctionModel(function=respond)


_COURSE_ANSWER = "Điểm tổng kết học phần của bạn là **7.5**, tương đương điểm chữ **B**."


@pytest.mark.asyncio
async def test_a_fitting_review_answers_first_then_shows_the_python_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retrieval = _RecordingRetrieval([_CHUNK])
    models = GraphModels(
        classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=_note_model(_review(True, _COURSE_ANSWER)),
        retrieval=retrieval,
    )
    tokens: list[str] = []

    with caplog.at_level("INFO", logger="unisage.graph"):
        result = await run_graph(
            _input(_COURSE_QUERY), models, _sink(tokens), _TRACE, _usage_recorder()
        )

    assert "".join(tokens) == result.response_text
    assert result.response_text.startswith(f"{_COURSE_ANSWER}\n\n**Điểm tổng kết học phần")
    assert result.response_text.endswith(NEW_RESULT_LINE)
    assert "Lưu ý" not in result.response_text
    assert result.pending_round is None
    assert retrieval.queries == []
    assert result.calculation_items[0]["status"] == "computed"
    assert result.calculation_items[0]["result_summary"] == "ĐTKHP: 7.5; Điểm chữ: B; Thang 4: 3.0"
    assert "question_raw" not in result.calculation_items[0]
    trace = result.calculation_traces[0]["trace"]
    assert trace["question_raw"] == _COURSE_QUERY
    assert trace["review"] == {"fits": True, "issue": None, "rejected": False}
    assert "chat_calculation_review" in trace["prompt_versions"]
    assert "06_QueryTransformationNode" not in _traced_nodes(caplog)


_TWO_YEARS = "tính giúp tôi học phí ngành công nghệ thông tin 2 năm đầu"
_TWO_YEARS_ISSUE = "Bạn hỏi học phí 2 năm đầu, nhưng phép tính dưới đây là học phí một năm học."
_TWO_YEARS_ANSWER = "Kết quả bên dưới là học phí một năm học: **165.200.000 đồng**."


def _recording(prompts: list[str], *outputs: str) -> FunctionModel:
    """Non-streamed calls answered in order, recording each prompt; raises once empty."""

    remaining = list(outputs)

    def respond(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        if not remaining:
            raise RuntimeError("model down")
        return ModelResponse(parts=[TextPart(content=remaining.pop(0))])

    return FunctionModel(function=respond)


async def _two_years_turn(generation: FunctionModel) -> GraphOutput:
    """A one-year result for a two-year question: the review says it does not fit."""

    return await run_graph(
        _input(_TWO_YEARS),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": _TWO_YEARS}]},
                {"formula_id": "tuition_year", "params": {"tc_lt": 100, "tc_th": 42}},
                _PRICES_FOUND,
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=_RecordingRetrieval([_FEE_TABLE]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )


@pytest.mark.asyncio
async def test_a_mismatch_shows_the_warning_above_the_block() -> None:
    result = await _two_years_turn(_note_model(_review(False, _TWO_YEARS_ANSWER, _TWO_YEARS_ISSUE)))

    assert result.response_text.startswith(
        f"**Lưu ý:** {_TWO_YEARS_ISSUE}\n\n{_TWO_YEARS_ANSWER}\n\n"
    )
    assert "Học phí một năm học **165.200.000 đồng**" in result.response_text
    trace = result.calculation_traces[0]["trace"]
    assert trace["review"] == {"fits": False, "issue": _TWO_YEARS_ISSUE, "rejected": False}
    assert "redone_from" not in trace


@pytest.mark.parametrize(
    "answer",
    [
        "Bạn cần thêm 0.8 điểm để lên B+.",  # a number Python never showed
        "Điểm tổng kết của bạn khoảng 7,47.",  # Python showed 7.5, not this rounding
    ],
)
@pytest.mark.asyncio
async def test_a_review_with_a_number_python_never_showed_leaves_only_the_block(
    answer: str,
) -> None:
    models = GraphModels(
        classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=_note_model(_review(True, answer)),
        retrieval=_RecordingRetrieval([]),
    )
    result = await run_graph(_input(_COURSE_QUERY), models, _sink([]), _TRACE, _usage_recorder())

    assert result.response_text.startswith("**Điểm tổng kết học phần")
    assert answer not in result.response_text
    assert result.calculation_traces[0]["trace"]["review"] == {
        "fits": True,
        "issue": None,
        "rejected": True,
    }


@pytest.mark.parametrize(
    "generation",
    [
        _failing_model(),
        _note_model("Điểm chữ B nghĩa là bạn đã qua học phần."),  # not JSON
        _note_model('{"fits": false, "issue": null, "answer": "Sai rồi."}'),  # no issue
    ],
)
@pytest.mark.asyncio
async def test_a_failed_or_unreadable_review_leaves_only_the_block(
    generation: FunctionModel,
) -> None:
    models = GraphModels(
        classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=generation,
        retrieval=_RecordingRetrieval([]),
    )
    result = await run_graph(_input(_COURSE_QUERY), models, _sink([]), _TRACE, _usage_recorder())

    assert result.response_text.startswith("**Điểm tổng kết học phần")
    assert result.response_text.endswith(NEW_RESULT_LINE)
    assert "review" not in result.calculation_traces[0]["trace"]


@pytest.mark.asyncio
async def test_a_mixed_turn_never_calls_the_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        return ModelResponse(parts=[TextPart(content=_review(True, "x"))])

    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        yield "Thủ tục gồm 3 bước [1]."

    mixed = {
        "tasks": [
            {"intent": "academic_calculation", "query": _COURSE_QUERY, "routing_mode": None},
            {"intent": "academic_advisory", "query": _ADVISORY_QUERY, "routing_mode": "SINGLE"},
        ],
        "confidence": 0.9,
    }
    models = GraphModels(
        classification=_scripted(mixed, _COURSE_PARAMS),
        query_transformation=_echo_model(),
        generation=FunctionModel(function=respond, stream_function=stream),
        retrieval=_RecordingRetrieval([_CHUNK]),
    )

    result = await run_graph(
        _input(f"{_COURSE_QUERY} {_ADVISORY_QUERY}"), models, _sink([]), _TRACE, _usage_recorder()
    )

    assert result.response_text.startswith("**Điểm tổng kết học phần")
    assert result.response_text.endswith("Thủ tục gồm 3 bước [1].")
    assert prompts and not any("<calculation_payload>" in prompt for prompt in prompts)
    assert "review" not in result.calculation_traces[0]["trace"]


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
        ("q2", "he_dao_tao", "advisory", "T2"),  # "training_type" under its registry name
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
            questions=[question_for(param_spec("course_score", "th"), {})],  # type: ignore[arg-type]
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
    assert result.confirmed_metadata == {"he_dao_tao": "chinh_quy"}
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


def _recording_model(*outputs: object) -> tuple[FunctionModel, list[str]]:
    """Non-streamed calls answered in order; records each prompt."""

    remaining = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs]
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompts.append("\n".join(str(getattr(part, "content", "")) for part in messages[-1].parts))
        return ModelResponse(parts=[TextPart(content=remaining.pop(0))])

    return FunctionModel(function=respond), prompts


@pytest.mark.asyncio
async def test_a_failed_calculation_rerank_is_traced_and_warns_the_ai_admin(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", True)
    caplog.set_level("INFO", logger="unisage.graph")
    result = await _fee_turn({"rerank": _note_model("không phải JSON")})

    assert "09a_LLMRerankNode" in _traced_nodes(caplog)
    assert any("parent=07_CalculationNode" in record.getMessage() for record in caplog.records)
    assert [warning.code for warning in result.admin_warnings] == ["LLM_RERANK_FAILED"]
    assert "850.000" in result.response_text


@pytest.mark.asyncio
async def test_a_missing_rerank_model_warns_the_ai_admin_on_a_calculation_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", True)
    result = await _fee_turn({"rerank_unavailable": "chưa có credential RERANK"})

    [warning] = result.admin_warnings
    assert warning.code == "LLM_RERANK_UNAVAILABLE"
    assert warning.message.endswith("chưa có credential RERANK")


_FEE_TABLE = RetrievedChunk(
    chunk_id="hp:3.1",
    content=(
        "| 3.1 | Khóa tuyển sinh năm học 2025-2026 | | |\n"
        "| | - Khối Công nghệ: Môn lý thuyết: 980.000 / Môn thực hành: 1.600.000 | | |"
    ),
    source="Quyet Dinh 1035 - Hoc phi 2025-2026",
    score=0.9,
)
_PRICES_FOUND = {
    "values": {
        "don_gia_lt": {"value": 980000, "chunk": 1},
        "don_gia_th": {"value": 1600000, "chunk": 1},
    },
    "ask": [],
}


def _tuition_models(*scripted: object, chunks: list[RetrievedChunk]) -> GraphModels:
    return GraphModels(
        classification=_scripted(*scripted),
        query_transformation=_echo_model(),
        generation=_note_model("Mình đã tính theo biểu phí."),
        retrieval=_RecordingRetrieval(chunks),
    )


@pytest.mark.asyncio
async def test_tuition_prices_come_from_the_fee_schedule_with_a_citation() -> None:
    query = "Khoá 2025-2026, kỳ này em học 15 tín lý thuyết 4 tín thực hành, học phí bao nhiêu?"
    result = await run_graph(
        _input(query),
        _tuition_models(
            {"tasks": [{"intent": "academic_calculation", "query": query}]},
            {
                "formula_id": "tuition_semester",
                "params": {"tc_lt": 15, "tc_th": 4, "khoa_nhap_hoc": "2025-2026"},
                "retrieval_query": None,
            },
            _PRICES_FOUND,
            chunks=[_FEE_TABLE],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert result.pending_round is None
    assert "Học phí một học kỳ **21.100.000 đồng**" in result.response_text
    assert "- Đơn giá 1 tín chỉ lý thuyết (đồng): 980.000 đồng [1]" in result.response_text
    assert "AI tự tính" not in result.response_text
    assert [(c["index"], c["title"]) for c in result.citations] == [
        (1, "Quyet Dinh 1035 - Hoc phi 2025-2026")
    ]


@pytest.mark.asyncio
async def test_tuition_asks_the_intake_first_then_reads_the_prices() -> None:
    query = "tính học phí học kỳ này giúp em"
    task = {"tasks": [{"intent": "academic_calculation", "query": query}]}
    extraction = {"formula_id": "tuition_semester", "params": {}, "retrieval_query": None}
    ask_intake = {
        "values": {},
        "ask": [{"field": "khoa_nhap_hoc", "options": ["2025-2026", "2024-2025"]}],
    }
    first = await run_graph(
        _input(query),
        _tuition_models(task, extraction, ask_intake, chunks=[_FEE_TABLE]),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert first.pending_round is not None
    questions = {q.field: q for q in first.pending_round.panel.questions}
    assert list(questions) == ["khoa_nhap_hoc", "tc_lt", "tc_th"]
    intake = questions["khoa_nhap_hoc"]
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[
            Answer(question_id=intake.id, option_id=intake.options[0].id),
            Answer(question_id=questions["tc_lt"].id, number=Decimal(15)),
            Answer(question_id=questions["tc_th"].id, number=Decimal(4)),
        ],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    second = await run_graph(
        _input("đã trả lời", resume=resume),
        _tuition_models(_PRICES_FOUND, chunks=[_FEE_TABLE]),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert second.pending_round is None
    assert "**21.100.000 đồng**" in second.response_text


@pytest.mark.asyncio
async def test_without_documents_tuition_asks_theory_and_practice_credits_separately() -> None:
    query = "tính giúp tôi học phí cả ngành công nghệ thông tin đi"
    result = await run_graph(
        _input(query),
        _tuition_models(
            {"tasks": [{"intent": "academic_calculation", "query": query}]},
            {"formula_id": "tuition_program", "params": {}, "retrieval_query": None},
            chunks=[],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert result.pending_round is not None
    assert [q.field for q in result.pending_round.panel.questions] == [
        "tc_lt",
        "tc_th",
        "don_gia_lt",
        "don_gia_th",
    ]


@pytest.mark.asyncio
async def test_tuition_for_two_years_asks_and_titles_the_period_the_student_named() -> None:
    query = "tính giúp tôi học phí ngành công nghệ thông tin 2 năm đầu"
    first = await run_graph(
        _input(query),
        _tuition_models(
            {"tasks": [{"intent": "academic_calculation", "query": query}]},
            {"formula_id": "tuition_year", "params": {"pham_vi": "2 năm đầu"}},
            _PRICES_FOUND,
            chunks=[_FEE_TABLE],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert first.pending_round is not None
    questions = {q.field: q for q in first.pending_round.panel.questions}
    assert questions["tc_lt"].prompt == "Số tín chỉ lý thuyết của 2 năm đầu"
    assert questions["tc_th"].prompt == "Số tín chỉ thực hành của 2 năm đầu"
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[
            Answer(question_id=questions["tc_lt"].id, number=Decimal(100)),
            Answer(question_id=questions["tc_th"].id, number=Decimal(42)),
        ],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    second = await run_graph(
        _input("đã trả lời", resume=resume),
        _tuition_models(_PRICES_FOUND, chunks=[_FEE_TABLE]),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert second.pending_round is None
    assert "Học phí 2 năm đầu **165.200.000 đồng**" in second.response_text
    assert "một năm học" not in second.response_text


@pytest.mark.asyncio
async def test_every_llm_call_of_a_calculation_turn_gets_the_current_time() -> None:
    classification: list[str] = []
    generation: list[str] = []
    await run_graph(
        _input(_TWO_YEARS),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": _TWO_YEARS}]}),
                json.dumps({"formula_id": "tuition_year", "params": {"tc_lt": 100, "tc_th": 42}}),
                json.dumps(_PRICES_FOUND),
            ),
            query_transformation=_echo_model(),
            generation=_recording(generation, _review(True, "ok"), "x"),
            retrieval=_RecordingRetrieval([_FEE_TABLE]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    # classifier, extractor, value extractor; then the review
    assert len(classification) == 3 and len(generation) == 1
    assert all("<current_time>" in prompt for prompt in classification + generation)
    assert "Học kỳ hiện tại:" in generation[0]


@pytest.mark.asyncio
async def test_every_llm_call_of_an_advisory_turn_gets_the_current_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    hyde: list[str] = []
    generation: list[str] = []

    def echo(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompt = str(getattr(messages[-1].parts[-1], "content", ""))
        hyde.append(prompt)
        return ModelResponse(parts=[TextPart(content=prompt.splitlines()[0])])

    await run_graph(
        _input("Học phí kỳ này bao nhiêu?"),
        GraphModels(
            classification=make_classification_llm_model(
                "academic_advisory", routing_mode="SINGLE"
            ),
            query_transformation=FunctionModel(function=echo),
            generation=_prompt_capturing_generation(generation),
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert hyde and generation
    assert all("<current_time>" in prompt for prompt in hyde + generation)


@pytest.mark.asyncio
async def test_a_target_question_after_a_calculation_is_solved_by_python() -> None:
    """ "cuối kỳ cần bao nhiêu để được A" reuses the previous calculation's numbers and
    is solved on the formula's grid - no document search, no LLM arithmetic."""

    first = await run_graph(
        _input(_COURSE_QUERY),
        GraphModels(
            classification=_scripted(_COURSE_TASK, _COURSE_PARAMS),
            query_transformation=_echo_model(),
            generation=_note_model("Bạn cố gắng nhé."),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert first.last_calculation is not None
    assert first.last_calculation.title == "Điểm tổng kết học phần (lý thuyết + thực hành)"

    follow_up = "thế cuối kỳ cần bao nhiêu để được A"
    retrieval = _RecordingRetrieval([_CHUNK])
    second = await run_graph(
        _input(
            follow_up,
            last_calculation=first.last_calculation,
            history=[HistoryMessage(role="USER", content=_COURSE_QUERY)],
        ),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": follow_up}]},
                {
                    "formula_id": "previous",
                    "params": {},
                    "solve_for": "ck",
                    "target": {"output": "dtkhp", "op": ">=", "value": "A"},
                },
            ),
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "Bạn cần cố gắng ở bài cuối kỳ nhé.")),
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    # CK 9.5 → ĐLT 8.45 → 8.5 → ĐTKHP (8.5 * 2 + 8.5) / 3 = 8.5 (đạt); CK 9.0 → 8.3.
    assert "Điểm cuối kỳ, thang 10 tối thiểu **9.5**" in second.response_text
    assert "Kiểm tra: Điểm cuối kỳ, thang 10 = 9.5 → ĐTKHP 8.5 (đạt)" in second.response_text
    assert "AI tự tính" not in second.response_text
    assert retrieval.queries == []  # registry rules only, no document search
    [item] = second.calculation_items
    assert (item["mode"], item["status"]) == ("builtin", "computed")
    trace = second.calculation_traces[0]["trace"]
    assert (trace["formula_id"], trace["solve_for"]) == ("course_score", "ck")
    assert trace["target"] == {"output": "dtkhp", "op": ">=", "value": "8.5"}
    assert trace["formula_version"] == "1"
    assert second.last_calculation is not None
    assert second.last_calculation.plan.solve_for == "ck"
    assert "ck" not in second.last_calculation.params


@pytest.mark.asyncio
async def test_a_question_no_formula_fits_is_answered_on_the_advisory_path(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arithmetic on the documents' figures has no registered formula: the extractor
    says `none`, the task joins the advisory part and node 10 answers from the
    documents (its prompt tells it to show the figures, not to compute)."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    query = "Ngành Logistics, 4 môn đó cộng lại mấy tín?"
    credits = RetrievedChunk(
        chunk_id="ctdt:3", content="Thủ tục hải quan 3 tín chỉ...", source="CTĐT.pdf", score=0.8
    )
    retrieval = _RecordingRetrieval([credits])
    tokens: list[str] = []
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "none", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=mock_streaming_llm_model(
                ["Thủ tục hải quan 3 tín chỉ [1], ", "bạn cộng với 3 môn còn lại nhé."]
            ),
            retrieval=retrieval,
        ),
        _sink(tokens),
        _TRACE,
        _usage_recorder(),
    )

    assert retrieval.queries and all(query in searched for searched in retrieval.queries)
    assert result.response_text == "Thủ tục hải quan 3 tín chỉ [1], bạn cộng với 3 môn còn lại nhé."
    assert "".join(tokens) == result.response_text
    assert result.calculation_items == [] and result.pending_round is None
    assert len(result.citations) == 1


def test_citations_of_document_values_are_numbered_after_the_advisory_sources() -> None:
    other = RetrievedChunk(chunk_id="c2", content="x", source="t", score=0.5)
    plan = CalculationPlan(formula_id="tuition_semester")
    result = calculate(
        "tuition_semester", {"tc_lt": 10, "tc_th": 0, "don_gia_lt": 980000, "don_gia_th": 0}
    )
    outcomes: list[TaskOutcome] = [
        Computed(
            "T1",
            result,
            plan,
            {},
            notes=("Đơn giá LT: 980.000 đồng [2]", "Đơn giá TH: 0 đồng [1]"),
            sources=[(2, other), (1, _CHUNK)],
        ),
        Computed(
            "T2", result, plan, {}, notes=("Đơn giá LT: 980.000 đồng [1]",), sources=[(1, _CHUNK)]
        ),
    ]
    numbered, citations = number_citations(outcomes, first_index=4)
    assert [o.notes for o in numbered if isinstance(o, Computed)] == [
        ("Đơn giá LT: 980.000 đồng [4]", "Đơn giá TH: 0 đồng [5]"),
        ("Đơn giá LT: 980.000 đồng [6]",),
    ]
    assert [(c["index"], c["title"]) for c in citations] == [(4, "t"), (5, "s"), (6, "s")]


_FEE_LOOKUP = "mức thu học phí 01 tín chỉ môn lý thuyết, môn thực hành"


@pytest.mark.asyncio
async def test_document_lookups_are_filtered_like_the_advisory_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master's fee note shares keywords with an undergraduate fee question: the LLM
    rerank drops it and the score threshold drops weak matches before the value
    extractor reads the prices."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.70)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", True)
    query = "Tính học phí kỳ này cho ngành Công nghệ thông tin, 15 tín lý thuyết 4 tín thực hành"
    master = RetrievedChunk(
        chunk_id="ths:52",
        content="Học phí thạc sĩ đóng theo học kỳ, căn cứ số học phần đăng ký",
        source="Thạc sĩ Tài chính ngân hàng.pdf",
        score=0.85,
    )
    undergrad = RetrievedChunk(
        chunk_id="hp:1",
        content="Đại học chính quy khóa 2025, Khối Công nghệ: 850.000 đồng/tín chỉ",
        source="Biểu học phí.pdf",
        score=0.8,
        heading_path=["Đại học chính quy"],
    )
    weak = RetrievedChunk(chunk_id="x:1", content="Lịch thi", source="Lịch.pdf", score=0.5)
    rerank, rerank_prompts = _recording_model({"SQ1": {"C2": "học phí ĐHCQ Khối Công nghệ"}})
    classification: list[str] = []
    retrieval = _RecordingRetrieval([weak, master, undergrad])
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": query}]}),
                json.dumps({"formula_id": "tuition_semester", "params": {"tc_lt": 15, "tc_th": 4}}),
                json.dumps(
                    {
                        "values": {
                            "don_gia_lt": {"value": 850000, "chunk": 1},
                            "don_gia_th": {"value": 850000, "chunk": 1},
                        },
                        "ask": [],
                    }
                ),
            ),
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "Học phí kỳ này của bạn như dưới đây.")),
            retrieval=retrieval,
            rerank=rerank,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert retrieval.queries == [f"{_FEE_LOOKUP}  {query}"]  # no profile context between
    assert query in rerank_prompts[0] and "Lịch thi" not in rerank_prompts[0]
    _classifier, _extractor, value_prompt = classification
    assert "850.000 đồng/tín chỉ" in value_prompt and "mục: Đại học chính quy" in value_prompt
    assert "Học phí thạc sĩ" not in value_prompt
    assert "Học phí một học kỳ **16.150.000 đồng**" in result.response_text
    assert "850.000 đồng [1]" in result.response_text


async def _fee_turn(models: dict[str, object]) -> GraphOutput:
    query = "Tính học phí kỳ này cho ngành Công nghệ thông tin, 15 tín lý thuyết 4 tín thực hành"
    fee = RetrievedChunk(chunk_id="hp:1", content="850.000 đồng/tín chỉ", source="HP", score=0.9)
    return await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "tuition_semester", "params": {"tc_lt": 15, "tc_th": 4}},
                {
                    "values": {
                        "don_gia_lt": {"value": 850000, "chunk": 1},
                        "don_gia_th": {"value": 850000, "chunk": 1},
                    },
                    "ask": [],
                },
            ),
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "Học phí kỳ này như dưới đây.")),
            retrieval=_RecordingRetrieval([fee]),
            **models,  # type: ignore[arg-type]
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )


@pytest.mark.asyncio
async def test_the_extractor_sees_what_the_student_typed_next_to_the_rewrite() -> None:
    typed = "tôi khogn biết nhưng môn này tx 8 gk 7 ck 6.5 th 9 8, 2 tín lt 1 tín th"
    rewrite = _COURSE_QUERY
    classification: list[str] = []
    await run_graph(
        _input(typed),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": rewrite}]}),
                json.dumps(_COURSE_PARAMS),
            ),
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "ok")),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    _classifier, extractor = classification
    assert extractor.startswith(rewrite)
    assert f"<original_message>{typed}</original_message>" in extractor


@pytest.mark.asyncio
async def test_a_query_equal_to_the_message_is_not_repeated() -> None:
    classification: list[str] = []
    await run_graph(
        _input(_COURSE_QUERY),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": _COURSE_QUERY}]}),
                json.dumps(_COURSE_PARAMS),
            ),
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "ok")),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert "<original_message>" not in classification[1]


@pytest.mark.asyncio
async def test_a_deferred_turn_flags_a_sum_the_model_states_itself(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict grounding on a deferred calculation question: "16 tín chỉ" is 3 + 4 + 2 + 7,
    derivable in a normal turn, but here the model was told to request the sum - the
    caution line and the AI-admin warning follow."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    query = "Ngành Logistics, 4 môn đó cộng lại mấy tín?"
    credits = RetrievedChunk(
        chunk_id="ctdt:3",
        content="Quản lý thông tin 3 TC; Giao nhận 4 TC; Phân loại 2 TC; Hải quan 7 TC",
        source="CTĐT.pdf",
        score=0.8,
    )
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "none", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=mock_streaming_llm_model(["Tổng là 16 tín chỉ (3 + 4 + 2 + 7) [1]."]),
            retrieval=_RecordingRetrieval([credits]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert "> Lưu ý: con số 16 không tìm thấy trong tài liệu được trích dẫn" in result.response_text
    assert "ANSWER_UNGROUNDED_NUMBERS" in [warning.code for warning in result.admin_warnings]


@pytest.mark.asyncio
async def test_a_calc_request_in_the_answer_is_computed_by_code_and_appended(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    query = "Ngành Logistics, 4 môn đó cộng lại mấy tín?"
    credits = RetrievedChunk(
        chunk_id="ctdt:3",
        content="Quản lý thông tin 3 TC; Giao nhận 4 TC; Phân loại 2 TC; Hải quan 3 TC",
        source="CTĐT.pdf",
        score=0.8,
    )
    block = (
        '```json\n{"type": "calc_request", "label": "Tổng tín chỉ 4 học phần", '
        '"expression": "3 + 4 + 2 + 3", "unit": "tín chỉ", "sources": [1]}\n```'
    )
    tokens: list[str] = []
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "none", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=mock_streaming_llm_model(
                ["4 học phần lần lượt 3, 4, 2 và 3 tín chỉ [1]; tổng tính ở dưới.\n\n", block]
            ),
            retrieval=_RecordingRetrieval([credits]),
        ),
        _sink(tokens),
        _TRACE,
        _usage_recorder(),
    )

    assert "calc_request" not in result.response_text
    assert "**Phép tính số học: Tổng tín chỉ 4 học phần**" in result.response_text
    assert "3 + 4 + 2 + 3 = 12" in result.response_text
    assert "Kết quả: Tổng tín chỉ 4 học phần: **12 tín chỉ**" in result.response_text
    # The streamed tokens keep the blank line before the (redacted) block; the stored
    # text is right-stripped before the Python block is appended.
    assert "".join(tokens).replace("\n\n\n\n", "\n\n") == result.response_text
    assert "ANSWER_UNGROUNDED_NUMBERS" not in [warning.code for warning in result.admin_warnings]
    assert result.calculation_items == []


@pytest.mark.asyncio
async def test_a_deferred_turn_tells_node_10_to_hand_the_arithmetic_to_a_calc_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.prompting.builder import ARITHMETIC_NOTICE

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    query = "Ngành Logistics, 4 môn đó cộng lại mấy tín?"
    seen: list[str] = []
    await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "none", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=_prompt_capturing_generation(seen),
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert seen and ARITHMETIC_NOTICE in seen[0]

    # An ordinary advisory question carries no such reminder.
    plain_seen: list[str] = []
    await run_graph(
        _input(_ADVISORY_QUERY),
        GraphModels(
            classification=make_classification_llm_model(
                "academic_advisory", routing_mode="SINGLE"
            ),
            query_transformation=_echo_model(),
            generation=_prompt_capturing_generation(plain_seen),
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert plain_seen and ARITHMETIC_NOTICE not in plain_seen[0]
