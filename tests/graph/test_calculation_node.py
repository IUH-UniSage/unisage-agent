"""CalculationNode placeholder, alone and merged with the advisory branch."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.calculation.formulas import TIMES, param_spec
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.calculation_turn import (
    LLM_NOTICE,
    NEEDS_INPUT_LEAD,
    REVIEW_ISSUE_KEY,
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
    UNRESOLVED_MESSAGES,
    LlmAnswered,
    TaskOutcome,
    llm_questions,
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
    LastCalculation,
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
async def test_a_result_that_does_not_fit_is_redone_by_the_llm() -> None:
    prompts: list[str] = []
    redone = "Học phí 2 năm đầu là **196.000.000 đồng** [1]."
    result = await _two_years_turn(
        _recording(prompts, _review(False, _TWO_YEARS_ANSWER, _TWO_YEARS_ISSUE), redone)
    )

    assert result.response_text == f"**{LLM_NOTICE}**\n\n{redone}"
    assert result.pending_round is None
    _review_prompt, llm_prompt = prompts
    assert '"Số tín chỉ lý thuyết của một năm học": 100' in llm_prompt
    assert f'"{REVIEW_ISSUE_KEY}": "{_TWO_YEARS_ISSUE}"' in llm_prompt
    assert f"Câu hỏi của người dùng: {_TWO_YEARS}" in llm_prompt
    [item] = result.calculation_items
    assert (item["mode"], item["status"]) == ("llm", "computed")
    trace = result.calculation_traces[0]["trace"]
    assert trace["redone_from"] == "tuition_year"
    assert trace["review"] == {"fits": False, "issue": _TWO_YEARS_ISSUE, "rejected": False}
    assert "chat_calculation_review" in trace["prompt_versions"]
    assert result.last_calculation is not None
    assert result.last_calculation.plan.formula_id == "llm"
    # Its inputs are kept for a follow-up, like a built-in one's parameters.
    params = result.last_calculation.params
    assert params["Số tín chỉ lý thuyết của một năm học"] == 100
    assert REVIEW_ISSUE_KEY not in params


@pytest.mark.asyncio
async def test_a_redo_that_needs_numbers_asks_on_the_panel_as_an_llm_task() -> None:
    ask = (
        "Mình cần số tín chỉ của cả 2 năm đầu.\n"
        '{"type": "ask_user_form", "fields": [{"field": "tc_lt_2_nam", '
        '"label": "Số tín chỉ lý thuyết của 2 năm đầu", "kind": "number", "min": 0, "max": 300}]}'
    )
    result = await _two_years_turn(
        _recording([], _review(False, _TWO_YEARS_ANSWER, _TWO_YEARS_ISSUE), ask)
    )

    assert result.pending_round is not None
    [question] = result.pending_round.panel.questions
    assert question.prompt == "Số tín chỉ lý thuyết của 2 năm đầu"
    [task] = result.pending_round.tasks
    assert isinstance(task, PendingCalculationTask)
    assert task.plan.formula_id == "llm"
    assert task.known_params[REVIEW_ISSUE_KEY] == _TWO_YEARS_ISSUE
    assert "165.200.000" not in result.response_text


@pytest.mark.asyncio
async def test_with_the_llm_path_off_a_mismatch_shows_the_warning_above_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_CALC_LLM_ENABLED", False)
    result = await _two_years_turn(_note_model(_review(False, _TWO_YEARS_ANSWER, _TWO_YEARS_ISSUE)))

    assert result.response_text.startswith(
        f"**Lưu ý:** {_TWO_YEARS_ISSUE}\n\n{_TWO_YEARS_ANSWER}\n\n"
    )
    assert "Học phí một năm học **165.200.000 đồng**" in result.response_text
    trace = result.calculation_traces[0]["trace"]
    assert trace["review"] == {"fits": False, "issue": _TWO_YEARS_ISSUE, "rejected": False}
    assert "redone_from" not in trace


@pytest.mark.asyncio
async def test_when_the_redo_fails_the_block_stays_under_the_warning() -> None:
    result = await _two_years_turn(
        _recording([], _review(False, _TWO_YEARS_ANSWER, _TWO_YEARS_ISSUE))
    )

    assert result.response_text.startswith(f"**Lưu ý:** {_TWO_YEARS_ISSUE}")
    assert "Học phí một năm học **165.200.000 đồng**" in result.response_text
    assert LLM_NOTICE not in result.response_text
    assert result.calculation_items[0]["mode"] == "builtin"


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


_LLM_ANSWER = (
    f"**Công thức**: ĐLT = 20% {TIMES} TBtx + 30% {TIMES} GK + 50% {TIMES} CK\n\n"
    "Thay số: CK 9.5 → ĐTKHP 8.5 (đạt); CK 9.0 → 8.3 (chưa đạt)\n\n"
    "**Kết quả: cần CK tối thiểu 9.5**"
)


@pytest.mark.asyncio
async def test_target_question_after_a_calculation_is_answered_by_the_llm() -> None:
    """Python only computes forward; "cuối kỳ cần bao nhiêu để được A" goes to the LLM
    with the built-in rules, the numbers and the chat, under the AI notice."""

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
    generation, prompts = _recording_model(_LLM_ANSWER)
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
                {"formula_id": "llm", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert second.response_text.startswith(f"**{LLM_NOTICE}**")
    assert "cần CK tối thiểu 9.5" in second.response_text
    assert retrieval.queries == []  # built-in rules only, no document search
    assert f"ĐLT = 20% {TIMES} TBtx + 30% {TIMES} GK + 50% {TIMES} CK" in prompts[0]
    assert _COURSE_QUERY in prompts[0]  # the chat, where the numbers are
    assert second.calculation_items[0]["mode"] == "llm"
    assert second.calculation_items[0]["status"] == "computed"
    assert second.calculation_traces[0]["trace"]["answer"] == _LLM_ANSWER
    assert second.last_calculation is not None and second.last_calculation.title == follow_up


@pytest.mark.asyncio
async def test_llm_asks_missing_numbers_on_the_panel_and_resume_skips_the_classifier() -> None:
    query = "học phí 20 tín bao nhiêu"
    ask = (
        "Mình cần đơn giá một tín chỉ.\n\n```json\n"
        + json.dumps(
            {
                "type": "ask_user_form",
                "fields": [
                    {"field": "don_gia", "label": "Đơn giá 1 tín chỉ (đồng)", "kind": "number"}
                ],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    generation, _ = _recording_model(ask)
    retrieval = _RecordingRetrieval([_CHUNK])
    first = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {
                    "formula_id": "llm",
                    "params": {"so_tin_chi": 20},
                    "retrieval_query": "công thức học phí",
                },
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert first.response_text == "Mình cần đơn giá một tín chỉ."
    # The LLM already saw the retrieval_query documents: no second search before the form.
    assert retrieval.queries == ["công thức học phí"]
    assert first.pending_round is not None
    [question] = first.pending_round.panel.questions
    assert (question.kind, question.field) == ("number", "don_gia")

    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[Answer(question_id=question.id, number=Decimal(420000))],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    generation, prompts = _recording_model(
        f"Đơn giá theo quy định [1][7]. Học phí = 20 {TIMES} 420.000 = **8.400.000đ**"
    )
    second = await run_graph(
        _input("Đơn giá 420000", resume=resume),
        GraphModels(
            classification=_scripted(),  # any classifier/extractor call would fail
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert '"so_tin_chi": 20' in prompts[0] and '"don_gia": "420000"' in prompts[0]
    assert "8.400.000đ" in second.response_text
    assert "[1]" in prompts[0] and "<academic_context>" in prompts[0]
    # Sources are the [n] markers the client links, not a "Nguồn:" list; a marker of a
    # document the LLM never saw ([7]) is dropped.
    assert "Nguồn:" not in second.response_text
    assert "theo quy định [1]." in second.response_text
    assert [(c["index"], c["title"]) for c in second.citations] == [(1, "s")]
    assert second.calculation_traces[0]["trace"]["sources"][0]["ref"] == "[1]"
    assert second.pending_round is None


@pytest.mark.asyncio
async def test_a_choice_answer_reaches_the_llm_as_its_label_so_it_is_not_asked_again() -> None:
    query = "tính điểm xét tuyển giúp tôi"
    ask = (
        "Bạn xét tuyển theo phương thức nào?\n\n```json\n"
        + json.dumps(
            {
                "type": "ask_user_form",
                "fields": [
                    {
                        "field": "phuong_thuc",
                        "label": "Phương thức xét tuyển",
                        "kind": "choice",
                        "options": [
                            {"id": "xt1", "label": "XT1 (Học bạ)"},
                            {"id": "xt3", "label": "XT3 (Kết quả thi ĐGNL)"},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    generation, _ = _recording_model(ask)
    first = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert first.pending_round is not None
    [question] = first.pending_round.panel.questions
    xt3 = next(option for option in question.options if option.label.startswith("XT3"))
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[Answer(question_id=question.id, option_id=xt3.id)],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    generation, prompts = _recording_model("Mình tính theo XT3.")
    await run_graph(
        _input("Phương thức xét tuyển: XT3", resume=resume),
        GraphModels(
            classification=_scripted(),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert '"phuong_thuc": "XT3 (Kết quả thi ĐGNL)"' in prompts[0]


def test_llm_citations_are_numbered_after_the_advisory_sources() -> None:
    other = RetrievedChunk(chunk_id="c2", content="Đơn giá", source="t", score=0.8)
    plan = CalculationPlan(formula_id="llm", retrieval_query="học phí")
    outcomes: list[TaskOutcome] = [
        LlmAnswered("T1", "q1", "A [2] rồi B [1, 2]", plan, {}, [(2, other), (1, _CHUNK)]),
        LlmAnswered("T2", "q2", "C [1]", plan, {}, [(1, _CHUNK)]),
    ]
    numbered, citations = number_citations(outcomes, first_index=4)
    assert [o.text for o in numbered if isinstance(o, LlmAnswered)] == [
        "A [4] rồi B [5][4]",
        "C [6]",
    ]
    assert [(c["index"], c["title"]) for c in citations] == [(4, "t"), (5, "s"), (6, "s")]


@pytest.mark.asyncio
async def test_qdrant_is_searched_with_the_question_before_asking_the_user() -> None:
    query = "Ngành Logistics, 4 môn đó cộng lại mấy tín?"
    ask = (
        "Mình cần số tín chỉ của từng môn.\n\n```json\n"
        + json.dumps(
            {
                "type": "ask_user_form",
                "fields": [{"field": "tin_chi", "label": "Số tín chỉ", "kind": "number"}],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    credits = RetrievedChunk(
        chunk_id="ctdt:3", content="Thủ tục hải quan 3 tín chỉ...", source="CTĐT.pdf", score=0.8
    )
    generation, prompts = _recording_model(ask, "Tổng là 3 + 4 + 2 + 3 = **12 tín chỉ** [1].")
    retrieval = _RecordingRetrieval([credits])
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert retrieval.queries == [query]  # no retrieval_query: searched with the question
    assert len(prompts) == 2 and "Thủ tục hải quan 3 tín chỉ" in prompts[1]
    assert "<academic_context>" in prompts[1]
    assert result.pending_round is None  # answered from the document, no form
    assert "12 tín chỉ" in result.response_text


@pytest.mark.asyncio
async def test_calculation_documents_are_filtered_like_the_advisory_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master's fee note shares keywords with an undergraduate fee question: the LLM
    rerank drops it, the score threshold drops weak matches, and the prompt carries the
    programme/level rules."""

    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.70)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", True)
    query = "Tính học phí cho ngành Công nghệ thông tin"
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
    generation, prompts = _recording_model("Đơn giá 850.000 đồng/tín chỉ [1].")
    retrieval = _RecordingRetrieval([weak, master, undergrad])
    await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {
                    "formula_id": "llm",
                    "params": {},
                    "retrieval_query": "định mức học phí ngành Công nghệ thông tin",
                },
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
            rerank=rerank,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert retrieval.queries == ["định mức học phí ngành Công nghệ thông tin"]
    assert query in rerank_prompts[0] and "Lịch thi" not in rerank_prompts[0]
    [prompt] = prompts
    assert "850.000 đồng/tín chỉ" in prompt and "mục: Đại học chính quy" in prompt
    assert "Học phí thạc sĩ" not in prompt
    assert "PHÂN BIỆT BẬC/HỆ ĐÀO TẠO" in prompt


async def _fee_turn(models: dict[str, object]) -> GraphOutput:
    query = "Tính học phí cho ngành Công nghệ thông tin"
    fee = RetrievedChunk(chunk_id="hp:1", content="850.000 đồng/tín chỉ", source="HP", score=0.9)
    return await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": "học phí CNTT"},
            ),
            query_transformation=_echo_model(),
            generation=_note_model("Đơn giá 850.000 đồng/tín chỉ [1]."),
            retrieval=_RecordingRetrieval([fee]),
            **models,  # type: ignore[arg-type]
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )


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


@pytest.mark.asyncio
async def test_an_unfenced_ask_form_becomes_a_panel_not_text() -> None:
    query = "tính giúp tôi học phí cả ngành công nghệ thông tin đi"
    ask = "Mình chưa tìm thấy biểu phí. Vui lòng chọn:\n\n" + json.dumps(
        {
            "type": "ask_user_form",
            "fields": [
                {
                    "field": "bac_he",
                    "label": "Bạn thuộc bậc/hệ đào tạo nào?",
                    "kind": "choice",
                    "options": [
                        {"id": "dhcq", "label": "Đại học chính quy"},
                        {"id": "ths", "label": "Cao học (Thạc sĩ)"},
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": None},
            ),
            query_transformation=_echo_model(),
            generation=_note_model(ask),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert "ask_user_form" not in result.response_text
    assert result.pending_round is not None
    [question] = result.pending_round.panel.questions
    assert question.prompt == "Bạn thuộc bậc/hệ đào tạo nào?"


_COURSE_FORM = (
    "Bạn vui lòng cung cấp các điểm thành phần nhé:\n\n```json\n"
    + json.dumps(
        {
            "type": "ask_user_form",
            "fields": [
                {"field": "tclt", "label": "Số tín chỉ lý thuyết", "kind": "number", "min": 0},
                {"field": "tcth", "label": "Số tín chỉ thực hành", "kind": "number", "min": 0},
                {"field": "tbtx", "label": "Điểm thường xuyên", "kind": "number", "max": 10},
                {"field": "gk", "label": "Điểm giữa kỳ", "kind": "number", "max": 10},
                {"field": "ck", "label": "Điểm cuối kỳ", "kind": "number", "max": 10},
                {"field": "th", "label": "Điểm thực hành", "kind": "number", "max": 10},
            ],
        },
        ensure_ascii=False,
    )
    + "\n```"
)


@pytest.mark.asyncio
async def test_an_llm_form_answered_with_a_builtin_formulas_numbers_is_computed_by_python() -> None:
    query = "tính điểm"
    generation, _ = _recording_model(_COURSE_FORM)
    retrieval = _RecordingRetrieval([_CHUNK])
    first = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": "công thức tính điểm"},
            ),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    # The student's own numbers are missing: the LLM's form is asked, nothing is handed over.
    assert first.pending_round is not None
    questions = first.pending_round.panel.questions
    assert [q.field for q in questions] == ["tclt", "tcth", "tbtx", "gk", "ck", "th"]
    numbers = {"tclt": 3, "tcth": 1, "tbtx": 9, "gk": 8, "ck": 9, "th": 9}
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[Answer(question_id=q.id, number=Decimal(numbers[q.field])) for q in questions],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    # The answers are the course score's numbers: Python computes, the LLM is not called.
    generation, prompts = _recording_model()
    second = await run_graph(
        _input("đã trả lời", resume=resume),
        GraphModels(
            classification=_scripted(),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert prompts == []  # no calculation LLM call and no review call
    assert second.pending_round is None
    assert LLM_NOTICE not in second.response_text
    assert "Điểm tổng kết học phần" in second.response_text
    assert "thang 4 **3.8**" in second.response_text
    # A follow-up ("làm tròn thang 4") can reuse it as the previous built-in calculation.
    assert second.last_calculation is not None
    assert second.last_calculation.plan.formula_id == "course_score"
    [item] = second.calculation_items
    assert (item["mode"], item["status"]) == ("builtin", "computed")


@pytest.mark.asyncio
async def test_a_form_asking_only_known_values_is_unresolved_not_an_ai_result() -> None:
    query = "4 môn đó cộng lại mấy tín?"
    ask = (
        "Mình cần số môn.\n\n```json\n"
        + json.dumps(
            {
                "type": "ask_user_form",
                "fields": [{"field": "so_mon", "label": "Số môn", "kind": "number"}],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    result = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {"so_mon": 4}, "retrieval_query": "tín chỉ"},
            ),
            query_transformation=_echo_model(),
            generation=_note_model(ask),
            retrieval=_RecordingRetrieval([_CHUNK]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert result.pending_round is None
    assert LLM_NOTICE not in result.response_text
    assert result.response_text.strip() == UNRESOLVED_MESSAGES["form_dropped"]
    [item] = result.calculation_items
    assert item["status"] == "unresolved"  # no Đúng/Sai buttons


async def _asked_for_a_price(retrieval: _RecordingRetrieval) -> GraphOutput:
    query = "học phí 20 tín bao nhiêu"
    ask = (
        "Mình cần đơn giá một tín chỉ.\n\n```json\n"
        + json.dumps(
            {
                "type": "ask_user_form",
                "fields": [{"field": "don_gia", "label": "Đơn giá (đồng)", "kind": "number"}],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    return await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}, "retrieval_query": "công thức học phí"},
            ),
            query_transformation=_echo_model(),
            generation=_note_model(ask),
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )


def _price_answer(first: GraphOutput) -> ResumeInput:
    assert first.pending_round is not None
    [question] = first.pending_round.panel.questions
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[Answer(question_id=question.id, number=Decimal(420000))],
    )
    return ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("indexed", "searched"),
    [
        ([_CHUNK], []),  # still indexed: fetched by id, no search and no rerank
        ([], ["công thức học phí"]),  # gone since: searched with the retrieval_query again
    ],
)
async def test_resume_reuses_the_documents_the_llm_saw_when_it_asked(
    indexed: list[RetrievedChunk], searched: list[str]
) -> None:
    first = await _asked_for_a_price(_RecordingRetrieval([_CHUNK]))
    assert first.pending_round is not None
    [task] = first.pending_round.tasks
    assert isinstance(task, PendingCalculationTask)
    assert task.plan.chunk_ids == [_CHUNK.chunk_id]

    retrieval = _RecordingRetrieval(indexed)
    generation, prompts = _recording_model(f"Học phí = 20 {TIMES} 420.000 = **8.400.000đ** [1]")
    await run_graph(
        _input("Đơn giá 420000", resume=_price_answer(first)),
        GraphModels(
            classification=_scripted(),
            query_transformation=_echo_model(),
            generation=generation,
            retrieval=retrieval,
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert retrieval.queries == searched
    assert (_CHUNK.content in prompts[0]) == bool(indexed)


@pytest.mark.asyncio
async def test_a_follow_up_reuses_the_numbers_of_a_calculation_the_llm_did() -> None:
    # The student answered six numbers on an LLM form; the LLM computed the course score.
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="llm"),
        params={"tclt": "3", "tcth": "1", "tbtx": "9", "gk": "8", "ck": "8.5", "th": "9"},
        title="tính điểm",
    )
    query = "Tính điểm làm tròn thang 4 dựa trên điểm tổng kết học phần đã cung cấp"
    extractor, prompts = _recording_model(
        {"tasks": [{"intent": "academic_calculation", "query": query}]},
        {"formula_id": "grade_conversion", "params": {}, "retrieval_query": None},
    )
    result = await run_graph(
        _input("làm tròn thang 4", recent_calculations=[previous]),
        GraphModels(
            classification=extractor,
            query_transformation=_echo_model(),
            generation=_note_model(_review(True, "Thang 4 của bạn là **3.8**.")),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert '"ck": "8.5"' in prompts[1]  # the extractor sees the previous numbers
    assert result.pending_round is None  # nothing asked again
    assert result.calculation_items[0]["result_summary"] == "ĐTKHP: 8.6; Điểm chữ: A; Thang 4: 3.8"


def test_llm_questions_accepts_the_models_small_format_slips() -> None:
    form = {
        "type": "ask_user_form",
        "fields": [
            {
                "field": "phuong_thuc",
                "label": "Phương thức",
                "kind": "radio",
                "options": ["XT1", "XT2", "XT3"],
            },
            {
                "field": "loai_mon",
                "question": "Khối ngành",
                "kind": "single_choice",
                "options": [{"id": "a", "text": "Công nghệ"}, {"id": "b", "value": "Kinh tế"}],
            },
            {"field": "diem", "label": "Điểm", "kind": "integer"},
            {"field": "x", "label": "Lạ", "kind": "slider"},
        ],
    }
    questions = llm_questions([form], {})
    assert [(q["field"], q["kind"]) for q in questions] == [
        ("phuong_thuc", "choice"),
        ("loai_mon", "choice"),
        ("diem", "number"),
    ]
    assert [o["label"] for o in questions[0]["options"]] == ["XT1", "XT2", "XT3"]
    assert [o["label"] for o in questions[1]["options"]] == ["Công nghệ", "Kinh tế"]


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
    assert LLM_NOTICE not in result.response_text
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
async def test_the_extractor_and_the_llm_see_what_the_student_typed_next_to_the_rewrite() -> None:
    typed = "tôi khogn biết kì nay tổng bao nhiêu nhưng tôi đã nộp 6 triệu rồi"
    rewrite = "Tính học phí còn lại phải nộp của học kỳ này, biết tổng học phí và đã nộp 6 triệu"
    classification: list[str] = []
    generation: list[str] = []
    await run_graph(
        _input(typed),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": rewrite}]}),
                json.dumps({"formula_id": "llm", "params": {"da_nop": 6000000}}),
            ),
            query_transformation=_echo_model(),
            generation=_recording(generation, "Học phí còn lại cần tính từ số tín chỉ."),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    _classifier, extractor = classification
    assert extractor.startswith(rewrite)
    assert f"<original_message>{typed}</original_message>" in extractor
    [llm] = generation
    assert f"Câu hỏi của người dùng: {rewrite}\nTin nhắn gốc người dùng gõ" in llm
    assert llm.rstrip().endswith(typed)
    assert "KHÔNG hỏi lại chính giá trị đó" in llm


@pytest.mark.asyncio
async def test_a_query_equal_to_the_message_is_not_repeated() -> None:
    generation: list[str] = []
    classification: list[str] = []
    query = "Cuối kỳ cần bao nhiêu để được A+?"
    await run_graph(
        _input(query),
        GraphModels(
            classification=_recording(
                classification,
                json.dumps({"tasks": [{"intent": "academic_calculation", "query": query}]}),
                json.dumps({"formula_id": "llm", "params": {}}),
            ),
            query_transformation=_echo_model(),
            generation=_recording(generation, "ok"),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert "<original_message>" not in classification[1]
    assert "Tin nhắn gốc người dùng gõ" not in generation[0]


@pytest.mark.asyncio
async def test_an_intake_answered_on_the_form_is_kept_as_a_year_and_names_the_student_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = datetime(2026, 10, 13, 9, 0)
    monkeypatch.setattr("app.rag.prompting.builder.now_ict", lambda: today)
    monkeypatch.setattr("app.graph.streaming_graph.now_ict", lambda: today)
    query = "học phí kỳ này của tôi bao nhiêu"
    ask = json.dumps(
        {
            "type": "ask_user_form",
            "fields": [
                {
                    "field": "khoa_nhap_hoc",
                    "label": "Bạn nhập học năm nào (hoặc đang học kỳ thứ mấy)?",
                    "kind": "choice",
                    "options": [
                        {"id": "2022", "label": "Nhập học 2022 (đang học kỳ 9)"},
                        {"id": "2023", "label": "Nhập học 2023 (đang học kỳ 7)"},
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    first_prompts: list[str] = []
    first = await run_graph(
        _input(query),
        GraphModels(
            classification=_scripted(
                {"tasks": [{"intent": "academic_calculation", "query": query}]},
                {"formula_id": "llm", "params": {}},
            ),
            query_transformation=_echo_model(),
            generation=_recording(first_prompts, ask),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert "Học kỳ của sinh viên trong khoá học: CHƯA BIẾT" in first_prompts[0]
    assert first.pending_round is not None
    [question] = first.pending_round.panel.questions
    submit = ClarificationSubmit(
        action="submit",
        panel_id=first.pending_round.panel.panel_id,
        answers=[Answer(question_id=question.id, option_id=question.options[0].id)],
    )
    resume = ResumeInput(
        pending_round=first.pending_round,
        answers=validate_answers(first.pending_round.panel, submit),
    )
    second_prompts: list[str] = []
    second = await run_graph(
        _input("Nhập học 2022 (đang học kỳ 9)", resume=resume),
        GraphModels(
            classification=_scripted(),
            query_transformation=_echo_model(),
            generation=_recording(second_prompts, "Học kỳ 9 của bạn ..."),
            retrieval=_RecordingRetrieval([]),
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert second.confirmed_metadata["khoa_nhap_hoc"] == "2022"
    assert (
        "Học kỳ của sinh viên trong khoá học: học kỳ 9 của khoá học (nhập học 2022, khóa 18)"
        in second_prompts[0]
    )
