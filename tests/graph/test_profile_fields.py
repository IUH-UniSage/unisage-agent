"""Profile values and registry names in questions and answers: a field the student
already declared, or a value a calculation already has under another name, is never
asked again; a built-in formula reads its profile-scoped parameters from the profile."""

import json
from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.calculation.formulas import calculate, param_spec
from app.core.config import settings
from app.graph.calculation_turn import calculation_titles, known_fields
from app.graph.clarification_answers import validate_answers
from app.graph.clarification_round import TaskQuestions, advisory_questions, build_round
from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    profile_params,
    question_for,
    run_calculation_task,
)
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphModels, ResumeInput
from app.schemas.clarification import (
    Answer,
    CalculationPlan,
    ClarificationSubmit,
    LastCalculation,
    PendingCalculationTask,
)
from tests.graph.test_calculation_document_values import _FEES, _Retrieval
from tests.graph.test_calculation_node import (
    _COURSE_QUERY,
    _TRACE,
    _input,
    _sink,
    _usage_recorder,
)

_PROFILE = {"khoa_nhap_hoc": "2022", "khoi_nganh": "Khối Công nghệ", "nganh_hoc": "CNTT"}
_PRICES = {
    "values": {
        "don_gia_lt": {"value": 980000, "chunk": 1},
        "don_gia_th": {"value": 1600000, "chunk": 1},
    }
}


def _scripted(*answers: object) -> tuple[GraphModels, list[str]]:
    """Every model call answered in order with one JSON document, prompts recorded."""

    prompts: list[str] = []
    remaining = [json.dumps(answer, ensure_ascii=False) for answer in answers]

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        return ModelResponse(parts=[TextPart(content=remaining.pop(0))])

    model = FunctionModel(function=respond)
    models = GraphModels(
        classification=model,
        query_transformation=model,
        generation=model,
        retrieval=_Retrieval([_FEES]),
    )
    return models, prompts


def _never() -> FunctionModel:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise AssertionError("no model call expected")

    return FunctionModel(function=fail)


@pytest.fixture(autouse=True)
def _score_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)


def test_advisory_questions_skip_what_the_turns_calculations_already_have() -> None:
    form = {
        "type": "ask_user_form",
        "fields": [
            {
                "field": "khoa",
                "options": [{"id": "k18", "label": "K18"}, {"id": "k19", "label": "K19"}],
            },
            {
                "field": "he",
                "options": [{"id": "cq", "label": "CQ"}, {"id": "clc", "label": "CLC"}],
            },
        ],
    }
    questions = advisory_questions([form], confirmed_metadata={}, known_fields={"khoa_nhap_hoc"})
    assert [q["field"] for q in questions] == ["he_dao_tao"]


def test_node_10_is_told_the_names_of_the_values_a_calculation_used_never_the_numbers() -> None:
    outcome = Computed(
        "T1",
        calculate("grade_conversion", {"score10": 8}),
        CalculationPlan(formula_id="grade_conversion"),
        {"score10": 8, "so_tc_lt": 15},
    )
    assert known_fields([outcome]) == {"score10", "tc_lt"}
    [title] = calculation_titles([outcome])
    assert title.endswith("[đã có: score10, tc_lt]")
    assert not any(number in title for number in ("8", "15"))


def test_profile_params_are_only_the_formulas_profile_scoped_ones() -> None:
    assert profile_params("tuition_semester", _PROFILE) == {
        "khoa_nhap_hoc": "2022",
        "khoi_nganh": "Khối Công nghệ",
    }
    assert profile_params("grade_conversion", _PROFILE) == {}


@pytest.mark.asyncio
async def test_a_builtin_formula_reads_the_intake_from_the_profile_instead_of_asking() -> None:
    extraction = {"formula_id": "tuition_semester", "params": {"tc_lt": 15, "tc_th": 4}}
    models, prompts = _scripted(extraction, _PRICES)
    deps = CalculationDeps(models=models, confirmed_metadata=_PROFILE)

    outcome = await run_calculation_task("T1", "học phí kỳ này 15 tín LT 4 tín TH", deps)

    assert isinstance(outcome, Computed)
    assert outcome.params["khoa_nhap_hoc"] == "2022"
    assert "khoa_nhap_hoc: 2022" in prompts[1]


@pytest.mark.asyncio
async def test_a_value_typed_this_turn_wins_over_the_profile() -> None:
    extraction = {
        "formula_id": "tuition_semester",
        "params": {"tc_lt": 15, "tc_th": 4, "khoa_nhap_hoc": "2021"},
    }
    models, prompts = _scripted(extraction, _PRICES)
    deps = CalculationDeps(models=models, confirmed_metadata=_PROFILE)

    outcome = await run_calculation_task("T1", "khoá 2021 học phí kỳ này bao nhiêu", deps)

    assert isinstance(outcome, Computed) and outcome.params["khoa_nhap_hoc"] == "2021"
    assert "khoa_nhap_hoc: 2021" in prompts[1]


@pytest.mark.asyncio
async def test_the_profile_replaces_the_previous_calculations_old_intake() -> None:
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params={"tc_lt": 15, "tc_th": 4, "khoa_nhap_hoc": "2020"},
    )
    models, _prompts = _scripted({"formula_id": "previous", "params": {"tc_th": 6}}, _PRICES)
    deps = CalculationDeps(models=models, confirmed_metadata=_PROFILE, previous=previous)

    outcome = await run_calculation_task("T1", "nếu học phí 6 tín thực hành thì sao", deps)

    assert isinstance(outcome, Computed)
    assert (outcome.params["khoa_nhap_hoc"], outcome.params["tc_th"]) == ("2022", 6)


@pytest.mark.asyncio
async def test_a_panel_answer_under_an_alias_goes_into_the_profile_under_its_registry_name() -> (
    None
):
    task = PendingCalculationTask(
        task_id="T1",
        query=_COURSE_QUERY,
        plan=CalculationPlan(formula_id="course_score"),
        known_params={"tclt": 2, "tcth": 1, "tbtx": 8, "gk": 7, "ck": 6.5},
    )
    intake_question = {
        "tab_label": "Khoá",
        "prompt": "Khoá tuyển sinh",
        "kind": "text",
        "max_length": 200,
        "origin": "calculation",
        "field": "khoa_tuyen_sinh",
    }
    th_question = question_for(param_spec("course_score", "th"), {})  # type: ignore[arg-type]
    pending = build_round(
        [TaskQuestions(task=task, questions=[th_question, intake_question])],
        original_query=_COURSE_QUERY,
        chain_depth=1,
    )
    assert pending is not None
    submit = ClarificationSubmit(
        action="submit",
        panel_id=pending.panel.panel_id,
        answers=[
            Answer(question_id="q1", numbers=[Decimal("9"), Decimal("8")]),
            Answer(question_id="q2", text="2022"),
        ],
    )
    resume = ResumeInput(pending_round=pending, answers=validate_answers(pending.panel, submit))
    models = GraphModels(
        classification=_never(),  # a panel answer calls neither the classifier nor the extractor
        query_transformation=_never(),
        generation=_never(),
        retrieval=_Retrieval([]),
    )

    result = await run_graph(
        _input("x", resume=resume), models, _sink([]), _TRACE, _usage_recorder()
    )

    # The intake is a profile field: kept as the intake year; the score stays with its task.
    assert result.confirmed_metadata == {"khoa_nhap_hoc": "2022"}
    assert result.last_calculation is not None and "th" in result.last_calculation.params
