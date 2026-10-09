"""run_calculation_task / builtin path (SPEC-calculation-node §1)."""

import json
from collections.abc import Callable

import pytest
from pydantic import JsonValue
from pydantic_ai.models.function import FunctionModel

from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    NeedsInput,
    Unresolved,
    builtin_outcome,
    parse_extraction,
    resume_builtin,
    run_calculation_task,
)
from app.graph.streaming_state import GraphModels
from app.schemas.clarification import CalculationPlan
from tests.llm_mocks import FakeRetrievalService


def _deps(extractor: FunctionModel) -> CalculationDeps:
    return CalculationDeps(
        models=GraphModels(
            classification=extractor,
            query_transformation="unused",
            generation="unused",
            retrieval=FakeRetrievalService(),
        )
    )


FULL: dict[str, JsonValue] = {"tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8], "tclt": 2, "tcth": 1}


@pytest.mark.asyncio
async def test_builtin_with_every_param_is_computed(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    raw = json.dumps({"formula_id": "course_score", "params": FULL})
    outcome = await run_calculation_task(
        "T1", "điểm tổng kết học phần TX 8 GK 7...", _deps(mock_sync_llm_model(raw))
    )
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs) == {"ĐTKHP": "7.5", "Điểm chữ": "B", "Thang 4": "3.0"}


@pytest.mark.asyncio
async def test_missing_practice_scores_become_a_number_list_question(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    params = {key: value for key, value in FULL.items() if key != "th"}
    raw = json.dumps({"formula_id": "course_score", "params": params})
    outcome = await run_calculation_task(
        "T1", "điểm tổng kết học phần", _deps(mock_sync_llm_model(raw))
    )
    assert isinstance(outcome, NeedsInput)
    assert [(q["field"], q["kind"]) for q in outcome.questions] == [("th", "number_list")]
    assert outcome.questions[0]["number"] == {"min": "0", "max": "10", "step": "0.01", "unit": None}
    assert outcome.known_params == params


def test_rejected_value_is_asked_again_with_its_reason() -> None:
    outcome = builtin_outcome("T1", "course_score", {**FULL, "ck": 11})
    assert isinstance(outcome, NeedsInput)
    assert outcome.questions[0]["field"] == "ck"
    assert "bạn nhập 11" in outcome.questions[0]["prompt"]
    assert "ck" not in outcome.known_params


def test_unknown_params_are_ignored() -> None:
    outcome = builtin_outcome("T1", "grade_conversion", {"score10": 8, "hacker": "x"})
    assert isinstance(outcome, Computed)
    assert outcome.params == {"score10": 8}


def test_gpa_without_courses_asks_for_a_course_table() -> None:
    outcome = builtin_outcome("T2", "gpa", {})
    assert isinstance(outcome, NeedsInput)
    assert outcome.questions == [
        {
            "tab_label": "Các môn",
            "prompt": "Các môn: số tín chỉ và điểm (thang 10 hoặc điểm chữ)",
            "kind": "course_table",
            "origin": "calculation",
            "field": "courses",
            "max_items": 30,
        }
    ]


@pytest.mark.parametrize(
    "raw", ["không phải json", '{"formula_id": "tuition", "params": {}}', "[]"]
)
@pytest.mark.asyncio
async def test_broken_extractor_output_fails_closed_without_retrieval(
    raw: str, mock_sync_llm_model: Callable[[str], FunctionModel]
) -> None:
    retrieval = FakeRetrievalService()
    deps = _deps(mock_sync_llm_model(raw))
    deps.models.retrieval = retrieval
    outcome = await run_calculation_task("T1", "tính học phí", deps)
    assert outcome == Unresolved("T1", "extraction_failed")


def test_router_decision_overrides_the_model() -> None:
    extraction = parse_extraction(
        '{"formula_id": "retrieved", "params": {}}', allowed=["gpa"], query="GPA?"
    )
    assert extraction is not None and extraction.formula_id == "gpa"


def test_model_must_pick_within_several_routed_formulas() -> None:
    raw = '{"formula_id": "retrieved", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa", "grade_conversion"], query="q") is None
    raw = '{"formula_id": "gpa", "params": {}}'
    extraction = parse_extraction(raw, allowed=["gpa", "grade_conversion"], query="q")
    assert extraction is not None and extraction.formula_id == "gpa"


def test_unrouted_question_may_go_to_the_regulations() -> None:
    raw = json.dumps(
        {"formula_id": "retrieved", "params": {"so_tc": 20}, "retrieval_query": "công thức học phí"}
    )
    extraction = parse_extraction(raw, allowed=[], query="học phí?")
    assert extraction is not None
    assert (extraction.formula_id, extraction.retrieval_query) == ("retrieved", "công thức học phí")


def test_resume_builtin_merges_answers_and_computes() -> None:
    known = {key: value for key, value in FULL.items() if key != "th"}
    outcome = resume_builtin(
        "T1", CalculationPlan(formula_id="course_score"), known, {"th": ["9", "8"]}
    )
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs)["ĐTKHP"] == "7.5"


def test_resume_gpa_from_course_table_rows() -> None:
    rows: list[JsonValue] = [
        {"name": "Toán", "credits": 3, "score": "8.5"},
        {"name": None, "credits": 2, "score": "B+"},
    ]
    outcome = resume_builtin("T1", CalculationPlan(formula_id="gpa"), {}, {"courses": rows})
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs)["GPA"] == "3.68"
