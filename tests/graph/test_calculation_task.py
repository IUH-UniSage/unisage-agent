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
    builtin_outcome,
    llm_questions,
    parse_extraction,
    resume_calculation_task,
    run_calculation_task,
)
from app.graph.streaming_state import GraphModels
from app.schemas.clarification import CalculationPlan, LastCalculation
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
def test_unreadable_or_unknown_extractor_output_goes_to_the_llm(raw: str) -> None:
    extraction = parse_extraction(raw, allowed=[])
    assert (extraction.formula_id, extraction.retrieval_query) == ("llm", None)


def test_router_decision_overrides_a_wrong_or_unreadable_builtin() -> None:
    raw = '{"formula_id": "grade_conversion", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa"]).formula_id == "gpa"
    assert parse_extraction("không phải json", allowed=["gpa"]).formula_id == "gpa"


def test_llm_choice_is_never_overridden_by_the_router() -> None:
    # "cuối kỳ cần bao nhiêu để được A+" matches the grade_conversion trigger.
    raw = '{"formula_id": "llm", "params": {"gk": 9}}'
    extraction = parse_extraction(raw, allowed=["grade_conversion"])
    assert (extraction.formula_id, extraction.params) == ("llm", {"gk": 9})


def test_model_must_pick_within_several_routed_formulas() -> None:
    raw = '{"formula_id": "course_score", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa", "grade_conversion"]).formula_id == "llm"
    raw = '{"formula_id": "gpa", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa", "grade_conversion"]).formula_id == "gpa"


def test_llm_extraction_keeps_its_retrieval_query() -> None:
    raw = json.dumps(
        {"formula_id": "llm", "params": {"so_tc": 20}, "retrieval_query": "công thức học phí"}
    )
    extraction = parse_extraction(raw, allowed=[])
    assert (extraction.formula_id, extraction.retrieval_query) == ("llm", "công thức học phí")


def test_previous_needs_a_previous_builtin_calculation() -> None:
    raw = '{"formula_id": "previous", "params": {"gk": 8}}'
    assert parse_extraction(raw, allowed=[]).formula_id == "llm"
    previous = LastCalculation(plan=CalculationPlan(formula_id="course_score"), params=FULL)
    assert parse_extraction(raw, allowed=[], previous=previous).formula_id == "previous"


@pytest.mark.asyncio
async def test_forward_follow_up_reuses_the_previous_numbers(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    previous = LastCalculation(plan=CalculationPlan(formula_id="course_score"), params=FULL)
    deps = _deps(mock_sync_llm_model('{"formula_id": "previous", "params": {"ck": 10}}'))
    deps = CalculationDeps(models=deps.models, previous=previous)
    outcome = await run_calculation_task("T1", "nếu cuối kỳ 10 thì sao", deps)
    assert isinstance(outcome, Computed)
    assert outcome.params["ck"] == 10


@pytest.mark.asyncio
async def test_resume_builtin_merges_answers_and_computes(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    known = {key: value for key, value in FULL.items() if key != "th"}
    outcome = await resume_calculation_task(
        "T1",
        "q",
        CalculationPlan(formula_id="course_score"),
        known,
        {"th": ["9", "8"]},
        _deps(mock_sync_llm_model("unused")),
    )
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs)["ĐTKHP"] == "7.5"


@pytest.mark.asyncio
async def test_resume_gpa_from_course_table_rows(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    rows: list[JsonValue] = [
        {"name": "Toán", "credits": 3, "score": "8.5"},
        {"name": None, "credits": 2, "score": "B+"},
    ]
    outcome = await resume_calculation_task(
        "T1",
        "q",
        CalculationPlan(formula_id="gpa"),
        {},
        {"courses": rows},
        _deps(mock_sync_llm_model("unused")),
    )
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs)["GPA"] == "3.68"


def test_llm_questions_drop_answered_duplicate_and_malformed_fields() -> None:
    form = {
        "type": "ask_user_form",
        "fields": [
            {"field": "Đơn giá", "label": "Đơn giá 1 tín chỉ", "kind": "number", "min": 0},
            {"field": "don_gia", "label": "lặp lại", "kind": "number"},
            {"field": "so_tc", "label": "Số tín chỉ", "kind": "number"},
            {"field": "he", "label": "Hệ", "kind": "choice", "options": [{"label": "Chính quy"}]},
            {
                "field": "phuong_thuc",
                "label": "Phương thức xét tuyển",
                "kind": "choice",
                "options": [{"label": "Thi THPT"}, {"label": "ĐGNL"}],
            },
            {"field": "x", "label": "lạ", "kind": "formula"},
        ],
    }
    questions = llm_questions([form], known={"so_tc": 20})
    # "Đơn giá" and "don_gia" are the same field: asked once; so_tc is already known.
    assert [(q["field"], q["kind"]) for q in questions] == [
        ("don_gia", "number"),
        ("phuong_thuc", "choice"),
    ]
    number = questions[0]["number"]
    assert (number["min"], number["max"]) == ("0", "1000000000000")
    assert [o["id"] for o in questions[1]["options"]] == ["o1", "o2"]
