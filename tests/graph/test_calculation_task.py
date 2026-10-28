"""run_calculation_task / registered formulas (SPEC-calculation-node §1)."""

import json
from collections.abc import Callable
from decimal import Decimal

import pytest
from pydantic import JsonValue
from pydantic_ai.models.function import FunctionModel

from app.calculation.formulas import route_builtin
from app.calculation.solver import Target, Unknown
from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    Deferred,
    NeedsInput,
    Unresolved,
    builtin_outcome,
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
def test_unreadable_or_unknown_extractor_output_is_deferred(raw: str) -> None:
    extraction = parse_extraction(raw, allowed=[])
    assert (extraction.formula_id, extraction.retrieval_query) == ("none", None)


def test_the_router_fills_in_only_for_unreadable_or_unknown_output() -> None:
    assert parse_extraction("không phải json", allowed=["gpa"]).formula_id == "gpa"
    assert parse_extraction('{"formula_id": "tuition"}', allowed=["gpa"]).formula_id == "gpa"
    # The model's registered choice stands even when the keyword router saw another one.
    raw = '{"formula_id": "grade_conversion", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa"]).formula_id == "grade_conversion"


def test_an_explicit_none_is_never_overridden_by_the_router() -> None:
    # "điểm xét tuyển ngành CNTT năm 2026 bao nhiêu" (a cutoff score) matches the admission
    # trigger; the year is a digit, so this is a lookup and the model's `none` stands.
    raw = '{"formula_id": "none", "params": {}, "retrieval_query": "điểm chuẩn 2026"}'
    question = "điểm xét tuyển ngành CNTT năm 2026 bao nhiêu"
    extraction = parse_extraction(raw, allowed=["admission_score_2026"], question=question)
    assert (extraction.formula_id, extraction.retrieval_query) == ("none", "điểm chuẩn 2026")


def test_several_routed_formulas_leave_the_choice_to_the_model() -> None:
    raw = '{"formula_id": "course_score", "params": {}}'
    assert parse_extraction(raw, allowed=["gpa", "grade_conversion"]).formula_id == "course_score"
    assert parse_extraction("hỏng", allowed=["gpa", "grade_conversion"]).formula_id == "none"


def test_a_deferred_extraction_keeps_its_retrieval_query() -> None:
    raw = json.dumps(
        {"formula_id": "none", "params": {"so_tc": 20}, "retrieval_query": "công thức học phí"}
    )
    extraction = parse_extraction(raw, allowed=[])
    assert (extraction.formula_id, extraction.retrieval_query) == ("none", "công thức học phí")


def test_an_inverse_extraction_names_the_unknown_and_the_target() -> None:
    raw = json.dumps(
        {
            "formula_id": "course_score",
            "params": {"tclt": 3, "tcth": 0, "tbtx": 8, "gk": 7},
            "solve_for": "ck",
            "target": {"output": "dtkhp", "op": ">=", "value": "A+"},
        }
    )
    extraction = parse_extraction(raw, allowed=["course_score"])
    assert extraction.unknown == Unknown("ck")
    assert extraction.target == Target("dtkhp", ">=", Decimal("9.0"))
    assert "ck" not in extraction.params


def test_an_unreadable_target_falls_back_to_the_forward_calculation() -> None:
    raw = json.dumps(
        {
            "formula_id": "course_score",
            "params": {"tclt": 3},
            "solve_for": "ck",
            "target": {"output": "nope", "op": ">=", "value": 9},
        }
    )
    extraction = parse_extraction(raw, allowed=[])
    assert (extraction.formula_id, extraction.unknown, extraction.target) == (
        "course_score",
        None,
        None,
    )


def test_previous_needs_a_previous_builtin_calculation() -> None:
    raw = '{"formula_id": "previous", "params": {"gk": 8}}'
    assert parse_extraction(raw, allowed=[]).formula_id == "none"
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


def test_a_tuition_question_goes_to_its_builtin_formula() -> None:
    question = "kỳ này em học 15 tín lý thuyết 4 tín thực hành, học phí bao nhiêu?"
    raw = json.dumps(
        {"formula_id": "tuition_semester", "params": {"tc_lt": 15, "tc_th": 4}},
        ensure_ascii=False,
    )
    extraction = parse_extraction(raw, allowed=route_builtin(question))

    assert extraction.formula_id == "tuition_semester"
    assert extraction.params == {"tc_lt": 15, "tc_th": 4}


def test_a_reverse_tuition_question_is_solved_for_the_credits() -> None:
    question = "Đăng ký tối đa mấy tín để học phí kỳ này không quá 8 triệu?"
    raw = json.dumps(
        {
            "formula_id": "tuition_semester",
            "params": {"tc_th": 0},
            "solve_for": "tc_lt",
            "target": {"output": "hoc_phi", "op": "<=", "value": 8000000},
        }
    )

    assert route_builtin(question) == ["tuition_semester"]
    extraction = parse_extraction(raw, allowed=route_builtin(question))
    assert extraction.formula_id == "tuition_semester"
    assert extraction.unknown == Unknown("tc_lt")
    assert extraction.target == Target("hoc_phi", "<=", Decimal(8000000))


@pytest.mark.asyncio
async def test_a_deferred_task_is_returned_as_such(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    raw = json.dumps({"formula_id": "none", "params": {}, "retrieval_query": "thống kê phúc khảo"})
    outcome = await run_calculation_task(
        "T1", "tổng cộng bao nhiêu đơn phúc khảo?", _deps(mock_sync_llm_model(raw))
    )
    assert outcome == Deferred("T1", "tổng cộng bao nhiêu đơn phúc khảo?", "thống kê phúc khảo")


@pytest.mark.asyncio
async def test_an_inverse_task_is_solved_by_python(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    raw = json.dumps(
        {
            "formula_id": "course_score",
            "params": {"tclt": 3, "tcth": 0, "tbtx": 8, "gk": 7},
            "solve_for": "ck",
            "target": {"output": "dtkhp", "op": ">=", "value": "B+"},
        }
    )
    outcome = await run_calculation_task(
        "T1", "cuối kỳ cần bao nhiêu để được B+", _deps(mock_sync_llm_model(raw))
    )
    assert isinstance(outcome, Computed)
    assert outcome.plan.solve_for == "ck"
    assert outcome.plan.target is not None and outcome.plan.target.value == "8"
    assert outcome.result.values["nghiem"] == Decimal("8.5")  # 1.6 + 2.1 + 0.5 * 8.5 = 7.95 → 8.0
    assert "ck" not in outcome.params


def test_an_inverse_task_never_asks_for_its_unknown() -> None:
    outcome = builtin_outcome(
        "T1",
        "course_score",
        {"tclt": 3, "tcth": 0, "tbtx": 8},
        unknown=Unknown("ck"),
        target=Target("dtkhp", ">=", Decimal("8")),
    )
    assert isinstance(outcome, NeedsInput)
    assert [question["field"] for question in outcome.questions] == ["gk"]
    assert outcome.plan.solve_for == "ck" and outcome.plan.target is not None


@pytest.mark.asyncio
async def test_resume_finishes_an_inverse_plan(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    plan = CalculationPlan(
        formula_id="course_score",
        solve_for="ck",
        target={"output": "dtkhp", "op": ">=", "value": "8"},
    )
    outcome = await resume_calculation_task(
        "T1",
        "q",
        plan,
        {"tclt": 3, "tcth": 0, "tbtx": 8},
        {"gk": 7},
        _deps(mock_sync_llm_model("x")),
    )
    assert isinstance(outcome, Computed)
    assert outcome.result.values["nghiem"] == Decimal("8.5")


@pytest.mark.asyncio
async def test_resume_of_a_round_stored_by_the_retired_llm_path_says_so(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    plan = CalculationPlan(formula_id="llm", retrieval_query="học phí")
    outcome = await resume_calculation_task(
        "T1", "q", plan, {"so_tc": 20}, {"don_gia": 980000}, _deps(mock_sync_llm_model("x"))
    )
    assert outcome == Unresolved("T1", "legacy_llm")


def test_every_registry_formula_is_a_valid_plan() -> None:
    """A formula added to the registry must also be storable in a pending round's plan."""

    from app.calculation.formulas import FORMULAS

    for formula_id in FORMULAS:
        assert CalculationPlan(formula_id=formula_id).formula_id == formula_id  # type: ignore[arg-type]


def test_a_bare_request_for_a_routed_formula_is_not_deferred() -> None:
    """ "tính điểm xét tuyển giúp em": the model answered `none` (no numbers), but the router
    matched exactly one formula and the question holds no digit - the formula's panel asks
    for the inputs instead of the advisory path answering with prose."""

    raw = '{"formula_id": "none", "params": {}, "retrieval_query": "công thức điểm xét tuyển"}'
    question = "tính điểm xét tuyển giúp em"
    assert route_builtin(question) == ["admission_score_2026"]
    extraction = parse_extraction(raw, allowed=route_builtin(question), question=question)
    assert extraction.formula_id == "admission_score_2026" and extraction.params == {}
    # With numbers in the question the model's `none` stands (figures may come from documents).
    with_numbers = "điểm chuẩn ngành CNTT năm 2025 là 24, năm 2026 bao nhiêu?"
    assert (
        parse_extraction(raw, allowed=route_builtin(with_numbers), question=with_numbers).formula_id
        == "none"
    )
    # Several routed formulas: the model's `none` stands too.
    assert (
        parse_extraction(raw, allowed=["gpa", "course_score"], question=question).formula_id
        == "none"
    )
