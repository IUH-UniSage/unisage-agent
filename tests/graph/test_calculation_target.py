"""Target questions ("cần bao nhiêu để được A+") through CalculationNode, for
built-in and regulation formulas, and follow-ups on the previous calculation."""

import json
from typing import Any

import pytest
from pydantic import JsonValue

from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    NeedsInput,
    Unresolved,
    builtin_outcome,
    parse_extraction,
    parse_solve,
    resume_calculation,
    run_calculation_task,
)
from app.graph.streaming_state import GraphModels
from app.schemas.clarification import (
    CalculationPlan,
    LastCalculation,
    RetrievedFormulaPlan,
    SolveGoal,
    SolveSpec,
)
from tests.graph.test_calculation_retrieved import (
    CHUNK,
    EXTRACTION,
    FORMULA,
    VERIFIED,
    _found,
    _scripted,
)
from tests.llm_mocks import FakeRetrievalService

COURSE: dict[str, JsonValue] = {"tclt": 3, "tcth": 1, "tbtx": [9, 8, 7], "gk": 9, "th": [6, 9, 9]}
A_PLUS = SolveSpec(
    unknowns=["ck"], goal=SolveGoal(comparator=">=", value=9, grade="A+"), want="min"
)


def _deps(model: Any, previous: LastCalculation | None = None) -> CalculationDeps:
    return CalculationDeps(
        models=GraphModels(
            classification=model,
            query_transformation="unused",
            generation="unused",
            retrieval=FakeRetrievalService([CHUNK]),
        ),
        previous=previous,
    )


def test_letter_goal_becomes_the_band_lower_bound() -> None:
    request = parse_solve({"unknowns": ["ck"], "goal": {"grade": "a+"}})
    assert request is not None
    assert (request.goal.comparator, request.goal.value, request.goal.grade) == (">=", 9, "A+")
    assert request.want == "min"
    upper = parse_solve({"unknowns": [], "goal": {"comparator": "<=", "value": 8000000}})
    assert upper is not None and upper.want == "max"
    assert parse_solve({"unknowns": ["ck"], "goal": {"grade": "Z"}}) is None
    assert parse_solve({"unknowns": ["ck"]}) is None


def test_final_exam_needed_for_a_plus() -> None:
    outcome = builtin_outcome("T1", "course_score", COURSE, A_PLUS)
    assert isinstance(outcome, Computed)
    assert outcome.result.primary_value == 10
    assert outcome.plan.solve == A_PLUS
    assert "ck" not in outcome.params


def test_unknown_is_never_asked_but_other_missing_params_are() -> None:
    params = {key: value for key, value in COURSE.items() if key != "th"}
    outcome = builtin_outcome("T1", "course_score", {**params, "ck": 5}, A_PLUS)
    assert isinstance(outcome, NeedsInput)
    assert [question["field"] for question in outcome.questions] == ["th"]
    assert "ck" not in outcome.known_params

    resumed = resume_calculation("T1", outcome.plan, outcome.known_params, {"th": [6, 9, 9]})
    assert isinstance(resumed, Computed) and resumed.result.primary_value == 10


def test_invalid_given_value_is_asked_before_solving() -> None:
    outcome = builtin_outcome("T1", "course_score", {**COURSE, "gk": 12}, A_PLUS)
    assert isinstance(outcome, NeedsInput)
    assert [question["field"] for question in outcome.questions] == ["gk"]


@pytest.mark.asyncio
async def test_unsolvable_unknown_is_an_honest_refusal() -> None:
    model, _ = _scripted(
        {
            "formula_id": "gpa",
            "params": {},
            "solve": {"unknowns": ["courses"], "goal": {"grade": "A"}},
        }
    )
    outcome = await run_calculation_task("T1", "GPA cần bao nhiêu", _deps(model))
    assert outcome == Unresolved("T1", "target_unsupported")


@pytest.mark.asyncio
async def test_follow_up_reuses_the_previous_calculation_without_asking_again() -> None:
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="course_score"), params={**COURSE, "ck": 8}
    )
    model, prompts = _scripted(
        {
            "formula_id": "previous",
            "params": {},
            "solve": {"unknowns": ["ck"], "goal": {"grade": "A+"}},
        }
    )
    outcome = await run_calculation_task(
        "T1", "thế cuối kỳ cần bao nhiêu để được A+", _deps(model, previous)
    )
    assert isinstance(outcome, Computed)
    assert outcome.result.primary_value == 10
    assert "<previous_calculation>" in prompts[0]
    assert "<allowed_formula_ids>course_score, grade_conversion, previous" in prompts[0]


def test_previous_is_rejected_when_there_is_none() -> None:
    raw = json.dumps({"formula_id": "previous", "params": {}})
    assert parse_extraction(raw, allowed=[], query="q", previous=None) is None


@pytest.mark.asyncio
async def test_regulation_formula_is_solved_for_its_declared_unknown() -> None:
    extraction = {
        **EXTRACTION,
        "params": {},
        "solve": {"unknowns": [], "goal": {"comparator": "<=", "value": 8000000}},
    }
    model, _ = _scripted(extraction, _found(unknowns=["so_tc"]), VERIFIED)
    outcome = await run_calculation_task(
        "T1", "học phí không quá 8 triệu thì đăng ký tối đa bao nhiêu tín", _deps(model)
    )
    assert isinstance(outcome, Computed)
    assert outcome.result.primary_value == 19
    assert outcome.source is not None
    assert outcome.result.summary is not None and "tối đa **19**" in outcome.result.summary


@pytest.mark.asyncio
async def test_regulation_formula_rejects_an_undeclared_unknown() -> None:
    extraction = {
        **EXTRACTION,
        "solve": {"unknowns": [], "goal": {"comparator": "<=", "value": 8000000}},
    }
    model, _ = _scripted(extraction, _found(unknowns=["phi_khac"]), VERIFIED)
    outcome = await run_calculation_task("T1", "học phí", _deps(model))
    assert outcome == Unresolved("T1", "target_unsupported")


@pytest.mark.asyncio
async def test_follow_up_on_a_regulation_formula_skips_retrieval() -> None:
    retrieved = RetrievedFormulaPlan.model_validate(
        {
            "expression": FORMULA["expression"],
            "variables": FORMULA["variables"],
            "result_label": FORMULA["result_label"],
            "source": {
                "chunk_id": "c_8",
                "source": "QD-hoc-phi.pdf",
                "chunk_hash": "sha256:x",
                "source_quote": FORMULA["source_quote"],
            },
        }
    )
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="retrieved", retrieved=retrieved),
        params={"so_tc": 20, "don_gia": 420000},
    )
    model, prompts = _scripted(
        {
            "formula_id": "previous",
            "params": {},
            "solve": {"unknowns": ["so_tc"], "goal": {"comparator": "<=", "value": 6000000}},
        }
    )
    outcome = await run_calculation_task(
        "T1", "thế 6 triệu thì tối đa mấy tín", _deps(model, previous)
    )
    assert isinstance(outcome, Computed)
    assert outcome.result.primary_value == 14
    assert len(prompts) == 1  # extractor only: no retrieval, formula or verifier call
