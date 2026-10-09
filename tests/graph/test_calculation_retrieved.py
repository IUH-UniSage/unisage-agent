"""Regulation formulas (Qdrant): 7 fail-closed checks (SPEC-calculation-node §2)."""

import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    NeedsInput,
    QuoteOnly,
    Unresolved,
    resume_calculation,
    resume_calculation_task,
    run_calculation_task,
)
from app.graph.streaming_state import GraphModels
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService

QUOTE = "Học phí học kỳ = số tín chỉ đăng ký \u00d7 đơn giá tín chỉ"
CHUNK = RetrievedChunk(
    chunk_id="c_8",
    content=f"Điều 8. Học phí. {QUOTE}. Đơn giá do Hiệu trưởng ban hành.",
    source="QD-hoc-phi.pdf",
    score=0.9,
    heading_path=["Chương II", "Điều 8"],
    metadata={"document_id": "d_1"},
)
EXTRACTION = {
    "formula_id": "retrieved",
    "params": {"so_tc": 20},
    "retrieval_query": "công thức tính học phí",
}
FORMULA: dict[str, Any] = {
    "expression": "so_tc * don_gia",
    "variables": [
        {"name": "so_tc", "label": "Số tín chỉ đăng ký", "unit": "TC", "min": 1, "max": 40},
        {"name": "don_gia", "label": "Đơn giá tín chỉ", "unit": "đồng", "min": 0, "max": 10000000},
    ],
    "result_label": "Học phí học kỳ",
    "values": {"don_gia": 420000},
    "source_chunk_id": "c_8",
    "source_quote": QUOTE,
}


def _scripted(*outputs: object) -> tuple[FunctionModel, list[str]]:
    """Answers each call with the next scripted output; records the prompts."""

    remaining = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs]
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        return ModelResponse(parts=[TextPart(content=remaining.pop(0))])

    return FunctionModel(function=respond), prompts


def _deps(model: FunctionModel, chunks: list[RetrievedChunk] | None = None) -> CalculationDeps:
    return CalculationDeps(
        models=GraphModels(
            classification=model,
            query_transformation="unused",
            generation="unused",
            retrieval=FakeRetrievalService([CHUNK] if chunks is None else chunks),
        )
    )


def _found(**overrides: Any) -> dict[str, Any]:
    return {"status": "found", "formula": {**FORMULA, **overrides}, "candidates": []}


VERIFIED = {"equivalent": True, "reason": "khớp"}


@pytest.mark.asyncio
async def test_verified_formula_is_computed_with_its_source() -> None:
    model, prompts = _scripted(EXTRACTION, _found(), VERIFIED)
    outcome = await run_calculation_task("T1", "20 tín thì học phí bao nhiêu?", _deps(model))
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs) == {"Học phí học kỳ": "8400000"}
    assert outcome.source is not None
    assert (outcome.source.chunk_id, outcome.source.document_id) == ("c_8", "d_1")
    assert outcome.source.chunk_hash.startswith("sha256:")
    # The verifier never sees the student's question.
    assert "20 tín" not in prompts[2] and QUOTE in prompts[2]


@pytest.mark.asyncio
async def test_missing_variable_becomes_a_number_question() -> None:
    model, _ = _scripted(EXTRACTION, _found(values={}), VERIFIED)
    outcome = await run_calculation_task("T1", "học phí bao nhiêu?", _deps(model))
    assert isinstance(outcome, NeedsInput)
    assert [(q["field"], q["number"]["max"]) for q in outcome.questions] == [
        ("don_gia", "10000000")
    ]
    assert outcome.known_params == {"so_tc": 20}

    resumed = resume_calculation("T1", outcome.plan, outcome.known_params, {"don_gia": "420000"})
    assert isinstance(resumed, Computed)


@pytest.mark.parametrize(
    ("formula_output", "verifier", "check"),
    [
        (_found(source_chunk_id="c_999"), None, 1),
        (_found(source_quote="Học phí = số tín chỉ \u00d7 500.000"), None, 2),
        (_found(expression="so_tc ** don_gia"), None, 3),
        (_found(expression="so_tc * don_gia * 0.9"), None, 5),
        (
            _found(
                expression="so_tc * don_gia + phi_bh",
                variables=[*FORMULA["variables"], {"name": "phi_bh", "label": "Phí bảo hiểm"}],
            ),
            None,
            6,
        ),
        (_found(), {"equivalent": False, "reason": "sai"}, 7),
        (_found(), "không phải json", 7),
        ("rác", None, 0),
    ],
)
@pytest.mark.asyncio
async def test_each_check_fails_closed(
    formula_output: object, verifier: object, check: int, caplog: pytest.LogCaptureFixture
) -> None:
    outputs: list[object] = [EXTRACTION, formula_output]
    if verifier is not None:
        outputs.append(verifier)
    model, _ = _scripted(*outputs)
    outcome = await run_calculation_task("T1", "học phí?", _deps(model))
    assert outcome == Unresolved("T1", "formula_invalid")
    assert f"check={check}" in caplog.text


@pytest.mark.asyncio
async def test_out_of_range_value_is_asked_again() -> None:
    model, _ = _scripted(EXTRACTION, _found(values={"don_gia": 99_000_000}), VERIFIED)
    outcome = await run_calculation_task("T1", "học phí?", _deps(model))
    assert isinstance(outcome, NeedsInput)
    assert outcome.questions[0]["field"] == "don_gia"
    assert "bạn nhập 99000000" in outcome.questions[0]["prompt"]


AMBIGUOUS = {
    "status": "ambiguous",
    "formula": None,
    "candidates": [
        {"summary": "Khoá 2023: theo tín chỉ", "source_chunk_id": "c_8"},
        {"summary": "Khoá 2024: trọn gói theo năm", "source_chunk_id": "c_8"},
        {"summary": "bịa", "source_chunk_id": "c_404"},
    ],
}


@pytest.mark.asyncio
async def test_ambiguous_asks_the_case_on_the_panel_without_computing() -> None:
    model, _ = _scripted(EXTRACTION, AMBIGUOUS)
    outcome = await run_calculation_task("T1", "học phí?", _deps(model))
    assert isinstance(outcome, NeedsInput)
    assert outcome.lead is not None and "chọn trường hợp" in outcome.lead
    [question] = outcome.questions
    assert (question["kind"], question["field"], question["allow_other"]) == (
        "choice",
        "formula_case",
        False,
    )
    assert [(o["id"], o["label"]) for o in question["options"]] == [
        ("c1", "Khoá 2023: theo tín chỉ"),
        ("c2", "Khoá 2024: trọn gói theo năm"),
    ]
    assert (
        outcome.plan.retrieved is None and outcome.plan.retrieval_query == "công thức tính học phí"
    )
    assert outcome.known_params == {"so_tc": 20}


@pytest.mark.asyncio
async def test_chosen_case_reads_only_that_formula_then_computes() -> None:
    model, _ = _scripted(EXTRACTION, AMBIGUOUS)
    asked = await run_calculation_task("T1", "học phí?", _deps(model))
    assert isinstance(asked, NeedsInput)

    model, prompts = _scripted(_found(), VERIFIED)
    outcome = await resume_calculation_task(
        "T1", "học phí?", asked.plan, asked.known_params, {"formula_case": "c1"}, _deps(model)
    )
    assert isinstance(outcome, Computed)
    assert dict(outcome.result.outputs) == {"Học phí học kỳ": "8400000"}
    assert "<chosen_case>Khoá 2023: theo tín chỉ</chosen_case>" in prompts[0]


@pytest.mark.asyncio
async def test_still_ambiguous_after_choosing_is_never_asked_again() -> None:
    model, _ = _scripted(EXTRACTION, AMBIGUOUS)
    asked = await run_calculation_task("T1", "học phí?", _deps(model))
    assert isinstance(asked, NeedsInput)
    model, _ = _scripted(AMBIGUOUS)
    outcome = await resume_calculation_task(
        "T1", "học phí?", asked.plan, asked.known_params, {"formula_case": "c2"}, _deps(model)
    )
    assert outcome == Unresolved("T1", "formula_invalid")


@pytest.mark.asyncio
async def test_ambiguous_with_one_usable_candidate_fails_closed() -> None:
    one = {**AMBIGUOUS, "candidates": AMBIGUOUS["candidates"][:1]}
    model, _ = _scripted(EXTRACTION, one)
    assert await run_calculation_task("T1", "học phí?", _deps(model)) == Unresolved(
        "T1", "formula_invalid"
    )


@pytest.mark.asyncio
async def test_not_found_and_no_chunks_never_compute() -> None:
    model, _ = _scripted(EXTRACTION, {"status": "not_found", "formula": None, "candidates": []})
    assert await run_calculation_task("T1", "q", _deps(model)) == Unresolved(
        "T1", "formula_not_found"
    )
    model, prompts = _scripted(EXTRACTION)
    assert await run_calculation_task("T1", "q", _deps(model, chunks=[])) == Unresolved(
        "T1", "formula_not_found"
    )
    assert len(prompts) == 1  # no formula call without chunks


@pytest.mark.asyncio
async def test_kill_switch_only_quotes_the_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_CALC_RETRIEVED_FORMULA_ENABLED", False)
    model, prompts = _scripted(EXTRACTION, _found())
    outcome = await run_calculation_task("T1", "học phí?", _deps(model))
    assert isinstance(outcome, QuoteOnly)
    assert outcome.source.source_quote == QUOTE
    assert len(prompts) == 2  # no verifier call, nothing computed
