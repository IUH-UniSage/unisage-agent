"""CalculationNode - one `academic_calculation` task to a computed result, the
questions still needed, or an honest "cannot calculate this".

The LLM only reads: it picks the formula (after rule-based routing) and copies
the numbers the student actually gave. Missing or invalid parameters become
panel questions built from `ParamSpec`, never invented by the model. Every
number shown comes from `app/calculation/`. Spec: docs/specs/SPEC-calculation-node.md.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.calculation.formulas import (
    FORMULAS,
    FormulaId,
    ParamSpec,
    calculate,
    describe_builtin_formulas,
    missing_params,
    param_spec,
    plain,
    route_builtin,
)
from app.calculation.result import CalculationInputError, CalculationResult
from app.core.config import settings
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.graph.streaming_state import GraphModels
from app.rag.prompting import get_templates
from app.schemas.clarification import CalculationPlan, FormulaSource

logger = logging.getLogger(__name__)

BUILTIN_IDS: frozenset[str] = frozenset(FORMULAS)
UnresolvedReason = Literal[
    "extraction_failed", "formula_not_found", "formula_ambiguous", "formula_invalid"
]


@dataclass(frozen=True)
class Computed:
    task_id: str
    result: CalculationResult
    plan: CalculationPlan
    params: dict[str, JsonValue]
    source: FormulaSource | None = None


@dataclass(frozen=True)
class NeedsInput:
    task_id: str
    plan: CalculationPlan
    known_params: dict[str, JsonValue]
    questions: list[dict[str, Any]]


@dataclass(frozen=True)
class Unresolved:
    task_id: str
    reason: UnresolvedReason
    candidates: list[dict[str, str]] = field(default_factory=list)
    # Set when the formula was found but computing it is switched off (quote only).
    source: FormulaSource | None = None


TaskOutcome = Computed | NeedsInput | Unresolved


@dataclass(frozen=True)
class CalculationDeps:
    """LLM wiring shared by every call this node makes in one turn."""

    models: GraphModels
    on_attempt: AttemptRecorder | None = None
    on_failover: FailoverCallback | None = None
    budget: BudgetContext | None = None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def build_calculation_extractor_agent(model: Model | str) -> Agent[None, str]:
    templates = get_templates()
    return Agent(
        model=model,
        system_prompt=templates.agent_calculation_extractor.replace(
            "{builtin_formulas}", describe_builtin_formulas()
        ),
        model_settings=auxiliary_model_settings(model),
    )


@dataclass(frozen=True)
class Extraction:
    formula_id: FormulaId | Literal["retrieved"]
    params: dict[str, JsonValue]
    retrieval_query: str | None


def _load_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip().strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_extraction(raw: str, *, allowed: list[FormulaId], query: str) -> Extraction | None:
    """None (fail closed) for unparseable output or a formula id we don't know."""

    data = _load_json_object(raw)
    if data is None:
        return None
    formula_id = data.get("formula_id")
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    if len(allowed) == 1:
        # The rule-based router already decided; the model only extracted numbers.
        if formula_id != allowed[0]:
            logger.info("calculation.router_disagreement routed=%s llm=%s", allowed[0], formula_id)
        formula_id = allowed[0]
    elif allowed and formula_id not in allowed:
        return None
    if formula_id == "retrieved":
        retrieval_query = data.get("retrieval_query")
        return Extraction(
            "retrieved",
            params,
            retrieval_query if isinstance(retrieval_query, str) and retrieval_query else query,
        )
    if formula_id not in BUILTIN_IDS:
        return None
    return Extraction(formula_id, params, None)


async def extract_request(query: str, deps: CalculationDeps) -> Extraction | None:
    allowed = route_builtin(query)
    prompt = query
    if len(allowed) > 1:
        prompt += "\n\n<allowed_formula_ids>" + ", ".join(allowed) + "</allowed_formula_ids>"
    try:
        raw = await run_agent_text_with_failover(
            build_calculation_extractor_agent(deps.models.classification),
            prompt,
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=build_calculation_extractor_agent,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.extractor_failed", exc_info=True)
        return None
    return parse_extraction(raw, allowed=allowed, query=query)


# ---------------------------------------------------------------------------
# Questions (built from ParamSpec - never by the LLM)
# ---------------------------------------------------------------------------


def question_for(spec: ParamSpec, reason: str | None = None) -> dict[str, Any]:
    prompt = spec.label if reason is None else f"{spec.label} ({reason})"
    draft: dict[str, Any] = {
        "tab_label": spec.tab_label,
        "prompt": prompt[:200],
        "kind": spec.kind,
        "origin": "calculation",
        "field": spec.name,
    }
    if spec.kind in ("number", "number_list"):
        draft["number"] = {
            "min": plain(spec.min) if spec.min is not None else "0",
            "max": plain(spec.max) if spec.max is not None else "0",
            "step": plain(spec.step) if spec.step is not None else "1",
            "unit": spec.unit,
        }
    if spec.kind in ("number_list", "course_table"):
        draft["max_items"] = spec.max_items
    return draft


def builtin_outcome(
    task_id: str, formula_id: FormulaId, params: Mapping[str, JsonValue]
) -> Computed | NeedsInput:
    """Compute, or list what to ask: missing params plus the ones Python rejected."""

    plan = CalculationPlan(formula_id=formula_id)
    known = {key: value for key, value in params.items() if param_spec(formula_id, key)}
    missing = missing_params(formula_id, known)
    if not missing:
        try:
            return Computed(task_id, calculate(formula_id, known), plan, dict(known))
        except CalculationInputError as exc:
            reasons: dict[str, str] = {}
            for error in exc.errors:
                reasons.setdefault(error.field, error.reason)
            questions = []
            for name, reason in reasons.items():
                spec = param_spec(formula_id, name)
                if spec is not None:
                    questions.append(question_for(spec, reason))
                    known.pop(name, None)
            return NeedsInput(task_id, plan, known, questions)
    return NeedsInput(task_id, plan, known, [question_for(spec) for spec in missing])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_calculation_task(task_id: str, query: str, deps: CalculationDeps) -> TaskOutcome:
    extraction = await extract_request(query, deps)
    if extraction is None:
        return Unresolved(task_id, "extraction_failed")
    if extraction.formula_id == "retrieved":
        # Regulation formulas (Qdrant) are handled in a follow-up change (UNISAGE-99 T17).
        return Unresolved(task_id, "formula_not_found")
    return builtin_outcome(task_id, extraction.formula_id, extraction.params)


def resume_builtin(
    task_id: str,
    plan: CalculationPlan,
    known_params: Mapping[str, JsonValue],
    answers: Mapping[str, JsonValue],
) -> Computed | NeedsInput:
    """A panel answer for a built-in formula: no LLM, no retrieval - just compute."""

    assert plan.formula_id != "retrieved"
    formula_id: FormulaId = plan.formula_id
    return builtin_outcome(task_id, formula_id, {**known_params, **answers})


# Fixed sentences for a task that cannot be calculated (rendered by Python, no LLM).
UNRESOLVED_MESSAGES: dict[UnresolvedReason, str] = {
    "extraction_failed": (
        "Mình chưa hiểu bạn cần tính gì, bạn nói rõ hơn giúp mình nhé (ví dụ: tính GPA, "
        "tính điểm tổng kết học phần, quy đổi điểm chữ)."
    ),
    "formula_not_found": (
        "Mình chưa tìm thấy công thức tính này trong quy chế hiện có nên chưa tính giúp bạn "
        "được. Bạn có thể liên hệ Phòng Đào tạo để được hướng dẫn nhé."
    ),
    "formula_ambiguous": (
        "Quy chế có nhiều công thức khác nhau cho trường hợp này, mình chưa biết bạn thuộc "
        "trường hợp nào:"
    ),
    "formula_invalid": (
        "Mình tìm thấy nội dung liên quan trong quy chế nhưng không chắc đã đọc đúng công "
        "thức, nên chưa tính giúp bạn. Bạn có thể liên hệ Phòng Đào tạo để được hướng dẫn nhé."
    ),
}


# Still streamed by the graph until it is wired to `run_calculation_task` (T18).
CALCULATION_PLACEHOLDER_TEMPLATE = (
    "Phần tính toán (GPA, tín chỉ, học phí) hiện đang được phát triển nên mình "
    "chưa tính giúp bạn được. Bạn có thể tự tính theo công thức trong quy chế "
    "đào tạo, hoặc liên hệ Phòng Đào tạo để được hỗ trợ nhé."
)
