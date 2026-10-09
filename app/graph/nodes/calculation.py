"""CalculationNode - one `academic_calculation` task to a computed result, the
questions still needed, or an honest "cannot calculate this".

The LLM only reads: it picks the formula (after rule-based routing) and copies
the numbers the student actually gave. Missing or invalid parameters become
panel questions built from `ParamSpec`, never invented by the model. Every
number shown comes from `app/calculation/`. Spec: docs/specs/SPEC-calculation-node.md.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.calculation.expression import (
    FormulaRejected,
    FormulaVariable,
    RetrievedFormula,
    evaluate,
    validate,
)
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
from app.calculation.provenance import constants_anchored, quote_is_in_chunk, variables_anchored
from app.calculation.result import CalculationInputError, CalculationResult
from app.calculation.solver import Domain, Goal
from app.calculation.target import Target, Unknown, check_known, grade_goal, solve_one, solve_two
from app.core.config import settings
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.graph.streaming_state import GraphModels
from app.rag.prompting import get_templates
from app.schemas.clarification import (
    LETTER_GRADES,
    CalculationPlan,
    FormulaSource,
    FormulaVariableSpec,
    LastCalculation,
    RetrievedFormulaPlan,
    SolveGoal,
    SolveSpec,
)
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

BUILTIN_IDS: frozenset[str] = frozenset(FORMULAS)
UnresolvedReason = Literal[
    "extraction_failed",
    "formula_not_found",
    "formula_ambiguous",
    "formula_invalid",
    "target_unsupported",
]
# Upper bound for a regulation variable declared without a max (money...).
RETRIEVED_VALUE_MAX = Decimal(1_000_000_000)


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


@dataclass(frozen=True)
class QuoteOnly:
    """The formula was found and verified as a real quote, but computing regulation
    formulas is switched off: show the quote and its source, no numbers."""

    task_id: str
    source: FormulaSource


TaskOutcome = Computed | NeedsInput | Unresolved | QuoteOnly


@dataclass(frozen=True)
class CalculationDeps:
    """LLM wiring shared by every call this node makes in one turn."""

    models: GraphModels
    security: AcademicSecurityContext = field(default_factory=AcademicSecurityContext)
    on_attempt: AttemptRecorder | None = None
    on_failover: FailoverCallback | None = None
    budget: BudgetContext | None = None
    # The conversation's latest computed calculation, for follow-ups that reuse it.
    previous: LastCalculation | None = None


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
class SolveRequest:
    """A target question as the extractor read it. `unknowns` may be empty for a
    regulation formula not retrieved yet - the formula agent names them then."""

    unknowns: list[str]
    goal: SolveGoal
    want: Literal["min", "max"]


@dataclass(frozen=True)
class Extraction:
    formula_id: FormulaId | Literal["retrieved", "previous"]
    params: dict[str, JsonValue]
    retrieval_query: str | None
    solve: SolveRequest | None = None


def parse_solve(raw: object) -> SolveRequest | None:
    """The extractor's `solve` object, or None when absent or unusable. A letter-grade
    goal stays a letter here; it is turned into a number per formula."""

    if not isinstance(raw, dict):
        return None
    unknowns_raw = raw.get("unknowns")
    unknowns = [str(name) for name in unknowns_raw][:2] if isinstance(unknowns_raw, list) else []
    goal_raw = raw.get("goal")
    if not isinstance(goal_raw, dict):
        return None
    grade = goal_raw.get("grade")
    try:
        if isinstance(grade, str) and grade.strip().upper() in LETTER_GRADES:
            band = grade_goal(grade)
            assert band is not None
            goal = SolveGoal(comparator=">=", value=band.value, grade=grade.strip().upper())
        else:
            goal = SolveGoal.model_validate(
                {"comparator": goal_raw.get("comparator", ">="), "value": goal_raw.get("value")}
            )
    except (ValueError, TypeError):
        return None
    want: Literal["min", "max"] = "min" if goal.comparator == ">=" else "max"
    if raw.get("want") in ("min", "max"):
        want = "max" if raw.get("want") == "max" else "min"
    return SolveRequest(unknowns, goal, want)


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


def parse_extraction(
    raw: str,
    *,
    allowed: list[str],
    query: str,
    previous: LastCalculation | None = None,
) -> Extraction | None:
    """None (fail closed) for unparseable output or a formula id we don't know.
    "previous" (reuse the last calculation) is only accepted when there is one."""

    data = _load_json_object(raw)
    if data is None:
        return None
    formula_id = data.get("formula_id")
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    solve = parse_solve(data.get("solve"))
    if formula_id == "previous":
        # Only when the model says so explicitly: a new calculation of the same formula
        # must not silently inherit the previous numbers.
        return None if previous is None else Extraction("previous", params, None, solve)
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
            solve,
        )
    if formula_id not in BUILTIN_IDS:
        return None
    return Extraction(formula_id, params, None, solve)


def _previous_block(previous: LastCalculation) -> str:
    plan = previous.plan
    described: dict[str, Any] = {"formula_id": "previous", "params": previous.params}
    if plan.formula_id != "retrieved":
        described["formula"] = FORMULAS[plan.formula_id].title
    elif plan.retrieved is not None:
        described["formula"] = plan.retrieved.result_label
        described["variables"] = [
            {"name": v.name, "label": v.label} for v in plan.retrieved.variables
        ]
    return (
        "\n\n<previous_calculation>"
        + json.dumps(described, ensure_ascii=False)
        + "</previous_calculation>"
    )


async def extract_request(query: str, deps: CalculationDeps) -> Extraction | None:
    allowed: list[str] = list(route_builtin(query))
    prompt = query
    if deps.previous is not None:
        # A follow-up may reuse the previous calculation: the router must not force
        # another formula on it, so "previous" joins whatever the router matched.
        prompt += _previous_block(deps.previous)
        if allowed:
            allowed = [*allowed, "previous"]
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
    return parse_extraction(raw, allowed=allowed, query=query, previous=deps.previous)


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
    if spec.kind in ("number", "number_list", "number_or_list"):
        draft["number"] = {
            "min": plain(spec.min) if spec.min is not None else "0",
            "max": plain(spec.max) if spec.max is not None else "0",
            "step": plain(spec.step) if spec.step is not None else "1",
            "unit": spec.unit,
        }
    if spec.kind in ("number_list", "number_or_list", "course_table"):
        draft["max_items"] = spec.max_items
    return draft


def _input_questions(
    exc: CalculationInputError, spec_of: Callable[[str], dict[str, Any] | None]
) -> tuple[list[dict[str, Any]], set[str]]:
    """One question per rejected field (first reason wins) and the fields to forget."""

    reasons: dict[str, str] = {}
    for error in exc.errors:
        reasons.setdefault(error.field, error.reason)
    questions: list[dict[str, Any]] = []
    for name, reason in reasons.items():
        draft = spec_of(name)
        if draft is not None:
            if reason != "còn thiếu":
                draft["prompt"] = f"{draft['prompt']} ({reason})"[:200]
            questions.append(draft)
    return questions, set(reasons)


def builtin_solve_spec(formula_id: FormulaId, request: SolveRequest) -> SolveSpec | None:
    """The solve spec for a built-in formula, or None when this formula cannot answer
    that target question (unknown not solvable, letter goal on a non-score result)."""

    formula = FORMULAS[formula_id]
    if formula.target_label is None or not request.unknowns:
        return None
    if request.goal.grade is not None and not formula.target_is_score10:
        return None
    for name in request.unknowns:
        spec = param_spec(formula_id, name)
        if spec is None or spec.solve_step is None:
            return None
    return SolveSpec(unknowns=request.unknowns, goal=request.goal, want=request.want)


def _builtin_unknown(formula_id: FormulaId, name: str) -> Unknown:
    spec = param_spec(formula_id, name)
    assert spec is not None and spec.solve_step is not None
    assert spec.min is not None and spec.max is not None
    return Unknown(
        name=name,
        symbol=spec.symbol or spec.tab_label,
        domain=Domain(spec.min, spec.max, spec.solve_step),
        half_point=spec.half_point,
    )


def _target(label: str, solve: SolveSpec) -> Target:
    goal = Goal(solve.goal.comparator, solve.goal.value)
    return Target(label=label, goal=goal, grade=solve.goal.grade, want=solve.want)


def solve_target(
    compute: Callable[[Mapping[str, object]], CalculationResult],
    known: Mapping[str, object],
    unknowns: list[Unknown],
    target: Target,
) -> CalculationResult:
    """Raises `CalculationInputError` for the given (non-unknown) parameters."""

    check_known(compute, known, unknowns)
    if len(unknowns) == 1:
        return solve_one(compute, known, unknowns[0], target)
    return solve_two(compute, known, (unknowns[0], unknowns[1]), target)


def builtin_outcome(
    task_id: str,
    formula_id: FormulaId,
    params: Mapping[str, JsonValue],
    solve: SolveSpec | None = None,
) -> Computed | NeedsInput:
    """Compute (or solve the target question), or list what to ask: missing params
    plus the ones Python rejected. The unknowns of a target question are never asked."""

    plan = CalculationPlan(formula_id=formula_id, solve=solve)
    unknowns = solve.unknowns if solve is not None else []
    known = {
        key: value
        for key, value in params.items()
        if param_spec(formula_id, key) and key not in unknowns
    }
    missing = missing_params(formula_id, known, unknowns)
    if missing:
        return NeedsInput(task_id, plan, known, [question_for(spec) for spec in missing])

    def spec_of(name: str) -> dict[str, Any] | None:
        spec = param_spec(formula_id, name)
        return None if spec is None else question_for(spec)

    try:
        if solve is None:
            result = calculate(formula_id, known)
        else:
            formula = FORMULAS[formula_id]
            assert formula.target_label is not None
            result = solve_target(
                formula.compute,
                known,
                [_builtin_unknown(formula_id, name) for name in unknowns],
                _target(formula.target_label, solve),
            )
    except CalculationInputError as exc:
        questions, rejected = _input_questions(exc, spec_of)
        for name in rejected:
            known.pop(name, None)
        return NeedsInput(task_id, plan, known, questions)
    return Computed(task_id, result, plan, dict(known))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_calculation_task(task_id: str, query: str, deps: CalculationDeps) -> TaskOutcome:
    extraction = await extract_request(query, deps)
    if extraction is None:
        return Unresolved(task_id, "extraction_failed")
    if extraction.formula_id == "previous":
        assert deps.previous is not None
        return previous_outcome(task_id, deps.previous, extraction)
    if extraction.formula_id == "retrieved":
        return await retrieved_outcome(task_id, query, extraction, deps)
    return _builtin_with_request(
        task_id, extraction.formula_id, extraction.params, extraction.solve
    )


def _builtin_with_request(
    task_id: str,
    formula_id: FormulaId,
    params: Mapping[str, JsonValue],
    request: SolveRequest | None,
) -> TaskOutcome:
    if request is None:
        return builtin_outcome(task_id, formula_id, params)
    solve = builtin_solve_spec(formula_id, request)
    if solve is None:
        return Unresolved(task_id, "target_unsupported")
    return builtin_outcome(task_id, formula_id, params, solve)


def previous_outcome(
    task_id: str, previous: LastCalculation, extraction: Extraction
) -> TaskOutcome:
    """A follow-up on the last calculation: its formula (a regulation formula stays the
    one already verified - no new retrieval) with the new numbers on top of the old."""

    plan = previous.plan
    params = {**previous.params, **extraction.params}
    if plan.formula_id != "retrieved":
        return _builtin_with_request(task_id, plan.formula_id, params, extraction.solve)
    assert plan.retrieved is not None
    solve: SolveSpec | None = None
    if extraction.solve is not None:
        solve = retrieved_solve_spec(plan.retrieved, extraction.solve, extraction.solve.unknowns)
        if solve is None:
            return Unresolved(task_id, "target_unsupported")
    return retrieved_values_outcome(
        task_id,
        CalculationPlan(formula_id="retrieved", retrieved=plan.retrieved, solve=solve),
        params,
    )


# ---------------------------------------------------------------------------
# Regulation formulas (Qdrant) - 7 fail-closed checks before anything is computed
# ---------------------------------------------------------------------------

RETRIEVED_CHUNKS_LIMIT = 5


def build_formula_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_calculation_formula,
        model_settings=auxiliary_model_settings(model),
    )


def build_formula_verifier_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_calculation_formula_verifier,
        model_settings=auxiliary_model_settings(model),
    )


async def _ask(
    factory: Callable[[Model | str], Agent[None, str]], prompt: str, deps: CalculationDeps
) -> str | None:
    try:
        return await run_agent_text_with_failover(
            factory(deps.models.classification),
            prompt,
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=factory,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.llm_call_failed agent=%s", factory.__name__, exc_info=True)
        return None


def _chunk_hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _formula_prompt(
    query: str, chunks: list[RetrievedChunk], known: Mapping[str, JsonValue]
) -> str:
    blocks = "\n\n".join(
        f'<chunk id="{chunk.chunk_id}" source="{chunk.source}">\n{chunk.content}\n</chunk>'
        for chunk in chunks
    )
    return (
        f"<regulation_chunks>\n{blocks}\n</regulation_chunks>\n\n"
        f"<question>{query}</question>\n"
        f"<known_values>{json.dumps(dict(known), ensure_ascii=False)}</known_values>"
    )


def _rejected(task_id: str, check: int, detail: str = "") -> Unresolved:
    logger.warning("calculation.formula_rejected check=%d %s", check, detail)
    return Unresolved(task_id, "formula_invalid")


def _to_formula(plan: RetrievedFormulaPlan) -> RetrievedFormula:
    return RetrievedFormula(
        expression=plan.expression,
        variables=tuple(
            FormulaVariable(name=v.name, label=v.label, unit=v.unit, min=v.min, max=v.max)
            for v in plan.variables
        ),
        result_label=plan.result_label,
    )


def _variable_question(variable: FormulaVariableSpec) -> dict[str, Any]:
    return {
        "tab_label": variable.label[:24],
        "prompt": variable.label + (f" ({variable.unit})" if variable.unit else ""),
        "kind": "number",
        "origin": "calculation",
        "field": variable.name,
        "number": {
            "min": plain(variable.min) if variable.min is not None else "0",
            "max": plain(variable.max) if variable.max is not None else "1000000000",
            "step": "0.01",
            "unit": variable.unit,
        },
    }


def retrieved_solve_spec(
    retrieved: RetrievedFormulaPlan, request: SolveRequest, unknowns: list[str]
) -> SolveSpec | None:
    """Unknowns must be declared variables; a letter-grade goal is not accepted (we
    cannot know the regulation's result is a 10-scale score)."""

    names = {variable.name for variable in retrieved.variables}
    if not unknowns or request.goal.grade is not None or not set(unknowns) <= names:
        return None
    return SolveSpec(unknowns=unknowns[:2], goal=request.goal, want=request.want)


def _retrieved_unknown(variable: FormulaVariableSpec) -> Unknown:
    """Credits/counts are whole numbers; anything else is searched to 0.01."""

    unit = (variable.unit or "").lower()
    whole = any(word in unit for word in ("tc", "tín", "môn", "học kỳ", "học kì", "lần"))
    low = variable.min if variable.min is not None else Decimal(0)
    high = variable.max if variable.max is not None else RETRIEVED_VALUE_MAX
    return Unknown(
        name=variable.name,
        symbol=variable.label,
        domain=Domain(low, high, Decimal(1) if whole else Decimal("0.01")),
    )


def retrieved_values_outcome(
    task_id: str, plan: CalculationPlan, values: Mapping[str, JsonValue]
) -> Computed | NeedsInput | Unresolved:
    """Compute (or solve the target question on) a verified regulation formula, or ask
    for the variables still missing or out of range. Shared by the first turn, the
    resume turn and a follow-up on the previous calculation."""

    assert plan.retrieved is not None
    retrieved = plan.retrieved
    formula = _to_formula(retrieved)
    unknowns = plan.solve.unknowns if plan.solve is not None else []
    known = {
        name: value
        for name, value in values.items()
        if name not in unknowns and any(variable.name == name for variable in retrieved.variables)
    }
    by_name = {variable.name: variable for variable in retrieved.variables}

    def spec_of(name: str) -> dict[str, Any] | None:
        variable = by_name.get(name)
        return None if variable is None or name in unknowns else _variable_question(variable)

    def compute(params: Mapping[str, object]) -> CalculationResult:
        return evaluate(formula, params)

    try:
        if plan.solve is None:
            result = evaluate(formula, known)
        else:
            result = solve_target(
                compute,
                known,
                [_retrieved_unknown(by_name[name]) for name in unknowns],
                _target(retrieved.result_label, plan.solve),
            )
    except FormulaRejected as exc:
        return _rejected(task_id, 3, str(exc))
    except CalculationInputError as exc:
        questions, rejected = _input_questions(exc, spec_of)
        if not questions:  # e.g. division by zero on the expression itself
            return _rejected(task_id, 4, "; ".join(error.reason for error in exc.errors))
        for name in rejected:
            known.pop(name, None)
        return NeedsInput(task_id, plan, known, questions)
    return Computed(task_id, result, plan, known, source=retrieved.source)


async def retrieved_outcome(
    task_id: str, query: str, extraction: Extraction, deps: CalculationDeps
) -> TaskOutcome:
    retrieval_query = extraction.retrieval_query or query
    per_query = await retrieve_chunks([retrieval_query], deps.models.retrieval, deps.security)
    chunks = (per_query[0] if per_query else [])[:RETRIEVED_CHUNKS_LIMIT]
    if not chunks:
        return Unresolved(task_id, "formula_not_found")

    raw = await _ask(build_formula_agent, _formula_prompt(query, chunks, extraction.params), deps)
    data = _load_json_object(raw) if raw is not None else None
    if data is None:
        return _rejected(task_id, 0, "formula output unreadable")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    status = data.get("status")
    if status == "ambiguous":
        candidates = [
            {
                "summary": str(item.get("summary", ""))[:300],
                "source": by_id[item["source_chunk_id"]].source,
            }
            for item in data.get("candidates") or []
            if isinstance(item, dict) and item.get("source_chunk_id") in by_id
        ]
        return Unresolved(task_id, "formula_ambiguous", candidates)
    if status != "found" or not isinstance(data.get("formula"), dict):
        return Unresolved(task_id, "formula_not_found")
    found: dict[str, Any] = data["formula"]

    # 1. the chunk is one retrieved (permission-filtered) for this student
    chunk = by_id.get(str(found.get("source_chunk_id")))
    if chunk is None:
        return _rejected(task_id, 1, "unknown chunk id")
    # 2. the quote really is in that chunk
    quote = str(found.get("source_quote") or "")
    if not quote_is_in_chunk(quote, chunk.content):
        return _rejected(task_id, 2, "quote not in chunk")
    source = FormulaSource(
        chunk_id=chunk.chunk_id,
        document_id=str(chunk.metadata.get("document_id") or "") or None,
        source=chunk.source,
        heading_path=list(chunk.heading_path),
        chunk_hash=_chunk_hash(chunk.content),
        source_quote=quote[:600],
    )
    if not settings.CHAT_CALC_RETRIEVED_FORMULA_ENABLED:
        return QuoteOnly(task_id, source)

    try:
        retrieved = RetrievedFormulaPlan(
            expression=str(found.get("expression") or ""),
            variables=found.get("variables") or [],
            result_label=str(found.get("result_label") or "Kết quả")[:80],
            source=source,
        )
    except ValueError as exc:
        return _rejected(task_id, 3, f"bad formula shape: {exc}")
    formula = _to_formula(retrieved)
    # 3. allowlisted grammar, limits, declared == used variables
    try:
        tree = validate(formula)
    except FormulaRejected as exc:
        return _rejected(task_id, 3, str(exc))
    # 5. every coefficient appears in the quote
    if constants_anchored(tree, quote):
        return _rejected(task_id, 5, "constant not in quote")
    # 6. every variable is named in the quote
    if variables_anchored(formula, quote):
        return _rejected(task_id, 6, "variable not in quote")
    # 7. an independent verifier agrees the expression says what the quote says
    if not await _verified(retrieved, deps):
        return _rejected(task_id, 7, "verifier disagreed")

    solve: SolveSpec | None = None
    if extraction.solve is not None:
        raw_unknowns = found.get("unknowns")
        unknowns = (
            [str(name) for name in raw_unknowns] if isinstance(raw_unknowns, list) else []
        ) or extraction.solve.unknowns
        solve = retrieved_solve_spec(retrieved, extraction.solve, unknowns)
        if solve is None:
            return Unresolved(task_id, "target_unsupported")
    plan = CalculationPlan(formula_id="retrieved", retrieved=retrieved, solve=solve)
    raw_values = found.get("values")
    values: dict[str, JsonValue] = dict(raw_values) if isinstance(raw_values, dict) else {}
    # 4. values are declared variables within their ranges (enforced while computing)
    return retrieved_values_outcome(task_id, plan, {**extraction.params, **values})


async def _verified(plan: RetrievedFormulaPlan, deps: CalculationDeps) -> bool:
    variables = "\n".join(f"- {v.name}: {v.label}" for v in plan.variables)
    prompt = (
        f"<source_quote>{plan.source.source_quote}</source_quote>\n"
        f"<expression>{plan.expression}</expression>\n<variables>\n{variables}\n</variables>"
    )
    raw = await _ask(build_formula_verifier_agent, prompt, deps)
    data = _load_json_object(raw) if raw is not None else None
    return bool(data is not None and data.get("equivalent") is True)


def resume_builtin(
    task_id: str,
    plan: CalculationPlan,
    known_params: Mapping[str, JsonValue],
    answers: Mapping[str, JsonValue],
) -> Computed | NeedsInput:
    """A panel answer for a built-in formula: no LLM, no retrieval - just compute."""

    assert plan.formula_id != "retrieved"
    formula_id: FormulaId = plan.formula_id
    return builtin_outcome(task_id, formula_id, {**known_params, **answers}, plan.solve)


def resume_calculation(
    task_id: str,
    plan: CalculationPlan,
    known_params: Mapping[str, JsonValue],
    answers: Mapping[str, JsonValue],
) -> Computed | NeedsInput | Unresolved:
    """A panel answer: never calls the extractor or retrieval again."""

    if plan.formula_id == "retrieved":
        return retrieved_values_outcome(task_id, plan, {**known_params, **answers})
    return resume_builtin(task_id, plan, known_params, answers)


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
    "target_unsupported": (
        "Mình chưa tính ngược được cho đại lượng này. Mình có thể tìm điểm cần đạt của một "
        "hoặc hai cột điểm (ví dụ điểm cuối kỳ, giữa kỳ) để học phần đạt một điểm số hoặc "
        "điểm chữ, bạn hỏi lại theo cách đó giúp mình nhé."
    ),
}
