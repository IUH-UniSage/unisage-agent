"""CalculationNode - one `academic_calculation` task to a result, the questions
still needed, or an honest "cannot calculate this".

Python computes, the LLM routes (docs/specs/SPEC-calculation-node.md). The extractor
(`agents/calculation_extractor.yaml`) picks one formula of the registry
(`app/calculation/formulas.py`), copies the numbers the student gave and, for a target
question ("cuối kỳ cần bao nhiêu để được A+"), names the unknown and the target; the
registry's formula is then computed forward (`calculate`) or solved for the unknown
(`solver.solved_result`). Missing or invalid numbers become panel questions built from
`ParamSpec`; values the documents hold (unit prices...) are read from Qdrant first.

A question no registered formula fits is `Deferred`: the graph answers it on the
advisory path, which shows the formula and the figures from the documents and leaves
the arithmetic to the user (decision 26/10/2026: no LLM arithmetic in this node).
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.calculation.arithmetic import (
    ARITHMETIC_SOURCE_OWN,
    ArithmeticError,
    ArithmeticRequest,
    Evaluation,
    arithmetic_result,
    evaluate,
    parse_request,
)
from app.calculation.formulas import (
    FORMULAS,
    FormulaId,
    ParamSpec,
    calculate,
    describe_builtin_formulas,
    label_for,
    missing_params,
    money,
    param_spec,
    plain,
    route_builtin,
)
from app.calculation.result import CalculationInputError, CalculationResult, FieldError
from app.calculation.solver import Target, Unknown, parse_target, parse_unknown, solved_result
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.graph.calc_requests import missing_operands
from app.graph.evidence import levels_for
from app.graph.metadata_fields import canonical_field, is_profile_field
from app.graph.nodes.cross_encoder_rerank import (
    CROSS_ENCODER_SKIPPED,
    cross_encoder_rerank,
)
from app.graph.nodes.llm_rerank import (
    LLM_RERANK_SKIPPED,
    build_llm_rerank_agent,
    llm_rerank,
)
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.graph.streaming_state import AdminWarning, GraphModels
from app.integrations.cross_encoder_client import CrossEncoderClient
from app.rag.prompting import append_current_time, get_templates
from app.rag.prompting.citations import source_title
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import CalculationPlan, CalculationTarget, LastCalculation
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

BUILTIN_IDS: frozenset[str] = frozenset(FORMULAS)
UnresolvedReason = Literal["legacy_llm", "bad_target", "not_resumable"]
# Plans that are not a registry formula (never redone with new numbers, never resumed).
NON_REGISTRY_PLANS = frozenset({"llm", "arithmetic"})
# Documents read for one formula's lookups.
LLM_CHUNKS_LIMIT = 5


@dataclass(frozen=True)
class Computed:
    """A registered formula computed (or solved) by Python."""

    task_id: str
    result: CalculationResult
    plan: CalculationPlan
    params: dict[str, JsonValue]
    # Lines naming each value read from the documents, with its [n] marker, and the
    # chunk behind each marker; renumbered for the whole message before it is shown.
    notes: tuple[str, ...] = ()
    sources: list[tuple[int, RetrievedChunk]] = field(default_factory=list)


@dataclass(frozen=True)
class NeedsInput:
    task_id: str
    plan: CalculationPlan
    known_params: dict[str, JsonValue]
    questions: list[dict[str, Any]]
    # Shown above the panel instead of the generic lead.
    lead: str | None = None


@dataclass(frozen=True)
class Unresolved:
    task_id: str
    reason: UnresolvedReason


TaskOutcome = Computed | NeedsInput | Unresolved


@dataclass(frozen=True)
class Deferred:
    """No registered formula fits: the question is answered on the advisory path (the
    documents' formula and figures, no arithmetic by the model)."""

    task_id: str
    query: str
    retrieval_query: str | None = None


@dataclass(frozen=True)
class PlannedTask:
    """A calculation task the extractor routed to a registered formula."""

    task_id: str
    query: str
    formula_id: FormulaId
    params: dict[str, JsonValue]
    unknown: Unknown | None = None
    target: Target | None = None


@dataclass(frozen=True)
class ArithmeticTask:
    """Plain arithmetic over numbers the student stated themselves (policy level A):
    computed by code from the whitelisted expression, no formula, no document."""

    task_id: str
    query: str
    request: ArithmeticRequest
    evaluation: Evaluation


Planned = PlannedTask | ArithmeticTask


@dataclass(frozen=True)
class CalculationDeps:
    """LLM wiring shared by every call this node makes in one turn."""

    models: GraphModels
    security: AcademicSecurityContext = field(default_factory=AcademicSecurityContext)
    on_attempt: AttemptRecorder | None = None
    on_failover: FailoverCallback | None = None
    budget: BudgetContext | None = None
    # The conversation's latest built-in calculation, for forward follow-ups.
    previous: LastCalculation | None = None
    # Recent chat (kept for the trace and the extractor's context).
    history: Sequence[HistoryMessage] = ()
    recent_calculations: Sequence[LastCalculation] = ()
    # Self-declared attributes, shown apart from the verified identity.
    confirmed_metadata: Mapping[str, str] = field(default_factory=dict)
    # What the student typed this turn (None on a panel answer): the task query is the
    # classifier's normalised rewrite, and a rewrite can lose a word like "không".
    user_message: str | None = None
    trace: GraphTrace | None = None
    # AI-admin warnings raised while calculating, one per code.
    warnings: list[AdminWarning] = field(default_factory=list)

    def warn(self, code: str, message: str) -> None:
        if all(warning.code != code for warning in self.warnings):
            self.warnings.append(AdminWarning(code=code, message=message))


# ---------------------------------------------------------------------------
# Extractor: which formula, which numbers, which unknown
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
    formula_id: FormulaId | Literal["none", "previous", "arithmetic"]
    params: dict[str, JsonValue]
    retrieval_query: str | None = None
    unknown: Unknown | None = None
    target: Target | None = None
    arithmetic: ArithmeticRequest | None = None


def load_json_object(raw: str) -> dict[str, Any] | None:
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


def _none_extraction(params: dict[str, JsonValue], data: Mapping[str, Any]) -> Extraction:
    retrieval_query = data.get("retrieval_query")
    query = retrieval_query.strip()[:500] if isinstance(retrieval_query, str) else ""
    return Extraction("none", params, query or None)


def _solving(
    formula_id: FormulaId, data: Mapping[str, Any]
) -> tuple[Unknown | None, Target | None]:
    """The inverse part of the output, or (None, None) - a forward calculation - when
    it is absent or unreadable (logged: the forward result is still shown)."""

    raw_unknown = data.get("solve_for")
    if raw_unknown in (None, ""):
        return None, None
    unknown = parse_unknown(raw_unknown)
    target = parse_target(formula_id, data.get("target"))
    if (
        unknown is None
        or isinstance(target, FieldError)
        or param_spec(formula_id, unknown.name) is None
    ):
        logger.warning(
            "calculation.solve_dropped formula=%s solve_for=%r target=%r",
            formula_id,
            raw_unknown,
            data.get("target"),
        )
        return None, None
    return unknown, target


_DIGIT = re.compile(r"\d")


def parse_extraction(
    raw: str,
    *,
    allowed: list[str],
    previous: LastCalculation | None = None,
    question: str = "",
) -> Extraction:
    """The model's choice of a registered formula is kept (it saw the whole registry and
    the question's target; a keyword router cannot tell "cuối kỳ cần bao nhiêu để được
    B+" from a grade conversion). Its explicit `none` is kept too. Only unreadable or
    unknown output falls back to the single formula the rule-based router matched, and
    to `none` when the router matched none or several - the advisory path then answers
    from the documents."""

    data = load_json_object(raw) or {}
    formula_id = data.get("formula_id")
    raw_params = data.get("params")
    params: dict[str, JsonValue] = dict(raw_params) if isinstance(raw_params, dict) else {}
    if formula_id == "previous" and previous_formula(previous) is not None:
        chosen = previous_formula(previous)
        assert chosen is not None
        unknown, target = _solving(chosen, data)
        return Extraction("previous", params, None, unknown, target)
    routed = [formula for formula in allowed if formula in BUILTIN_IDS]
    if formula_id == "none":
        # "tính điểm xét tuyển giúp em": no number to compute from the documents, and the
        # router matched exactly one registered formula - the student is asking for that
        # formula, so its panel asks for the inputs (seen 27/10/2026: gpt-5-mini answered
        # `none` and the turn drifted to the advisory path, which cannot ask for numbers).
        if len(routed) == 1 and not _DIGIT.search(question or ""):
            logger.info("calculation.router_recovers_bare_request formula=%s", routed[0])
            formula_id = routed[0]
        else:
            return _none_extraction(params, data)
    if formula_id == "arithmetic":
        request = parse_request(data.get("arithmetic"))
        if request is None:
            logger.warning("calculation.arithmetic_unreadable %r", data.get("arithmetic"))
            return _none_extraction(params, data)
        return Extraction("arithmetic", {}, None, arithmetic=request)
    if formula_id not in BUILTIN_IDS:
        if len(routed) != 1:
            return _none_extraction(params, data)
        logger.info("calculation.router_fallback routed=%s llm=%r", routed[0], formula_id)
        formula_id = routed[0]
    elif routed and formula_id not in routed:
        logger.info("calculation.router_disagreement routed=%s llm=%s", routed, formula_id)
    builtin: FormulaId = next(known for known in FORMULAS if known == formula_id)
    unknown, target = _solving(builtin, data)
    return Extraction(builtin, params, None, unknown, target)


def registry_id(formula_id: str) -> FormulaId | None:
    """The plan's formula when it is a registry formula; None for `llm`/`arithmetic`."""

    return next((known for known in FORMULAS if known == formula_id), None)


def previous_formula(previous: LastCalculation | None) -> FormulaId | None:
    """The registered formula the previous calculation can be redone with."""

    return registry_id(previous.plan.formula_id) if previous is not None else None


def _previous_block(previous: LastCalculation) -> str:
    known = registry_id(previous.plan.formula_id)
    described = {
        "formula_id": "previous",
        "formula": FORMULAS[known].title if known is not None else previous.title,
        "params": previous.params,
    }
    return (
        "\n\n<previous_calculation>"
        + json.dumps(described, ensure_ascii=False)
        + "</previous_calculation>"
    )


async def extract_request(query: str, deps: CalculationDeps) -> Extraction:
    allowed: list[str] = list(route_builtin(query))
    prompt = query
    previous = deps.previous
    if previous is not None:
        prompt += _previous_block(previous)
        if allowed and previous_formula(previous) is not None:
            allowed = [*allowed, "previous"]
    if len(allowed) > 1:
        prompt += "\n\n<allowed_formula_ids>" + ", ".join(allowed) + "</allowed_formula_ids>"
    original = (deps.user_message or "").strip()
    if original and original != query.strip():
        prompt += f"\n\n<original_message>{original}</original_message>"
    prompt = append_current_time(prompt, deps.confirmed_metadata)
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
        raw = ""
    return parse_extraction(raw, allowed=allowed, previous=previous, question=query)


# ---------------------------------------------------------------------------
# Registered formulas: forward calculation or inverse solving by Python
# ---------------------------------------------------------------------------


def question_for(
    spec: ParamSpec, known: Mapping[str, JsonValue], reason: str | None = None
) -> dict[str, Any]:
    label = label_for(spec, known)
    prompt = label if reason is None else f"{label} ({reason})"
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
    if spec.kind == "text":
        draft["max_length"] = 200
    return draft


def plan_for(
    formula_id: FormulaId, unknown: Unknown | None, target: Target | None
) -> CalculationPlan:
    if unknown is None or target is None:
        return CalculationPlan(formula_id=formula_id)
    return CalculationPlan(
        formula_id=formula_id,
        solve_for=unknown.text(),
        target=CalculationTarget(output=target.output, op=target.op, value=plain(target.value)),
    )


def solving_of(plan: CalculationPlan) -> tuple[Unknown | None, Target | None]:
    """The unknown and target a stored plan asks for (None, None for a forward one)."""

    known = registry_id(plan.formula_id)
    if known is None or plan.solve_for is None or plan.target is None:
        return None, None
    unknown = parse_unknown(plan.solve_for)
    target = parse_target(
        known,
        {"output": plan.target.output, "op": plan.target.op, "value": plan.target.value},
    )
    if unknown is None or isinstance(target, FieldError):
        return None, None
    return unknown, target


def builtin_outcome(
    task_id: str,
    formula_id: FormulaId,
    params: Mapping[str, JsonValue],
    *,
    unknown: Unknown | None = None,
    target: Target | None = None,
) -> TaskOutcome:
    """Compute (or solve for `unknown`), or list what to ask: missing params plus the
    ones Python rejected. The unknown itself is never asked."""

    plan = plan_for(formula_id, unknown, target)
    known = {key: value for key, value in params.items() if param_spec(formula_id, key)}
    if unknown is not None and unknown.index is None:
        known.pop(unknown.name, None)
    missing = [
        spec
        for spec in missing_params(formula_id, known)
        if unknown is None or spec.name != unknown.name
    ]
    if missing:
        return NeedsInput(task_id, plan, known, [question_for(spec, known) for spec in missing])
    try:
        if unknown is not None and target is not None:
            result = solved_result(formula_id, known, unknown, target)
        else:
            result = calculate(formula_id, known)
    except CalculationInputError as exc:
        reasons: dict[str, str] = {}
        for error in exc.errors:
            reasons.setdefault(error.field, error.reason)
        questions = []
        for name, reason in reasons.items():
            spec = param_spec(formula_id, name)
            if spec is None or (unknown is not None and name == unknown.name):
                continue
            questions.append(question_for(spec, known, reason))
            known.pop(name, None)
        if not questions:
            logger.warning("calculation.bad_target formula=%s errors=%s", formula_id, reasons)
            return Unresolved(task_id, "bad_target")
        return NeedsInput(task_id, plan, known, questions)
    return Computed(task_id, result, plan, dict(known))


async def _search(
    question: str,
    queries: Sequence[str],
    deps: CalculationDeps,
    *,
    also: Sequence[RetrievedChunk] = (),
) -> list[RetrievedChunk]:
    """Best `LLM_CHUNKS_LIMIT` chunks over `queries` plus `also` (chunks already found),
    filtered like the advisory flow: the score threshold, then the LLM rerank judged
    against `question`, which drops a document of another programme or level that
    only shares keywords (a master's fee note for an undergraduate question)."""

    per_query = await retrieve_chunks(list(queries), deps.models.retrieval, deps.security)
    best: dict[str, RetrievedChunk] = {}
    for chunks in [list(also), *per_query]:
        for chunk in chunks:
            kept = best.get(chunk.chunk_id)
            if kept is None or chunk.score > kept.score:
                best[chunk.chunk_id] = chunk
    result = rerank_chunks([list(best.values())])
    models = deps.models
    # Same rerank stage as the main graph (CHAT_RERANK_MODE): cross-encoder first when
    # configured, the LLM rerank after it or instead of it, fail-over on a service error.
    evidence = levels_for([r.chunks for r in result.per_query])
    logger.info(
        "calculation.evidence mode=%s %s",
        settings.CHAT_RERANK_MODE,
        "; ".join(f"{lvl} {sig.describe()}" for sig, lvl in evidence),
    )
    llm_rerank_wanted = settings.CHAT_LLM_RERANK_ENABLED and (
        settings.CHAT_RERANK_MODE not in ("none", "auto")
        or (settings.CHAT_RERANK_MODE == "auto" and any(level == "weak" for _, level in evidence))
    )
    if (
        settings.CHAT_RERANK_MODE in ("cross_encoder", "cross_encoder_then_llm")
        and result.has_valid_context
    ):
        if deps.trace is not None:
            deps.trace.step("09a_CrossEncoderRerankNode", parent="07_CalculationNode")
        ce_outcome = await cross_encoder_rerank(CrossEncoderClient(), [question], result)
        if ce_outcome.failure is None:
            result = ce_outcome.result
            llm_rerank_wanted = (
                settings.CHAT_RERANK_MODE == "cross_encoder_then_llm"
                and settings.CHAT_LLM_RERANK_ENABLED
            )
        else:
            deps.warn("CROSS_ENCODER_FAILED", f"{CROSS_ENCODER_SKIPPED}{ce_outcome.failure}")
    if llm_rerank_wanted and result.has_valid_context:
        if models.rerank is not None:
            agent = build_llm_rerank_agent(models.rerank)
            if deps.trace is not None:
                deps.trace.step(
                    "09a_LLMRerankNode",
                    parent="07_CalculationNode",
                    agent=agent,
                    credential=models.rerank_credential,
                )
            outcome = await llm_rerank(
                agent,
                [question],
                result,
                credential=models.rerank_credential,
                purpose=models.rerank_purpose,
                snapshot_version=models.snapshot_version,
                on_attempt=deps.on_attempt,
                budget=deps.budget,
            )
            result = outcome.result
            if outcome.failure is not None:
                deps.warn("LLM_RERANK_FAILED", f"{LLM_RERANK_SKIPPED}{outcome.failure}")
        elif models.rerank_unavailable is not None:
            deps.warn("LLM_RERANK_UNAVAILABLE", f"{LLM_RERANK_SKIPPED}{models.rerank_unavailable}")
    chunks = result.chunks[:LLM_CHUNKS_LIMIT]
    logger.info(
        "calculation.retrieval queries=%r found=%d kept=[%s]",
        list(queries),
        len(best),
        "; ".join(f"{source_title(chunk.source)} ({chunk.score:.2f})" for chunk in chunks),
    )
    return chunks


def _builtin_params(formula_id: FormulaId, known: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Panel answers under the formula's param names; a single number answered for a
    list param (the practice columns) becomes a one-item list."""

    params: dict[str, JsonValue] = {}
    for name, value in known.items():
        spec = param_spec(formula_id, canonical_field(name))
        if spec is None:
            continue
        if spec.kind == "number_list" and not isinstance(value, list):
            value = [value]
        params[spec.name] = value
    return params


# ---------------------------------------------------------------------------
# Registered formulas: values looked up in the documents (unit prices, credits)
# ---------------------------------------------------------------------------

_AMOUNT = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+")
_OPTIONS_MAX = 12


@dataclass(frozen=True)
class DocumentValues:
    """Params read from the documents (each one checked against the chunk it cites),
    that chunk per param, and the context questions (which intake, which faculty
    group) to ask before the documents can be read for this student."""

    values: dict[str, JsonValue] = field(default_factory=dict)
    sources: dict[str, RetrievedChunk] = field(default_factory=dict)
    questions: list[dict[str, Any]] = field(default_factory=list)


def build_value_extractor_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_calculation_value_extractor,
        model_settings=auxiliary_model_settings(model),
    )


def _documents(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{index}] (nguồn: {chunk.source}"
        + (f"; mục: {' > '.join(chunk.heading_path)}" if chunk.heading_path else "")
        + f")\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _AMOUNT.fullmatch(value.strip()):
        return int(re.sub(r"\D", "", value))
    return None


def _in_range(spec: ParamSpec, number: int) -> bool:
    """A document value outside the param's own bounds is a misread (a "thang điểm 30" read
    as a 1200-point top score), never a value to compute with."""

    value = Decimal(number)
    if spec.min is not None and value < spec.min:
        return False
    return not (spec.max is not None and value > spec.max)


def _amounts(text: str) -> set[int]:
    """Every whole number written in `text`, thousands separators read as such."""

    return {int(re.sub(r"\D", "", match)) for match in _AMOUNT.findall(text)}


def _context_question(
    spec: ParamSpec, options: object, known: Mapping[str, JsonValue]
) -> dict[str, Any]:
    labels = list(
        dict.fromkeys(
            str(option).strip()[:120]
            for option in (options if isinstance(options, list) else [])
            if str(option).strip()
        )
    )[:_OPTIONS_MAX]
    if len(labels) < 2:
        return question_for(spec, known)
    return {
        "tab_label": spec.tab_label,
        "prompt": spec.label,
        "kind": "choice",
        "origin": "calculation",
        "field": spec.name,
        "options": [{"id": f"o{index}", "label": label} for index, label in enumerate(labels, 1)],
        "allow_other": True,
    }


def _value_prompt(
    query: str,
    wanted: Sequence[ParamSpec],
    known: Mapping[str, JsonValue],
    chunks: Sequence[RetrievedChunk],
    deps: CalculationDeps,
) -> str:
    context = {
        **dict(deps.confirmed_metadata),
        **{name: str(value) for name, value in known.items() if isinstance(value, str)},
    }
    known_lines = "; ".join(f"{name}: {value}" for name, value in context.items())
    wanted_lines = "; ".join(f"{spec.name}: {spec.label}" for spec in wanted)
    return append_current_time(
        f"<question>{query}</question>\n<known>{known_lines or '(trống)'}</known>\n"
        f"<wanted>{wanted_lines}</wanted>\n<documents>\n{_documents(chunks)}\n</documents>",
        deps.confirmed_metadata,
    )


async def fill_from_documents(
    formula_id: FormulaId,
    query: str,
    known: Mapping[str, JsonValue],
    deps: CalculationDeps,
    *,
    skip: Unknown | None = None,
) -> DocumentValues:
    """Missing params that can be looked up: search Qdrant, have the model copy each
    number with the chunk it is in, and keep a number only if that chunk really
    contains it. Anything not found is left for the panel. `skip`: the unknown of an
    inverse calculation, never looked up."""

    wanted = [
        spec
        for spec in missing_params(formula_id, known)
        if spec.lookup and (skip is None or spec.name != skip.name)
    ]
    if not wanted:
        return DocumentValues()
    context = " ".join(
        str(value) for name, value in known.items() if isinstance(value, str) and value.strip()
    )
    queries = [
        f"{lookup} {context} {query}".strip()
        for lookup in dict.fromkeys(spec.lookup for spec in wanted if spec.lookup)
    ]
    chunks = await _search(query, queries, deps)
    if not chunks:
        return DocumentValues()
    try:
        raw = await run_agent_text_with_failover(
            build_value_extractor_agent(deps.models.classification),
            _value_prompt(query, wanted, known, chunks, deps),
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=build_value_extractor_agent,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.value_extractor_failed", exc_info=True)
        return DocumentValues()

    data = load_json_object(raw) or {}
    wanted_names = {spec.name for spec in wanted}
    values: dict[str, JsonValue] = {}
    sources: dict[str, RetrievedChunk] = {}
    rejected: list[str] = []
    raw_values = data.get("values")
    for name, item in raw_values.items() if isinstance(raw_values, dict) else []:
        if name not in wanted_names or not isinstance(item, dict):
            continue
        number = _whole_number(item.get("value"))
        index = item.get("chunk")
        spec = param_spec(formula_id, name)
        if (
            number is None
            or spec is None
            or not isinstance(index, int)
            or not 1 <= index <= len(chunks)
            or number not in _amounts(chunks[index - 1].content)
            or not _in_range(spec, number)
        ):
            rejected.append(name)
            continue
        values[name] = number
        sources[name] = chunks[index - 1]

    questions: list[dict[str, Any]] = []
    if wanted_names - set(values):
        raw_ask = data.get("ask")
        for item in raw_ask if isinstance(raw_ask, list) else []:
            if not isinstance(item, dict):
                continue
            spec = param_spec(formula_id, str(item.get("field") or ""))
            if spec is None or spec.kind != "text" or known.get(spec.name) is not None:
                continue
            if all(question["field"] != spec.name for question in questions):
                questions.append(_context_question(spec, item.get("options"), known))
    logger.info(
        "calculation.document_values formula=%s found=%s rejected=%s ask=%s",
        formula_id,
        {name: f"{value} ({source_title(sources[name].source)})" for name, value in values.items()},
        rejected,
        [question["field"] for question in questions],
    )
    return DocumentValues(values, sources, questions)


_DOCUMENT_PRICE_NOTE = (
    "Đơn giá theo biểu phí đại học chính quy tại trụ sở chính; hệ hoặc cơ sở khác có đơn giá riêng."
)


def _value_text(spec: ParamSpec, value: JsonValue) -> str:
    if spec.unit == "đồng":
        return money(Decimal(str(value)))
    return f"{value} {spec.unit}" if spec.unit else str(value)


def _with_document_notes(
    outcome: Computed, formula_id: FormulaId, found: DocumentValues
) -> Computed:
    """The values read from the documents, each with its [n] marker (one per chunk)."""

    markers: dict[str, int] = {}
    sources: list[tuple[int, RetrievedChunk]] = []
    notes: list[str] = []
    prices = False
    for name, chunk in found.sources.items():
        spec = param_spec(formula_id, name)
        if spec is None:
            continue
        if chunk.chunk_id not in markers:
            markers[chunk.chunk_id] = len(markers) + 1
            sources.append((markers[chunk.chunk_id], chunk))
        label = label_for(spec, outcome.params)
        notes.append(
            f"{label}: {_value_text(spec, found.values[name])} [{markers[chunk.chunk_id]}]"
        )
        prices = prices or spec.unit == "đồng"
    if prices:
        notes.append(_DOCUMENT_PRICE_NOTE)
    return replace(outcome, notes=tuple(notes), sources=sources)


def profile_params(formula_id: FormulaId, profile: Mapping[str, str]) -> dict[str, JsonValue]:
    """The formula's profile-scoped parameters (the intake, the faculty group...) that
    the student already declared - never asked again."""

    return {
        spec.name: profile[spec.name]
        for spec in FORMULAS[formula_id].params
        if is_profile_field(spec.name) and profile.get(spec.name)
    }


async def builtin_task(
    task_id: str,
    formula_id: FormulaId,
    query: str,
    params: Mapping[str, JsonValue],
    deps: CalculationDeps,
    *,
    unknown: Unknown | None = None,
    target: Target | None = None,
) -> TaskOutcome:
    """A registered formula, documents first: values that can be looked up are read from
    Qdrant before anything is asked. When the documents need the intake or the faculty
    group first, that is asked together with the student's own missing values, and the
    lookup runs again on the answer. Whatever is still missing goes to the panel."""

    known = {key: value for key, value in params.items() if param_spec(formula_id, key)}
    if unknown is not None and unknown.index is None:
        known.pop(unknown.name, None)
    found = await fill_from_documents(formula_id, query, known, deps, skip=unknown)
    if found.questions:
        own = [
            spec
            for spec in missing_params(formula_id, known)
            if not spec.lookup and (unknown is None or spec.name != unknown.name)
        ]
        questions = [*found.questions, *(question_for(spec, known) for spec in own)]
        return NeedsInput(task_id, plan_for(formula_id, unknown, target), known, questions)
    outcome = builtin_outcome(
        task_id, formula_id, {**known, **found.values}, unknown=unknown, target=target
    )
    if isinstance(outcome, Computed) and found.sources:
        return _with_document_notes(outcome, formula_id, found)
    return outcome


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def plan_calculation(task_id: str, query: str, deps: CalculationDeps) -> Planned | Deferred:
    """The extractor's decision for one task: a registered formula with the numbers the
    student gave (a typed value wins over the profile, the profile over the previous
    calculation's numbers), plain arithmetic over the student's own numbers, or
    `Deferred` when nothing in the registry fits."""

    extraction = await extract_request(query, deps)
    previous = deps.previous
    if extraction.formula_id == "none":
        logger.info(
            "calculation.deferred task=%s retrieval_query=%r", task_id, extraction.retrieval_query
        )
        return Deferred(task_id, query, extraction.retrieval_query)
    if extraction.formula_id == "arithmetic":
        return _plan_arithmetic(task_id, query, extraction, deps)
    if extraction.formula_id == "previous":
        formula_id = previous_formula(previous)
        assert previous is not None and formula_id is not None
    else:
        formula_id = extraction.formula_id
    # The previous calculation's numbers are context for any registered formula: a
    # follow-up ("làm tròn thang 4") is not asked again for what the student already
    # gave, even when the router picked another formula.
    assert formula_id is not None
    reusable = previous is not None and previous_formula(previous) is not None
    params = {
        **(_builtin_params(formula_id, previous.params) if reusable and previous else {}),
        **profile_params(formula_id, deps.confirmed_metadata),
        **extraction.params,
    }
    return PlannedTask(task_id, query, formula_id, params, extraction.unknown, extraction.target)


def _plan_arithmetic(
    task_id: str, query: str, extraction: Extraction, deps: CalculationDeps
) -> ArithmeticTask | Deferred:
    """Permission and evidence for the student's own arithmetic: a whitelisted expression
    whose every operand is a number of the question (or the raw message). Anything else
    goes to the advisory path, which may still request the sum over cited figures."""

    request = extraction.arithmetic
    assert request is not None
    try:
        evaluation = evaluate(request.expression)
    except ArithmeticError as exc:
        logger.warning("calculation.arithmetic_refused task=%s %s", task_id, exc)
        return Deferred(task_id, query, None)
    missing = missing_operands(evaluation.operands, [query, deps.user_message or ""])
    if missing:
        logger.info(
            "calculation.arithmetic_deferred task=%s missing=%s",
            task_id,
            [plain(m) for m in missing],
        )
        return Deferred(task_id, query, None)
    logger.info(
        "calculation.arithmetic task=%s expression=%r value=%s",
        task_id,
        evaluation.expression,
        plain(evaluation.value),
    )
    return ArithmeticTask(task_id, query, request, evaluation)


async def run_planned_task(task: Planned, deps: CalculationDeps) -> TaskOutcome:
    if isinstance(task, ArithmeticTask):
        result = arithmetic_result(task.request, task.evaluation, source=ARITHMETIC_SOURCE_OWN)
        params: dict[str, JsonValue] = {
            "expression": task.evaluation.expression,
            "label": task.request.label,
            "unit": task.request.unit,
        }
        return Computed(task.task_id, result, CalculationPlan(formula_id="arithmetic"), params)
    return await builtin_task(
        task.task_id,
        task.formula_id,
        task.query,
        task.params,
        deps,
        unknown=task.unknown,
        target=task.target,
    )


async def run_calculation_task(
    task_id: str, query: str, deps: CalculationDeps
) -> TaskOutcome | Deferred:
    """Plan and run one task (the graph plans every task first, so a deferred one can
    join the advisory part before retrieval starts)."""

    planned = await plan_calculation(task_id, query, deps)
    if isinstance(planned, Deferred):
        return planned
    return await run_planned_task(planned, deps)


async def resume_calculation_task(
    task_id: str,
    query: str,
    plan: CalculationPlan,
    known_params: Mapping[str, JsonValue],
    answers: Mapping[str, JsonValue],
    deps: CalculationDeps,
) -> TaskOutcome:
    """A panel answer: never calls the extractor again. The formula computes (or solves
    for the plan's unknown), looking its values up again when an answer told which fee
    schedule row applies. A round stored by the retired LLM path (`formula_id: llm`)
    cannot be finished any more and says so."""

    if plan.formula_id == "llm":
        return Unresolved(task_id, "legacy_llm")
    known = registry_id(plan.formula_id)
    if known is None:
        return Unresolved(task_id, "not_resumable")
    unknown, target = solving_of(plan)
    values = {**profile_params(known, deps.confirmed_metadata), **known_params, **answers}
    return await builtin_task(task_id, known, query, values, deps, unknown=unknown, target=target)


# Fixed sentences for a task that cannot be calculated (rendered by Python, no LLM).
UNRESOLVED_MESSAGES: dict[UnresolvedReason, str] = {
    "legacy_llm": (
        "Phần tính toán này thuộc phiên bản cũ nên mình không hoàn tất được nữa. Bạn hỏi lại "
        "phép tính trong một câu mới, kèm các số liệu, để mình tính giúp nhé."
    ),
    "bad_target": (
        "Mình chưa xác định được đại lượng cần tìm trong câu hỏi. Bạn nêu rõ cần tìm giá trị "
        'nào và mốc cần đạt (VD "cuối kỳ cần bao nhiêu để tổng kết đạt 8,5") giúp mình nhé.'
    ),
    "not_resumable": (
        "Phép tính này không cần thêm số liệu; bạn hỏi lại trong một câu mới giúp mình nhé."
    ),
}
