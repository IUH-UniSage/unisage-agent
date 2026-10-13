"""CalculationNode - one `academic_calculation` task to a result, the questions
still needed, or an honest "cannot calculate this".

Two ways to compute (docs/specs/SPEC-calculation-node.md):
- Python, for the FORWARD calculation of the 3 built-in formulas (`formulas.py`):
  the LLM only picks the formula and copies the numbers the student gave; missing
  or invalid ones become panel questions built from `ParamSpec`.
- The LLM (`main/chat_calculation_llm.yaml`), for everything else: target
  questions ("cuối kỳ cần bao nhiêu để được A+"), formulas found in the documents
  (regulations, tuition, lessons...) and follow-ups that need the chat history. Its
  answer is labelled "AI tự tính, có thể sai" and gets Đúng/Sai feedback; when it
  needs numbers it returns an ask_user_form block, asked on the panel.
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

from app.calculation.formulas import (
    FORMULAS,
    FormulaId,
    ParamSpec,
    calculate,
    describe_builtin_formulas,
    describe_builtin_rules,
    label_for,
    missing_params,
    money,
    param_spec,
    plain,
    route_builtin,
)
from app.calculation.result import CalculationInputError, CalculationResult
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.graph.clarification_round import slug
from app.graph.fence_redactor import FenceRedactor, take_bare_blocks
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
    generation_model_settings,
    run_agent_text_with_failover,
)
from app.graph.streaming_state import AdminWarning, GraphModels
from app.rag.prompting import build_calculation_llm_prompt, get_templates
from app.rag.prompting.citations import cited_indexes, source_title
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import CalculationPlan, LastCalculation
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

BUILTIN_IDS: frozenset[str] = frozenset(FORMULAS)
UnresolvedReason = Literal["llm_disabled", "llm_failed"]
# Documents handed to the LLM for one calculation.
LLM_CHUNKS_LIMIT = 5


@dataclass(frozen=True)
class Computed:
    """A built-in formula computed by Python."""

    task_id: str
    result: CalculationResult
    plan: CalculationPlan
    params: dict[str, JsonValue]
    # Lines naming each value read from the documents, with its [n] marker, and the
    # chunk behind each marker; renumbered for the whole message before it is shown.
    notes: tuple[str, ...] = ()
    sources: list[tuple[int, RetrievedChunk]] = field(default_factory=list)


@dataclass(frozen=True)
class LlmAnswered:
    """A calculation the LLM did itself - shown with the "AI tự tính" notice and the
    documents it cited."""

    task_id: str
    query: str
    text: str
    plan: CalculationPlan
    known_params: dict[str, JsonValue]
    # (the [n] marker in `text`, chunk) for every document it cited, in first-cited
    # order; renumbered for the whole message before it is shown.
    sources: list[tuple[int, RetrievedChunk]] = field(default_factory=list)


@dataclass(frozen=True)
class NeedsInput:
    task_id: str
    plan: CalculationPlan
    known_params: dict[str, JsonValue]
    questions: list[dict[str, Any]]
    # Shown above the panel instead of the generic lead (the LLM's own sentence).
    lead: str | None = None


@dataclass(frozen=True)
class Unresolved:
    task_id: str
    reason: UnresolvedReason


TaskOutcome = Computed | LlmAnswered | NeedsInput | Unresolved


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
    # Recent chat, so the LLM can pick up the numbers of an earlier calculation.
    history: Sequence[HistoryMessage] = ()
    # Self-declared attributes, shown apart from the verified identity.
    confirmed_metadata: Mapping[str, str] = field(default_factory=dict)
    trace: GraphTrace | None = None
    # AI-admin warnings raised while calculating, one per code.
    warnings: list[AdminWarning] = field(default_factory=list)

    def warn(self, code: str, message: str) -> None:
        if all(warning.code != code for warning in self.warnings):
            self.warnings.append(AdminWarning(code=code, message=message))


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
    formula_id: FormulaId | Literal["llm", "previous"]
    params: dict[str, JsonValue]
    retrieval_query: str | None


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


def _llm_extraction(params: dict[str, JsonValue], data: Mapping[str, Any]) -> Extraction:
    retrieval_query = data.get("retrieval_query")
    query = retrieval_query.strip()[:500] if isinstance(retrieval_query, str) else ""
    return Extraction("llm", params, query or None)


def parse_extraction(
    raw: str, *, allowed: list[str], previous: LastCalculation | None = None
) -> Extraction:
    """Anything that is not a forward built-in calculation goes to the LLM, which can
    still ask for what it needs. An explicit "llm" always wins; otherwise a single
    built-in formula matched by the rule-based router wins (even over unreadable
    output), and unreadable output without one goes to the LLM."""

    data = load_json_object(raw) or {}
    formula_id = data.get("formula_id")
    raw_params = data.get("params")
    params: dict[str, JsonValue] = dict(raw_params) if isinstance(raw_params, dict) else {}
    if formula_id == "previous" and previous is not None and previous.plan.formula_id != "llm":
        return Extraction("previous", params, None)
    routed = [formula for formula in allowed if formula in BUILTIN_IDS]
    if formula_id == "llm" or (formula_id not in BUILTIN_IDS and len(routed) != 1):
        return _llm_extraction(params, data)
    if len(routed) == 1 and formula_id != routed[0]:
        # The rule-based router already decided which built-in formula this is.
        logger.info("calculation.router_disagreement routed=%s llm=%s", routed[0], formula_id)
        formula_id = routed[0]
    elif routed and formula_id not in routed:
        return _llm_extraction(params, data)
    builtin: FormulaId = next(known for known in FORMULAS if known == formula_id)
    return Extraction(builtin, params, None)


def _previous_block(previous: LastCalculation) -> str:
    assert previous.plan.formula_id != "llm"
    described = {
        "formula_id": "previous",
        "formula": FORMULAS[previous.plan.formula_id].title,
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
    if previous is not None and previous.plan.formula_id != "llm":
        prompt += _previous_block(previous)
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
        raw = ""
    return parse_extraction(raw, allowed=allowed, previous=previous)


# ---------------------------------------------------------------------------
# Built-in formulas: forward calculation by Python
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
                    questions.append(question_for(spec, known, reason))
                    known.pop(name, None)
            return NeedsInput(task_id, plan, known, questions)
    return NeedsInput(task_id, plan, known, [question_for(spec, known) for spec in missing])


# ---------------------------------------------------------------------------
# Everything else: the LLM computes (target questions, document formulas...)
# ---------------------------------------------------------------------------


def build_calculation_llm_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, model_settings=generation_model_settings(model))


def _documents(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{index}] (nguồn: {chunk.source}"
        + (f"; mục: {' > '.join(chunk.heading_path)}" if chunk.heading_path else "")
        + f")\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )


def _number(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        float(text)
    except ValueError:
        return None
    return text


def llm_questions(
    forms: Sequence[Mapping[str, Any]], known: Mapping[str, JsonValue]
) -> list[dict[str, Any]]:
    """Panel questions from the LLM's ask_user_form blocks. A field already answered,
    asked twice, or malformed is dropped - that is what stops asking in circles."""

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for form in forms:
        fields = form.get("fields")
        if form.get("type") != "ask_user_form" or not isinstance(fields, list):
            continue
        for item in fields:
            if not isinstance(item, Mapping):
                continue
            name = slug(str(item.get("field") or ""))
            label = str(item.get("label") or "").strip()
            kind = item.get("kind")
            if not name or not label or name in seen or name in known:
                continue
            draft: dict[str, Any] = {
                "tab_label": label[:24],
                "prompt": label[:200],
                "origin": "calculation",
                "field": name,
            }
            if kind in ("number", "number_list"):
                low, high = _number(item.get("min")), _number(item.get("max"))
                draft["kind"] = kind
                draft["number"] = {
                    "min": low if low is not None else "-1000000000000",
                    "max": high if high is not None else "1000000000000",
                    "step": "0.01",
                    "unit": None,
                }
                if kind == "number_list":
                    draft["max_items"] = 20
            elif kind == "choice":
                options = [
                    {"id": f"o{index}", "label": str(option.get("label") or "").strip()[:120]}
                    for index, option in enumerate(item.get("options") or [], start=1)
                    if isinstance(option, Mapping) and str(option.get("label") or "").strip()
                ][:12]
                if len(options) < 2:
                    continue
                draft.update(kind="choice", options=options, allow_other=True)
            elif kind == "text":
                draft.update(kind="text", max_length=200)
            else:
                continue
            seen.add(name)
            questions.append(draft)
    return questions


def _cited(text: str, chunks: Sequence[RetrievedChunk]) -> list[tuple[int, RetrievedChunk]]:
    return [(index, chunks[index - 1]) for index in cited_indexes(text, len(chunks))]


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
    if settings.CHAT_LLM_RERANK_ENABLED and result.has_valid_context:
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


async def _ask_llm(
    task_id: str,
    query: str,
    plan: CalculationPlan,
    known: Mapping[str, JsonValue],
    chunks: list[RetrievedChunk],
    deps: CalculationDeps,
) -> LlmAnswered | NeedsInput | Unresolved:
    prompt = build_calculation_llm_prompt(
        user_query=query,
        builtin_rules=describe_builtin_rules(),
        documents=_documents(chunks),
        known_values=json.dumps(dict(known), ensure_ascii=False) if known else "(chưa có)",
        history=list(deps.history),
        security=deps.security,
        confirmed_metadata=dict(deps.confirmed_metadata),
    )
    try:
        raw = await run_agent_text_with_failover(
            build_calculation_llm_agent(deps.models.generation),
            prompt,
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=build_calculation_llm_agent,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_CALC_LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.llm_failed", exc_info=True)
        return Unresolved(task_id, "llm_failed")
    redactor = FenceRedactor()
    text, bare = take_bare_blocks((redactor.feed(raw) + redactor.finish()).strip())
    questions = llm_questions([*redactor.captured, *bare], known)
    if questions:
        return NeedsInput(task_id, plan, dict(known), questions, lead=text or None)
    if not text:
        return Unresolved(task_id, "llm_failed")
    return LlmAnswered(task_id, query, text, plan, dict(known), _cited(text, chunks))


async def llm_outcome(
    task_id: str,
    query: str,
    plan: CalculationPlan,
    known: Mapping[str, JsonValue],
    deps: CalculationDeps,
) -> LlmAnswered | NeedsInput | Unresolved:
    """Non-streamed LLM call(s) over the built-in rules, the retrieved documents, the known
    values and the recent chat. Shared by the first turn and every resume.

    Documents before the user's form: when the LLM would ask for missing numbers and Qdrant
    has not been searched with the question itself yet (a short chat-style question often
    gets no `retrieval_query`, so the LLM saw no document), search with it and ask again. Only
    if numbers are still missing does the panel go to the user."""

    if not settings.CHAT_CALC_LLM_ENABLED:
        return Unresolved(task_id, "llm_disabled")
    chunks = await _search(query, [plan.retrieval_query], deps) if plan.retrieval_query else []
    outcome = await _ask_llm(task_id, query, plan, known, chunks, deps)
    if not isinstance(outcome, NeedsInput) or plan.retrieval_query == query:
        return outcome
    more = await _search(query, [query], deps, also=chunks)
    if {chunk.chunk_id for chunk in more} <= {chunk.chunk_id for chunk in chunks}:
        return outcome  # nothing new to show the LLM
    return await _ask_llm(task_id, query, plan, known, more, deps)


# ---------------------------------------------------------------------------
# Built-in formulas: values looked up in the documents (unit prices, credits)
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
    return (
        f"<question>{query}</question>\n<known>{known_lines or '(trống)'}</known>\n"
        f"<wanted>{wanted_lines}</wanted>\n<documents>\n{_documents(chunks)}\n</documents>"
    )


async def fill_from_documents(
    formula_id: FormulaId,
    query: str,
    known: Mapping[str, JsonValue],
    deps: CalculationDeps,
) -> DocumentValues:
    """Missing params that can be looked up: search Qdrant, have the model copy each
    number with the chunk it is in, and keep a number only if that chunk really
    contains it. Anything not found is left for the panel."""

    wanted = [spec for spec in missing_params(formula_id, known) if spec.lookup]
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
        if (
            number is None
            or not isinstance(index, int)
            or not 1 <= index <= len(chunks)
            or number not in _amounts(chunks[index - 1].content)
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


async def builtin_task(
    task_id: str,
    formula_id: FormulaId,
    query: str,
    params: Mapping[str, JsonValue],
    deps: CalculationDeps,
) -> Computed | NeedsInput:
    """A built-in formula, documents first: values that can be looked up are read from
    Qdrant before anything is asked. When the documents need the intake or the faculty
    group first, that is asked together with the student's own missing values, and the
    lookup runs again on the answer. Whatever is still missing goes to the panel."""

    known = {key: value for key, value in params.items() if param_spec(formula_id, key)}
    found = await fill_from_documents(formula_id, query, known, deps)
    if found.questions:
        own = [spec for spec in missing_params(formula_id, known) if not spec.lookup]
        questions = [*found.questions, *(question_for(spec, known) for spec in own)]
        return NeedsInput(task_id, CalculationPlan(formula_id=formula_id), known, questions)
    outcome = builtin_outcome(task_id, formula_id, {**known, **found.values})
    if isinstance(outcome, Computed) and found.sources:
        return _with_document_notes(outcome, formula_id, found)
    return outcome


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_calculation_task(task_id: str, query: str, deps: CalculationDeps) -> TaskOutcome:
    extraction = await extract_request(query, deps)
    if extraction.formula_id == "previous":
        previous = deps.previous
        assert previous is not None and previous.plan.formula_id != "llm"
        params = {**previous.params, **extraction.params}
        return await builtin_task(task_id, previous.plan.formula_id, query, params, deps)
    if extraction.formula_id == "llm":
        plan = CalculationPlan(formula_id="llm", retrieval_query=extraction.retrieval_query)
        return await llm_outcome(task_id, query, plan, extraction.params, deps)
    return await builtin_task(task_id, extraction.formula_id, query, extraction.params, deps)


async def resume_calculation_task(
    task_id: str,
    query: str,
    plan: CalculationPlan,
    known_params: Mapping[str, JsonValue],
    answers: Mapping[str, JsonValue],
    deps: CalculationDeps,
) -> TaskOutcome:
    """A panel answer: never calls the extractor again. A built-in formula computes
    (looking its values up again when an answer told which fee schedule row applies);
    an LLM calculation is asked again with the answers as known values."""

    values = {**known_params, **answers}
    if plan.formula_id == "llm":
        return await llm_outcome(task_id, query, plan, values, deps)
    return await builtin_task(task_id, plan.formula_id, query, values, deps)


# Fixed sentences for a task that cannot be calculated (rendered by Python, no LLM).
UNRESOLVED_MESSAGES: dict[UnresolvedReason, str] = {
    "llm_disabled": (
        "Hiện mình chỉ tự tính được GPA, điểm tổng kết học phần, quy đổi điểm và học phí. "
        "Bạn có thể liên hệ Phòng Đào tạo hoặc giảng viên phụ trách để được hướng dẫn nhé."
    ),
    "llm_failed": "Mình chưa tính được câu này lúc này, bạn thử hỏi lại sau ít phút giúp mình nhé.",
}
