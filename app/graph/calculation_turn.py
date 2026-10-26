"""The calculation part of a turn, between the node and the graph: what is shown
(a built-in result rendered by Python, or the LLM's own calculation under the "AI
tự tính" notice), the review of a built-in calculation-only turn (does the result
answer what was asked, and the opening answer - checked against the numbers Python
showed; a result that does not fit is redone by the LLM), and the trace.

Spec: docs/specs/SPEC-calculation-node.md §3 and §7.
"""

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.calculation.render import render_markdown
from app.calculation.result import CalculationResult
from app.core.config import settings
from app.graph.clarification_round import TaskQuestions
from app.graph.metadata_fields import canonical_field
from app.graph.nodes.calculation import (
    UNRESOLVED_MESSAGES,
    CalculationDeps,
    Computed,
    NeedsInput,
    TaskOutcome,
    Unresolved,
    load_json_object,
)
from app.graph.streaming import generation_model_settings, run_agent_text_with_failover
from app.rag.prompting import build_calculation_review_prompt, get_templates
from app.rag.prompting.citations import chunk_citation, renumber_markers
from app.schemas.clarification import PendingCalculationTask
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

NEEDS_INPUT_LEAD = "Mình cần thêm vài thông tin để tính giúp bạn:"
# Dotted thousands first, so "165.200.000" is one number and not 165.2 and 0.
_NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})+(?!\d)|\d+(?:[.,]\d+)?")
_HEADING_SEPARATOR = " \u203a "  # single right angle quote (RUF001 forbids the literal)
# Scale names the answer may mention without them being "new numbers".
_ALWAYS_ALLOWED = frozenset({Decimal(0), Decimal(4), Decimal(10)})
_EXTRACTOR_PROMPTS = ("agent_calculation_extractor",)
_REVIEW_PROMPT = "chat_calculation_review"


@dataclass(frozen=True)
class CalculationTask:
    task_id: str
    query: str


# ---------------------------------------------------------------------------
# What the student sees (deterministic)
# ---------------------------------------------------------------------------


def number_citations(
    outcomes: Sequence[TaskOutcome], first_index: int
) -> tuple[list[TaskOutcome], list[dict[str, Any]]]:
    """Give every document a calculation read a value from one `[n]` for the whole
    message, from `first_index` on (the advisory answer of a mixed turn keeps
    1..first_index-1), and the `citations` entries the client turns those markers into
    links with."""

    numbered: list[TaskOutcome] = []
    citations: list[dict[str, Any]] = []
    next_index = first_index
    for outcome in outcomes:
        if not isinstance(outcome, Computed) or not outcome.sources:
            numbered.append(outcome)
            continue
        mapping: dict[int, int] = {}
        sources: list[tuple[int, RetrievedChunk]] = []
        for local, chunk in outcome.sources:
            mapping[local] = next_index
            sources.append((next_index, chunk))
            citations.append(chunk_citation(next_index, chunk))
            next_index += 1
        notes = tuple(renumber_markers(note, mapping) for note in outcome.notes)
        numbered.append(replace(outcome, notes=notes, sources=sources))
    return numbered, citations


def render_outcome(outcome: TaskOutcome) -> str:
    """Markdown for one task; a task that needs input is asked on the panel (only its
    own lead, if any, is shown here)."""

    if isinstance(outcome, Computed):
        rendered = render_markdown(outcome.result)
        if not outcome.notes:
            return rendered
        lines = "\n".join(f"- {note}" for note in outcome.notes)
        return f"{rendered}\n\nSố liệu lấy từ tài liệu:\n{lines}"
    if isinstance(outcome, Unresolved):
        return UNRESOLVED_MESSAGES[outcome.reason]
    return outcome.lead or ""


def render_outcomes(outcomes: Sequence[TaskOutcome]) -> str:
    blocks = [text for text in (render_outcome(outcome) for outcome in outcomes) if text]
    if not blocks and any(isinstance(outcome, NeedsInput) for outcome in outcomes):
        return NEEDS_INPUT_LEAD
    return "\n\n---\n\n".join(blocks)


def known_fields(outcomes: Sequence[TaskOutcome]) -> set[str]:
    """Registry names of every value this turn's calculations already have - node 10
    must not ask for them again."""

    names: set[str] = set()
    for outcome in outcomes:
        if isinstance(outcome, Computed):
            values: Mapping[str, JsonValue] = outcome.params
        elif isinstance(outcome, NeedsInput):
            values = outcome.known_params
        else:
            continue
        names.update(canonical_field(name) for name, value in values.items() if value is not None)
    return names


def calculation_titles(outcomes: Sequence[TaskOutcome]) -> list[str]:
    """What node 10 is told was already shown - titles and the names of the values
    used, never a number."""

    titles: list[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, Computed):
            continue
        title = outcome.result.title
        names = sorted(known_fields([outcome]))
        titles.append(f"{title} [đã có: {', '.join(names)}]" if names else title)
    return titles


def needs_input_parts(
    outcomes: Sequence[TaskOutcome], queries: Mapping[str, str]
) -> list[TaskQuestions]:
    return [
        TaskQuestions(
            task=PendingCalculationTask(
                task_id=outcome.task_id,
                query=queries[outcome.task_id],
                plan=outcome.plan,
                known_params=dict(outcome.known_params),
            ),
            questions=outcome.questions,
        )
        for outcome in outcomes
        if isinstance(outcome, NeedsInput)
    ]


# ---------------------------------------------------------------------------
# Review of a calculation-only turn (main/chat_calculation_review.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Review:
    """Whether the built-in results answer what was asked, and the sentences shown
    above them. `rejected`: the model wrote a number Python never showed, so nothing
    it wrote is shown (`fits`/`issue` are kept for the staff trace only)."""

    fits: bool
    issue: str | None
    answer: str
    rejected: bool = False

    def lead(self) -> str:
        """Markdown shown before the Python blocks; empty when nothing may be shown."""

        if self.rejected:
            return ""
        parts = [] if self.fits or not self.issue else [f"**Lưu ý:** {self.issue}"]
        if self.answer:
            parts.append(self.answer)
        return "\n\n".join(parts)


def build_review_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, model_settings=generation_model_settings(model))


def _payload(results: Sequence[CalculationResult]) -> str:
    return json.dumps(
        [
            {
                "title": result.title,
                "inputs": dict(result.inputs),
                "steps": [f"{step.label}: {step.substituted}" for step in result.steps],
                "outputs": dict(result.outputs),
            }
            for result in results
        ],
        ensure_ascii=False,
    )


def _as_number(text: str) -> Decimal | None:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    try:
        return Decimal(text.replace(",", ".")).normalize()
    except InvalidOperation:
        return None


def _numbers(text: str) -> set[Decimal]:
    return {
        number
        for match in _NUMBER.finditer(text)
        if (number := _as_number(match.group(0))) is not None
    }


def allowed_numbers(outcomes: Sequence[Computed], question: str) -> set[Decimal]:
    """Numbers the answer may use: everything Python showed for these results (title,
    formula, steps, result, document notes) and the student's own question - "2 năm
    đầu" must be sayable when the result is for one year."""

    allowed = set(_ALWAYS_ALLOWED) | _numbers(question)
    for outcome in outcomes:
        allowed |= _numbers(render_outcome(outcome))
    return allowed


def unknown_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    return [
        match.group(0)
        for match in _NUMBER.finditer(text)
        if (number := _as_number(match.group(0))) is None or number not in allowed
    ]


def parse_review(raw: str) -> Review | None:
    data = load_json_object(raw)
    if data is None or not isinstance(data.get("fits"), bool):
        return None
    issue = data.get("issue")
    answer = data.get("answer")
    issue = issue.strip() if isinstance(issue, str) and issue.strip() else None
    answer = answer.strip() if isinstance(answer, str) else ""
    if not data["fits"] and issue is None:
        return None  # a mismatch with nothing to say about it cannot be shown
    return Review(fits=data["fits"], issue=None if data["fits"] else issue, answer=answer)


async def review(
    outcomes: Sequence[Computed],
    question: str,
    deps: CalculationDeps,
) -> Review | None:
    """Not streamed: the whole review is checked before anything is sent. A failed
    call, unreadable JSON (None) or a number Python never showed (`rejected`) leaves
    the Python blocks on their own - they never depend on this call."""

    if not outcomes:
        return None
    prompt = build_calculation_review_prompt(
        user_query=question,
        calculation_payload=_payload([outcome.result for outcome in outcomes]),
        history=list(deps.history),
        confirmed_metadata=deps.confirmed_metadata,
    )
    try:
        raw = await run_agent_text_with_failover(
            build_review_agent(deps.models.generation),
            prompt,
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=build_review_agent,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.review_failed", exc_info=True)
        return None
    result = parse_review(raw)
    if result is None:
        logger.warning("calculation.review_unreadable")
        return None
    unknown = unknown_numbers(
        f"{result.issue or ''} {result.answer}", allowed_numbers(outcomes, question)
    )
    if unknown:
        logger.warning("calculation.review_rejected numbers=%s", unknown)
        return replace(result, rejected=True)
    if not result.fits:
        logger.info("calculation.review_mismatch")
    return result


# ---------------------------------------------------------------------------
# Trace: public summary on the message, full trace for staff (SPEC §7.2)
# ---------------------------------------------------------------------------


def _hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _status(outcome: TaskOutcome) -> str:
    if isinstance(outcome, Computed):
        return "computed"
    if isinstance(outcome, NeedsInput):
        return "needs_input"
    return "unresolved"


def _mode(outcome: TaskOutcome) -> str:
    """Every calculation is Python's since 26/10/2026; `llm` only names a round stored
    by the retired path (the client's enum knows both)."""

    del outcome
    return "builtin"


def _chunk_trace(index: int, chunk: RetrievedChunk) -> dict[str, JsonValue]:
    return {
        "ref": f"[{index}]",
        "chunk_id": chunk.chunk_id,
        "document_id": str(chunk.metadata.get("document_id") or "") or None,
        "source": chunk.source,
        "heading_path": list(chunk.heading_path),
        "chunk_hash": _hash(chunk.content),
    }


def trace_items(
    outcomes: Sequence[TaskOutcome],
    *,
    queries: Mapping[str, str],
    run_id: str,
    deps: CalculationDeps,
    review: Review | None = None,
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    """(public items for `metadata.calculation`, private traces for Java's staff table).
    `review` covers every computed result of a calculation-only turn, so each item
    carries it; the trace names the formula, its version and the inverse target."""

    versions = get_templates().versions
    credential = deps.models.generation_credential
    model_name = f"{credential.provider}/{credential.model_name}" if credential else "unknown"
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    for outcome in outcomes:
        mode = _mode(outcome)
        sources = outcome.sources if isinstance(outcome, Computed) else []
        item: dict[str, JsonValue] = {
            "item_id": outcome.task_id,
            "run_id": run_id,
            "mode": mode,
            "status": _status(outcome),
            "result_summary": (
                "; ".join(f"{label}: {value}" for label, value in outcome.result.outputs)
                if isinstance(outcome, Computed)
                else None
            ),
            "source_summary": (
                {
                    "title": sources[0][1].source,
                    "heading": _HEADING_SEPARATOR.join(sources[0][1].heading_path),
                }
                if sources
                else None
            ),
        }
        public.append(item)

        reviewed = review is not None and isinstance(outcome, Computed)
        prompts: tuple[str, ...] = _EXTRACTOR_PROMPTS
        if reviewed:
            prompts = (*prompts, _REVIEW_PROMPT)
        trace: dict[str, JsonValue] = {
            "question_raw": queries.get(outcome.task_id, ""),
            "status": item["status"],
            "mode": mode,
            "models": {"llm": model_name},
            "prompt_versions": {name: f"sha256:{versions.get(name, '')}" for name in prompts},
        }
        if isinstance(outcome, Computed):
            trace["formula_id"] = outcome.plan.formula_id
            trace["formula_version"] = outcome.result.version
            trace["formula_hash"] = _hash(f"{outcome.plan.formula_id}@{outcome.result.version}")
            trace["solve_for"] = outcome.plan.solve_for
            trace["target"] = (
                outcome.plan.target.model_dump() if outcome.plan.target is not None else None
            )
            trace["inputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.inputs
            ]
            trace["outputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.outputs
            ]
            if sources:
                trace["sources"] = [_chunk_trace(index, chunk) for index, chunk in sources]
        if review is not None and reviewed:
            trace["review"] = {
                "fits": review.fits,
                "issue": review.issue,
                "rejected": review.rejected,
            }
        if isinstance(outcome, NeedsInput):
            trace["formula_id"] = outcome.plan.formula_id
            trace["known_params"] = dict(outcome.known_params)
        if isinstance(outcome, Unresolved):
            trace["reason"] = outcome.reason
        private.append({"itemId": outcome.task_id, "runId": run_id, "trace": trace})
    return public, private
