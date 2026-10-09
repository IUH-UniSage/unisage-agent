"""The calculation part of a turn, between the node and the graph: what is shown
(rendered by Python), the short note after a calculation-only turn (checked
against the numbers actually computed), and the trace.

Spec: docs/specs/SPEC-calculation-node.md §3 and §7.
"""

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.calculation.render import render_markdown
from app.calculation.result import CalculationResult
from app.core.config import settings
from app.graph.clarification_round import TaskQuestions
from app.graph.nodes.calculation import (
    UNRESOLVED_MESSAGES,
    CalculationDeps,
    Computed,
    NeedsInput,
    QuoteOnly,
    TaskOutcome,
    Unresolved,
)
from app.graph.streaming import generation_model_settings, run_agent_text_with_failover
from app.rag.prompting import build_calculation_commentary_prompt, get_templates
from app.schemas.clarification import FormulaSource, PendingCalculationTask

logger = logging.getLogger(__name__)

RETRIEVED_NOTICE = "Kết quả tham khảo theo quy chế"
NEEDS_INPUT_LEAD = "Mình cần thêm vài thông tin để tính giúp bạn:"
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_HEADING_SEPARATOR = " \u203a "  # single right angle quote (RUF001 forbids the literal)
# Scale names the note may mention without them being "new numbers".
_ALWAYS_ALLOWED = frozenset({Decimal(0), Decimal(4), Decimal(10)})
_EXTRACTOR_PROMPTS = ("agent_calculation_extractor",)
_RETRIEVED_PROMPTS = (
    "agent_calculation_extractor",
    "agent_calculation_formula",
    "agent_calculation_formula_verifier",
)


@dataclass(frozen=True)
class CalculationTask:
    task_id: str
    query: str


# ---------------------------------------------------------------------------
# What the student sees (deterministic)
# ---------------------------------------------------------------------------


def _source_lines(source: FormulaSource) -> list[str]:
    heading = _HEADING_SEPARATOR.join(source.heading_path)
    where = f"{source.source}" + (f" - {heading}" if heading else "")
    return [f"Nguồn: {where}", f'> "{source.source_quote}"']


def render_outcome(outcome: TaskOutcome) -> str:
    """Markdown for one task, or "" for a task that only needs input (asked on the panel)."""

    if isinstance(outcome, Computed):
        if outcome.source is None:
            return render_markdown(outcome.result)
        body = render_markdown(outcome.result, notice=RETRIEVED_NOTICE)
        return "\n".join([body, "", *_source_lines(outcome.source)])
    if isinstance(outcome, QuoteOnly):
        return "\n".join(
            [
                "**Công thức theo quy chế**",
                "",
                *_source_lines(outcome.source),
                "",
                "Bạn thế số vào công thức trên để tính nhé, hiện mình chưa tự tính loại công "
                "thức này.",
            ]
        )
    if isinstance(outcome, Unresolved):
        message = UNRESOLVED_MESSAGES[outcome.reason]
        if outcome.reason == "formula_ambiguous" and outcome.candidates:
            lines = [
                f"- {item['summary']} (nguồn: {item['source']})" for item in outcome.candidates
            ]
            return "\n".join(
                [message, *lines, "", "Bạn cho mình biết bạn thuộc trường hợp nào nhé?"]
            )
        return message
    return ""


def render_outcomes(outcomes: Sequence[TaskOutcome]) -> str:
    blocks = [text for text in (render_outcome(outcome) for outcome in outcomes) if text]
    if not blocks and any(isinstance(outcome, NeedsInput) for outcome in outcomes):
        return NEEDS_INPUT_LEAD
    return "\n\n---\n\n".join(blocks)


def calculation_titles(outcomes: Sequence[TaskOutcome]) -> list[str]:
    """What node 10 is told was already shown - titles only, never a number."""

    return [outcome.result.title for outcome in outcomes if isinstance(outcome, Computed)]


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
# The note after a calculation-only turn (main/chat_calculation.yaml)
# ---------------------------------------------------------------------------


def build_commentary_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, model_settings=generation_model_settings(model))


def _payload(results: Sequence[CalculationResult]) -> str:
    return json.dumps(
        [
            {
                "title": result.title,
                "inputs": dict(result.inputs),
                "outputs": dict(result.outputs),
            }
            for result in results
        ],
        ensure_ascii=False,
    )


def _as_number(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ".")).normalize()
    except InvalidOperation:
        return None


def allowed_numbers(results: Sequence[CalculationResult]) -> set[Decimal]:
    allowed = set(_ALWAYS_ALLOWED)
    for result in results:
        texts = [value for _, value in result.inputs] + [value for _, value in result.outputs]
        texts += [step.display for step in result.steps]
        for text in texts:
            for match in _NUMBER.finditer(text):
                number = _as_number(match.group(0))
                if number is not None:
                    allowed.add(number)
    return allowed


def unknown_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    return [
        match.group(0)
        for match in _NUMBER.finditer(text)
        if (number := _as_number(match.group(0))) is None or number not in allowed
    ]


def fixed_note(results: Sequence[CalculationResult]) -> str:
    """Used when the LLM note fails the check (or the call fails): built from outputs only."""

    parts = [", ".join(f"{label} {value}" for label, value in result.outputs) for result in results]
    return "Kết quả: " + "; ".join(parts) + "."


async def commentary(
    results: Sequence[CalculationResult], user_query: str, deps: CalculationDeps
) -> str:
    """Not streamed: the whole note is checked before anything is sent."""

    if not results:
        return ""
    prompt = build_calculation_commentary_prompt(
        user_query=user_query, calculation_payload=_payload(results)
    )
    try:
        text = await run_agent_text_with_failover(
            build_commentary_agent(deps.models.generation),
            prompt,
            purpose="CHAT",
            credential=deps.models.generation_credential,
            snapshot_version=deps.models.snapshot_version,
            agent_factory=build_commentary_agent,
            on_failover=deps.on_failover,
            on_attempt=deps.on_attempt,
            budget=deps.budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("calculation.commentary_failed", exc_info=True)
        return fixed_note(results)
    text = text.strip()
    unknown = unknown_numbers(text, allowed_numbers(results))
    if not text or unknown:
        logger.warning("calculation.commentary_rejected unknown_numbers=%s", unknown)
        return fixed_note(results)
    return text


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
    if isinstance(outcome, QuoteOnly):
        return "quote_only"
    return "unresolved"


def _source_of(outcome: TaskOutcome) -> FormulaSource | None:
    if isinstance(outcome, Computed | QuoteOnly):
        return outcome.source
    if isinstance(outcome, NeedsInput) and outcome.plan.retrieved is not None:
        return outcome.plan.retrieved.source
    return None


def _mode(outcome: TaskOutcome) -> str:
    if isinstance(outcome, Computed | NeedsInput):
        return "retrieved" if outcome.plan.formula_id == "retrieved" else "builtin"
    if isinstance(outcome, QuoteOnly):
        return "retrieved"
    return "unknown"


def trace_items(
    outcomes: Sequence[TaskOutcome],
    *,
    queries: Mapping[str, str],
    run_id: str,
    deps: CalculationDeps,
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    """(public items for `metadata.calculation`, private traces for Java's staff table)."""

    versions = get_templates().versions
    credential = deps.models.generation_credential
    model_name = f"{credential.provider}/{credential.model_name}" if credential else "unknown"
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    for outcome in outcomes:
        source = _source_of(outcome)
        mode = _mode(outcome)
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
                {"title": source.source, "heading": _HEADING_SEPARATOR.join(source.heading_path)}
                if source is not None
                else None
            ),
        }
        public.append(item)

        trace: dict[str, JsonValue] = {
            "question_raw": queries.get(outcome.task_id, ""),
            "status": item["status"],
            "models": {"llm": model_name},
        }
        prompt_names = _RETRIEVED_PROMPTS if mode == "retrieved" else _EXTRACTOR_PROMPTS
        trace["prompt_versions"] = {
            name: f"sha256:{versions.get(name, '')}" for name in prompt_names
        }
        if isinstance(outcome, Computed | NeedsInput):
            plan = outcome.plan
            trace["formula_id"] = plan.formula_id
            if plan.retrieved is not None:
                trace["expression"] = plan.retrieved.expression
                trace["variables"] = [v.model_dump(mode="json") for v in plan.retrieved.variables]
                trace["formula_hash"] = _hash(
                    {"expression": plan.retrieved.expression, "variables": trace["variables"]}
                )
            else:
                trace["formula_hash"] = _hash(plan.formula_id)
        if isinstance(outcome, Computed):
            trace["inputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.inputs
            ]
            trace["outputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.outputs
            ]
            trace["checks_passed"] = 7 if source is not None else None
        if isinstance(outcome, NeedsInput):
            trace["known_params"] = dict(outcome.known_params)
        if isinstance(outcome, Unresolved):
            trace["reason"] = outcome.reason
        if source is not None:
            trace["source"] = source.model_dump(mode="json")
        private.append({"itemId": outcome.task_id, "runId": run_id, "trace": trace})
    return public, private
