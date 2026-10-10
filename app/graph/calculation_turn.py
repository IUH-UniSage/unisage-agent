"""The calculation part of a turn, between the node and the graph: what is shown
(a built-in result rendered by Python, or the LLM's own calculation under the "AI
tự tính" notice), the short note after a built-in calculation-only turn (checked
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
    LlmAnswered,
    NeedsInput,
    TaskOutcome,
    Unresolved,
)
from app.graph.streaming import generation_model_settings, run_agent_text_with_failover
from app.rag.prompting import build_calculation_commentary_prompt, get_templates
from app.schemas.clarification import PendingCalculationTask
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

LLM_NOTICE = "Kết quả do AI tự tính, có thể sai - bạn kiểm tra lại giúp mình nhé"
NEEDS_INPUT_LEAD = "Mình cần thêm vài thông tin để tính giúp bạn:"
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_HEADING_SEPARATOR = " \u203a "  # single right angle quote (RUF001 forbids the literal)
# Scale names the note may mention without them being "new numbers".
_ALWAYS_ALLOWED = frozenset({Decimal(0), Decimal(4), Decimal(10)})
_EXTRACTOR_PROMPTS = ("agent_calculation_extractor",)
_LLM_PROMPTS = ("agent_calculation_extractor", "chat_calculation_llm")


@dataclass(frozen=True)
class CalculationTask:
    task_id: str
    query: str


# ---------------------------------------------------------------------------
# What the student sees (deterministic)
# ---------------------------------------------------------------------------


def _source_line(chunk: RetrievedChunk, index: int) -> str:
    heading = _HEADING_SEPARATOR.join(chunk.heading_path)
    return f"- [C{index}] {chunk.source}" + (f" - {heading}" if heading else "")


def render_outcome(outcome: TaskOutcome) -> str:
    """Markdown for one task; a task that needs input is asked on the panel (only its
    own lead, if any, is shown here)."""

    if isinstance(outcome, Computed):
        return render_markdown(outcome.result)
    if isinstance(outcome, LlmAnswered):
        lines = [f"**{LLM_NOTICE}**", "", outcome.text]
        if outcome.sources:
            # Numbered as the LLM saw them, so [C2] in the text matches its line here.
            lines += ["", "Nguồn:"]
            lines += [_source_line(chunk, index) for index, chunk in outcome.sources]
        return "\n".join(lines)
    if isinstance(outcome, Unresolved):
        return UNRESOLVED_MESSAGES[outcome.reason]
    return outcome.lead or ""


def render_outcomes(outcomes: Sequence[TaskOutcome]) -> str:
    blocks = [text for text in (render_outcome(outcome) for outcome in outcomes) if text]
    if not blocks and any(isinstance(outcome, NeedsInput) for outcome in outcomes):
        return NEEDS_INPUT_LEAD
    return "\n\n---\n\n".join(blocks)


def calculation_titles(outcomes: Sequence[TaskOutcome]) -> list[str]:
    """What node 10 is told was already shown - titles only, never a number."""

    titles: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, Computed):
            titles.append(outcome.result.title)
        elif isinstance(outcome, LlmAnswered):
            titles.append(f"Phép tính theo yêu cầu: {outcome.query[:80]}")
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
    """Numbers the note may mention: the student's own inputs and the scale names.
    Never a result - it is already shown right above, so repeating it is noise."""

    allowed = set(_ALWAYS_ALLOWED)
    for result in results:
        for _, value in result.inputs:
            for match in _NUMBER.finditer(value):
                number = _as_number(match.group(0))
                if number is not None:
                    allowed.add(number)
        for _, value in result.outputs:
            for match in _NUMBER.finditer(value):
                number = _as_number(match.group(0))
                if number is not None:
                    allowed.discard(number)
    return allowed


def unknown_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    return [
        match.group(0)
        for match in _NUMBER.finditer(text)
        if (number := _as_number(match.group(0))) is None or number not in allowed
    ]


async def commentary(
    results: Sequence[CalculationResult], user_query: str, deps: CalculationDeps
) -> str:
    """Not streamed: the whole note is checked before anything is sent. A note that
    fails the check (or a failed call) is simply left out - the result line above
    already says everything; a fallback sentence would only repeat it."""

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
        return ""
    text = text.strip()
    unknown = unknown_numbers(text, allowed_numbers(results))
    if unknown:
        # A number that is not an input - an invented one, or the result repeated.
        logger.warning("calculation.commentary_rejected numbers=%s", unknown)
        return ""
    return text


# ---------------------------------------------------------------------------
# Trace: public summary on the message, full trace for staff (SPEC §7.2)
# ---------------------------------------------------------------------------


def _hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _status(outcome: TaskOutcome) -> str:
    if isinstance(outcome, Computed | LlmAnswered):
        return "computed"
    if isinstance(outcome, NeedsInput):
        return "needs_input"
    return "unresolved"


def _mode(outcome: TaskOutcome) -> str:
    if isinstance(outcome, Computed):
        return "builtin"
    if isinstance(outcome, LlmAnswered):
        return "llm"
    if isinstance(outcome, NeedsInput):
        return "llm" if outcome.plan.formula_id == "llm" else "builtin"
    return "llm"


def _chunk_trace(index: int, chunk: RetrievedChunk) -> dict[str, JsonValue]:
    return {
        "ref": f"C{index}",
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
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    """(public items for `metadata.calculation`, private traces for Java's staff table).
    Đúng/Sai buttons go on the `llm` items that were computed."""

    versions = get_templates().versions
    credential = deps.models.generation_credential
    model_name = f"{credential.provider}/{credential.model_name}" if credential else "unknown"
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    for outcome in outcomes:
        mode = _mode(outcome)
        sources = outcome.sources if isinstance(outcome, LlmAnswered) else []
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

        trace: dict[str, JsonValue] = {
            "question_raw": queries.get(outcome.task_id, ""),
            "status": item["status"],
            "mode": mode,
            "models": {"llm": model_name},
            "prompt_versions": {
                name: f"sha256:{versions.get(name, '')}"
                for name in (_LLM_PROMPTS if mode == "llm" else _EXTRACTOR_PROMPTS)
            },
        }
        if isinstance(outcome, Computed):
            trace["formula_id"] = outcome.plan.formula_id
            trace["formula_hash"] = _hash(outcome.plan.formula_id)
            trace["inputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.inputs
            ]
            trace["outputs"] = [
                {"label": label, "value": value} for label, value in outcome.result.outputs
            ]
        if isinstance(outcome, LlmAnswered):
            trace["retrieval_query"] = outcome.plan.retrieval_query
            trace["known_params"] = dict(outcome.known_params)
            trace["answer"] = outcome.text
            trace["sources"] = [_chunk_trace(index, chunk) for index, chunk in sources]
        if isinstance(outcome, NeedsInput):
            trace["formula_id"] = outcome.plan.formula_id
            trace["known_params"] = dict(outcome.known_params)
        if isinstance(outcome, Unresolved):
            trace["reason"] = outcome.reason
        private.append({"itemId": outcome.task_id, "runId": run_id, "trace": trace})
    return public, private
