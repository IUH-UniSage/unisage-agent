"""Build the one clarification panel a turn may raise.

Questions come from two places: the advisory answer's captured ask_user_form
blocks (choice questions the model asked for) and calculation tasks missing
parameters (built deterministically from `ParamSpec`). Each task's questions
keep their order; ids are assigned q1, q2, ... across the whole panel, and
every question is asked - there is no per-panel or per-chain limit. A field
the student already answered is in `confirmed_metadata` and never asked again,
which is what keeps the model from asking in circles. Spec:
docs/specs/SPEC-clarification-panel.md §1, SPEC-calculation-node.md §4.
"""

import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.schemas.clarification import (
    PANEL_QUESTIONS_MAX,
    ChoiceOption,
    ClarificationPanel,
    PendingAdvisoryTask,
    PendingCalculationTask,
    PendingRound,
    Question,
)
from app.schemas.intent import ClassifiedTask

logger = logging.getLogger(__name__)

_CHOICE_OPTIONS_MAX = 12


@dataclass(frozen=True)
class TaskQuestions:
    """One task's pending state plus the questions it needs answered, without ids."""

    task: PendingAdvisoryTask | PendingCalculationTask
    questions: list[dict[str, Any]]


def _slug(text: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9_-]+", "_", ascii_text).strip("_")[:64]


def advisory_questions(
    ask_forms: Sequence[Mapping[str, Any]], *, confirmed_metadata: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Choice questions from the model's ask_user_form blocks. A field already
    confirmed, asked twice, or with fewer than 2 usable options is dropped."""

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for form in ask_forms:
        fields = form.get("fields")
        if not isinstance(fields, list):
            continue
        for item in fields:
            if not isinstance(item, Mapping):
                continue
            field = str(item.get("field") or "").strip()[:64]
            if not field or field in seen or field in confirmed_metadata:
                continue
            options: list[ChoiceOption] = []
            for raw in item.get("options") or []:
                if not isinstance(raw, Mapping):
                    continue
                label = str(raw.get("label") or raw.get("id") or "").strip()[:120]
                option_id = _slug(str(raw.get("id") or label))
                if label and option_id and option_id not in {o.id for o in options}:
                    options.append(ChoiceOption(id=option_id, label=label))
            if len(options) < 2:
                continue
            label = str(item.get("label") or field).strip()
            seen.add(field)
            questions.append(
                {
                    "tab_label": label[:24],
                    "prompt": label[:200],
                    "kind": "choice",
                    "options": options[:_CHOICE_OPTIONS_MAX],
                    "allow_other": True,
                    "origin": "advisory",
                    "field": field,
                }
            )
    return questions


def advisory_task(task_id: str, origin_tasks: Sequence[ClassifiedTask]) -> PendingAdvisoryTask:
    return PendingAdvisoryTask(task_id=task_id, origin_tasks=list(origin_tasks))


def build_round(
    parts: Sequence[TaskQuestions], *, original_query: str, chain_depth: int
) -> PendingRound | None:
    """One panel for every task that needs input - every question is asked - or None
    when nothing is asked. Only an absurd number of questions (> PANEL_QUESTIONS_MAX)
    is cut, and logged."""

    questions: list[Question] = []
    tasks: list[PendingAdvisoryTask | PendingCalculationTask] = []
    dropped = 0
    for part in parts:
        taken: list[Question] = []
        for draft in part.questions:
            if len(questions) + len(taken) >= PANEL_QUESTIONS_MAX:
                dropped += 1
                continue
            index = len(questions) + len(taken) + 1
            taken.append(
                Question.model_validate({**draft, "id": f"q{index}", "task_id": part.task.task_id})
            )
        if taken:
            questions += taken
            tasks.append(part.task)
    if dropped:
        logger.warning("clarification.questions_truncated dropped=%d", dropped)
    if not questions:
        return None
    return PendingRound(
        panel=ClarificationPanel(panel_id=uuid4(), questions=questions),
        original_query=original_query,
        tasks=tasks,
        chain_depth=chain_depth,
        created_at=datetime.now(UTC),
    )
