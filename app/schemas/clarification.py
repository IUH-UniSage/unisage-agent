"""Clarification panel: the questions a turn asks, the server-side round that
remembers them, and the answers a client submits.

Contract: contracts/chat-sse.md. Spec: docs/specs/SPEC-clarification-panel.md.
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.schemas.intent import ClassifiedTask

QUESTION_ID_PATTERN = r"^q[1-9][0-9]?$"
TASK_ID_PATTERN = r"^T[1-3]$"
# Every question of a turn is asked (no truncation); this only rejects absurd payloads.
PANEL_QUESTIONS_MAX = 50
LETTER_GRADES = ("A+", "A", "B+", "B", "C+", "C", "D+", "D", "F")

QuestionKind = Literal["choice", "number", "number_list", "number_or_list", "text", "course_table"]
QuestionOrigin = Literal["advisory", "calculation"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceOption(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    label: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=200)
    recommended: bool = False


class NumberConstraint(_Strict):
    min: Decimal
    max: Decimal
    step: Decimal = Field(gt=0)
    unit: str | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _ordered(self) -> "NumberConstraint":
        if self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


class _QuestionBody(_Strict):
    id: str = Field(pattern=QUESTION_ID_PATTERN)
    tab_label: str = Field(min_length=1, max_length=24)
    prompt: str = Field(min_length=1, max_length=200)
    kind: QuestionKind
    options: list[ChoiceOption] = Field(default_factory=list)
    allow_other: bool = False
    number: NumberConstraint | None = None
    max_items: int | None = Field(default=None, ge=1, le=30)
    max_length: int | None = Field(default=None, ge=1, le=200)

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "_QuestionBody":
        kind = self.kind
        has_options = bool(self.options)
        rules: dict[str, bool] = {
            "options 2..12 only for choice": (
                2 <= len(self.options) <= 12 if kind == "choice" else not has_options
            ),
            "allow_other only for choice": kind == "choice" or not self.allow_other,
            "number only for number/number_list/number_or_list": (self.number is not None)
            == (kind in {"number", "number_list", "number_or_list"}),
            "max_items only for number_list/number_or_list (<=20) / course_table (<=30)": (
                self.max_items is not None and self.max_items <= 20
                if kind in {"number_list", "number_or_list"}
                else self.max_items is not None
                if kind == "course_table"
                else self.max_items is None
            ),
            "max_length only for text": (self.max_length is not None) == (kind == "text"),
        }
        broken = [rule for rule, ok in rules.items() if not ok]
        if broken:
            raise ValueError(f"invalid {kind} question: {'; '.join(broken)}")
        if kind == "choice" and len({option.id for option in self.options}) != len(self.options):
            raise ValueError("duplicate option id")
        return self


class PublicQuestion(_QuestionBody):
    """What the client sees - no routing information."""


class Question(_QuestionBody):
    origin: QuestionOrigin
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    field: str = Field(min_length=1, max_length=64)

    def public(self) -> PublicQuestion:
        return PublicQuestion.model_validate(
            self.model_dump(exclude={"origin", "task_id", "field"})
        )


class PublicClarificationPanel(_Strict):
    schema_version: Literal[1] = 1
    panel_id: UUID
    questions: list[PublicQuestion] = Field(min_length=1, max_length=PANEL_QUESTIONS_MAX)


class ClarificationPanel(_Strict):
    schema_version: Literal[1] = 1
    panel_id: UUID
    questions: list[Question] = Field(min_length=1, max_length=PANEL_QUESTIONS_MAX)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ClarificationPanel":
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question id")
        return self

    def public(self) -> PublicClarificationPanel:
        return PublicClarificationPanel(
            panel_id=self.panel_id,
            questions=[question.public() for question in self.questions],
        )


# --- pending round (server-side state) ---------------------------------------


class FormulaVariableSpec(_Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(min_length=1, max_length=80)
    unit: str | None = Field(default=None, max_length=20)
    min: Decimal | None = None
    max: Decimal | None = None


class FormulaSource(_Strict):
    chunk_id: str
    document_id: str | None = None
    source: str
    heading_path: list[str] = Field(default_factory=list)
    chunk_hash: str
    source_quote: str = Field(max_length=600)


class RetrievedFormulaPlan(_Strict):
    expression: str = Field(max_length=200)
    variables: list[FormulaVariableSpec] = Field(min_length=1, max_length=10)
    result_label: str = Field(min_length=1, max_length=80)
    source: FormulaSource


class CalculationPlan(_Strict):
    formula_id: Literal["gpa", "course_score", "grade_conversion", "retrieved"]
    retrieved: RetrievedFormulaPlan | None = None

    @model_validator(mode="after")
    def _retrieved_iff_needed(self) -> "CalculationPlan":
        if (self.formula_id == "retrieved") != (self.retrieved is not None):
            raise ValueError("retrieved plan is required exactly for formula_id 'retrieved'")
        return self


class PendingAdvisoryTask(_Strict):
    """The advisory part of the turn: one generation call answers every advisory task
    together, so its questions resume all of them at once."""

    kind: Literal["advisory"] = "advisory"
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    origin_tasks: list[ClassifiedTask] = Field(min_length=1, max_length=3)


class PendingCalculationTask(_Strict):
    kind: Literal["calculation"] = "calculation"
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    query: str = Field(min_length=1)
    plan: CalculationPlan
    known_params: dict[str, JsonValue] = Field(default_factory=dict)


PendingTask = Annotated[PendingAdvisoryTask | PendingCalculationTask, Field(discriminator="kind")]


class PendingRound(_Strict):
    schema_version: Literal[2] = 2
    panel: ClarificationPanel
    assistant_message_id: UUID | None = None
    original_query: str
    tasks: list[PendingTask] = Field(min_length=1, max_length=3)
    # How many panels in a row this question has needed (no limit - each needs the student).
    chain_depth: int = Field(default=1, ge=1)
    created_at: datetime

    @model_validator(mode="after")
    def _questions_match_tasks(self) -> "PendingRound":
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate task id")
        asked = {question.task_id for question in self.panel.questions}
        if asked != set(task_ids):
            raise ValueError("every task needs a question and every question a task")
        for question in self.panel.questions:
            task = next(task for task in self.tasks if task.task_id == question.task_id)
            if question.origin != task.kind:
                raise ValueError(f"question {question.id} origin does not match its task")
        return self


# --- answers (from the client) ----------------------------------------------


class CourseRow(_Strict):
    name: str | None = Field(default=None, max_length=80)
    credits: int
    score: Decimal | str  # a number, or a letter grade (A+..F)

    @field_validator("score", mode="before")
    @classmethod
    def _number_or_letter(cls, value: object) -> object:
        """Pydantic keeps "8.5" as `str` in this union - turn numeric text into a
        Decimal so only real letter grades stay strings."""

        if isinstance(value, str) and value.strip().upper() not in LETTER_GRADES:
            try:
                return Decimal(value.strip().replace(",", "."))
            except InvalidOperation:
                return value
        return value


class Answer(_Strict):
    question_id: str = Field(pattern=QUESTION_ID_PATTERN)
    option_id: str | None = Field(default=None, max_length=64)
    other_text: str | None = Field(default=None, max_length=200)
    number: Decimal | None = None
    numbers: list[Decimal] | None = Field(default=None, max_length=20)
    text: str | None = Field(default=None, max_length=200)
    rows: list[CourseRow] | None = Field(default=None, max_length=30)

    @model_validator(mode="after")
    def _exactly_one_value(self) -> "Answer":
        given = [
            name
            for name in ("option_id", "other_text", "number", "numbers", "text", "rows")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise ValueError("an answer carries exactly one value field")
        return self


class ClarificationSubmit(_Strict):
    action: Literal["submit"]
    panel_id: UUID
    answers: list[Answer] = Field(min_length=1, max_length=PANEL_QUESTIONS_MAX)


class ClarificationCancel(_Strict):
    action: Literal["cancel"]
    panel_id: UUID


ClarificationAction = Annotated[
    ClarificationSubmit | ClarificationCancel, Field(discriminator="action")
]
