"""Validate a panel submit against the panel the server stored - never against
anything the client says about the questions (origin, kind, options).

Spec: docs/specs/SPEC-clarification-panel.md §1.5. The normalized answers also
produce the `clarification_answers` metadata for the card (contracts/chat-sse.md
§5) and the text summary stored as the USER message content.
"""

from dataclasses import dataclass
from decimal import Decimal

from pydantic import JsonValue

from app.calculation.formulas import plain
from app.schemas.clarification import (
    LETTER_GRADES,
    Answer,
    ClarificationPanel,
    ClarificationSubmit,
    CourseRow,
    Question,
)


@dataclass(frozen=True)
class AnswerError:
    question_id: str
    reason: str


class ClarificationInvalid(Exception):
    def __init__(self, errors: list[AnswerError]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{error.question_id}: {error.reason}" for error in errors))


@dataclass(frozen=True)
class NormalizedAnswer:
    question: Question
    # JSON-ready value handed to a calculation or to confirmed_metadata: option id,
    # free text, a number as a string, a list of those, or course-row dicts.
    value: JsonValue
    display: str
    option_id: str | None = None
    rows: list[dict[str, JsonValue]] | None = None

    def card_item(self) -> dict[str, JsonValue]:
        item: dict[str, JsonValue] = {
            "question_id": self.question.id,
            "tab_label": self.question.tab_label,
            "prompt": self.question.prompt,
            "kind": self.question.kind,
            "display": None if self.question.kind == "course_table" else self.display,
        }
        if self.question.kind == "choice":
            item["option_id"] = self.option_id
        if self.rows is not None:
            item["rows"] = list(self.rows)
        return item


def _fits_step(value: Decimal, step: Decimal) -> bool:
    return (value / step) == (value / step).to_integral_value()


def _check_number(value: Decimal, question: Question) -> str | None:
    constraint = question.number
    assert constraint is not None
    if not value.is_finite():
        return "phải là một số"
    if not constraint.min <= value <= constraint.max:
        return f"phải từ {plain(constraint.min)} đến {plain(constraint.max)}"
    if not _fits_step(value, constraint.step):
        return f"phải là bội số của {plain(constraint.step)}"
    return None


def _row_error(row: CourseRow) -> str | None:
    if not 1 <= row.credits <= 10:
        return "số tín chỉ phải từ 1 đến 10"
    if isinstance(row.score, str):
        if row.score.strip().upper() not in LETTER_GRADES:
            return "điểm chữ không hợp lệ"
        return None
    if not row.score.is_finite() or not Decimal(0) <= row.score <= Decimal(10):
        return "điểm phải từ 0 đến 10"
    if not _fits_step(row.score, Decimal("0.01")):
        return "điểm có tối đa 2 chữ số thập phân"
    return None


def _normalize(answer: Answer, question: Question) -> NormalizedAnswer | str:
    kind = question.kind
    if kind == "choice":
        if answer.option_id is not None:
            option = next((item for item in question.options if item.id == answer.option_id), None)
            if option is None:
                return "lựa chọn không có trong câu hỏi"
            return NormalizedAnswer(question, option.id, option.label, option_id=option.id)
        if answer.other_text is not None:
            if not question.allow_other:
                return "câu hỏi này không cho nhập lựa chọn khác"
            text = answer.other_text.strip()
            if not text:
                return "cần nhập nội dung cho lựa chọn khác"
            return NormalizedAnswer(question, text, text, option_id=None)
        return "cần chọn một lựa chọn"
    if kind == "number":
        if answer.number is None:
            return "cần nhập một số"
        error = _check_number(answer.number, question)
        return error or NormalizedAnswer(question, plain(answer.number), plain(answer.number))
    if kind == "number_list":
        numbers = answer.numbers
        if not numbers:
            return "cần nhập ít nhất một số"
        if question.max_items is not None and len(numbers) > question.max_items:
            return f"tối đa {question.max_items} giá trị"
        for number in numbers:
            error = _check_number(number, question)
            if error:
                return error
        values: list[JsonValue] = [plain(number) for number in numbers]
        return NormalizedAnswer(question, values, ", ".join(plain(n) for n in numbers))
    if kind == "text":
        if answer.text is None or not answer.text.strip():
            return "cần nhập nội dung"
        text = answer.text.strip()
        if question.max_length is not None and len(text) > question.max_length:
            return f"tối đa {question.max_length} ký tự"
        return NormalizedAnswer(question, text, text)
    # course_table
    rows = answer.rows
    if not rows:
        return "cần ít nhất một môn"
    if question.max_items is not None and len(rows) > question.max_items:
        return f"tối đa {question.max_items} môn"
    normalized: list[dict[str, JsonValue]] = []
    for index, row in enumerate(rows, start=1):
        error = _row_error(row)
        if error:
            return f"dòng {index}: {error}"
        score = row.score.strip().upper() if isinstance(row.score, str) else plain(row.score)
        normalized.append({"name": row.name or None, "credits": row.credits, "score": score})
    display = "; ".join(
        f"{row['name'] or f'Môn {index}'} {row['credits']} TC {row['score']}"
        for index, row in enumerate(normalized, start=1)
    )
    table: list[JsonValue] = list(normalized)
    return NormalizedAnswer(question, table, display, rows=normalized)


def validate_answers(
    panel: ClarificationPanel, submit: ClarificationSubmit
) -> dict[str, NormalizedAnswer]:
    """Every panel question answered exactly once, each valid for its stored kind."""

    errors: list[AnswerError] = []
    by_id = {question.id: question for question in panel.questions}
    seen: dict[str, Answer] = {}
    for answer in submit.answers:
        if answer.question_id not in by_id:
            errors.append(AnswerError(answer.question_id, "câu hỏi không có trong panel"))
        elif answer.question_id in seen:
            errors.append(AnswerError(answer.question_id, "trả lời trùng"))
        else:
            seen[answer.question_id] = answer
    for question_id in by_id:
        if question_id not in seen:
            errors.append(AnswerError(question_id, "chưa trả lời"))

    normalized: dict[str, NormalizedAnswer] = {}
    for question_id, answer in seen.items():
        result = _normalize(answer, by_id[question_id])
        if isinstance(result, str):
            errors.append(AnswerError(question_id, result))
        else:
            normalized[question_id] = result
    if errors:
        raise ClarificationInvalid(errors)
    return normalized


def answers_summary(answers: dict[str, NormalizedAnswer]) -> str:
    """USER message content for a submit turn (history + old clients)."""

    return " · ".join(
        f"{answer.question.tab_label}: {answer.display}" for answer in answers.values()
    )


def answers_metadata(panel: ClarificationPanel, answers: dict[str, NormalizedAnswer]) -> JsonValue:
    """`metadata.clarification_answers` for the USER message (contracts/chat-sse.md §5)."""

    return {
        "schema_version": 1,
        "panel_id": str(panel.panel_id),
        "items": [answers[question.id].card_item() for question in panel.questions],
    }
