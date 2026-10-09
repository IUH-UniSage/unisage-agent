"""Clarification panel v2 schemas and answer validation (contracts/chat-sse.md)."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.graph.clarification_answers import (
    ClarificationInvalid,
    answers_metadata,
    answers_summary,
    validate_answers,
)
from app.schemas.chat import ChatStreamRequest
from app.schemas.clarification import (
    ClarificationPanel,
    ClarificationSubmit,
    PendingRound,
    PublicClarificationPanel,
    Question,
)
from app.schemas.intent import ClassifiedTask

NUMBER = {"min": "0", "max": "10", "step": "0.01", "unit": None}

# The four question shapes from contracts/chat-sse.md §3, plus routing fields.
CHOICE: dict[str, Any] = {
    "id": "q1",
    "tab_label": "Khoá",
    "prompt": "Bạn thuộc khoá nào?",
    "kind": "choice",
    "options": [
        {"id": "k19", "label": "K19", "description": None, "recommended": False},
        {"id": "k20", "label": "K20", "description": "Nhập học 2020", "recommended": True},
    ],
    "allow_other": True,
    "origin": "advisory",
    "task_id": "T1",
    "field": "cohort",
}
SCORE: dict[str, Any] = {
    "id": "q2",
    "tab_label": "Điểm CK",
    "prompt": "Điểm cuối kỳ (thang 10)",
    "kind": "number",
    "number": NUMBER,
    "origin": "calculation",
    "task_id": "T2",
    "field": "ck",
}
PRACTICE: dict[str, Any] = {
    "id": "q3",
    "tab_label": "Điểm TH",
    "prompt": "Các cột điểm thực hành",
    "kind": "number_list",
    "number": NUMBER,
    "max_items": 20,
    "origin": "calculation",
    "task_id": "T2",
    "field": "th",
}
COURSES: dict[str, Any] = {
    "id": "q4",
    "tab_label": "Các môn",
    "prompt": "Nhập các môn để tính GPA",
    "kind": "course_table",
    "max_items": 30,
    "origin": "calculation",
    "task_id": "T3",
    "field": "courses",
}
TEXT: dict[str, Any] = {
    "id": "q5",
    "tab_label": "Ghi chú",
    "prompt": "Hệ đào tạo",
    "kind": "text",
    "max_length": 50,
    "origin": "advisory",
    "task_id": "T1",
    "field": "program",
}


def _panel(*questions: dict[str, Any]) -> ClarificationPanel:
    return ClarificationPanel.model_validate(
        {
            "panel_id": str(uuid4()),
            "questions": list(questions or (CHOICE, SCORE, PRACTICE, COURSES, TEXT)),
        }
    )


def _submit(panel: ClarificationPanel, *answers: dict[str, Any]) -> ClarificationSubmit:
    return ClarificationSubmit.model_validate(
        {"action": "submit", "panel_id": str(panel.panel_id), "answers": list(answers)}
    )


GOOD_ANSWERS: tuple[dict[str, Any], ...] = (
    {"question_id": "q1", "option_id": "k20"},
    {"question_id": "q2", "number": "6.5"},
    {"question_id": "q3", "numbers": ["9", "8"]},
    {
        "question_id": "q4",
        "rows": [
            {"name": "Toán", "credits": 3, "score": "8.5"},
            {"name": None, "credits": 2, "score": "b+"},
        ],
    },
    {"question_id": "q5", "text": "  Chất lượng cao "},
)


# --- question / panel shape ---------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [
        {**CHOICE, "options": CHOICE["options"][:1]},
        {**CHOICE, "options": CHOICE["options"] * 2},
        {**SCORE, "number": None},
        {**SCORE, "allow_other": True},
        {**SCORE, "options": CHOICE["options"]},
        {**PRACTICE, "max_items": 21},
        {**COURSES, "max_items": None},
        {**TEXT, "max_length": None},
        {**CHOICE, "id": "q13"},
        {**CHOICE, "unexpected": 1},
        {**SCORE, "number": {**NUMBER, "min": "11"}},
    ],
)
def test_invalid_question_shapes_are_rejected(broken: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(broken)


def test_panel_allows_twelve_questions_but_not_thirteen() -> None:
    questions = [{**SCORE, "id": f"q{i}"} for i in range(1, 13)]
    assert len(_panel(*questions).questions) == 12
    with pytest.raises(ValidationError):
        _panel(*questions, {**SCORE, "id": "q1"})


def test_public_panel_drops_routing_fields() -> None:
    public = _panel().public().model_dump(mode="json")
    for question in public["questions"]:
        assert not {"origin", "task_id", "field"} & question.keys()
    assert public["questions"][1]["number"] == NUMBER
    PublicClarificationPanel.model_validate(public)


def test_pending_round_requires_matching_tasks() -> None:
    panel = _panel(CHOICE, SCORE)
    advisory = {
        "kind": "advisory",
        "task_id": "T1",
        "origin_task": ClassifiedTask(intent="academic_advisory", query="q").model_dump(),
    }
    calculation = {
        "kind": "calculation",
        "task_id": "T2",
        "query": "điểm học phần",
        "plan": {"formula_id": "course_score"},
        "known_params": {"tclt": 2},
    }
    base = {"panel": panel.model_dump(), "original_query": "q", "created_at": datetime.now(UTC)}
    round_ = PendingRound.model_validate({**base, "tasks": [advisory, calculation]})
    assert round_.model_dump(mode="json")["schema_version"] == 2
    with pytest.raises(ValidationError):
        PendingRound.model_validate({**base, "tasks": [advisory]})
    with pytest.raises(ValidationError):
        PendingRound.model_validate(
            {**base, "tasks": [{**advisory, "task_id": "T2"}, {**calculation, "task_id": "T1"}]}
        )


def test_retrieved_plan_required_only_for_retrieved_formula() -> None:
    from app.schemas.clarification import CalculationPlan

    with pytest.raises(ValidationError):
        CalculationPlan.model_validate({"formula_id": "retrieved"})
    with pytest.raises(ValidationError):
        CalculationPlan.model_validate({"formula_id": "gpa", "retrieved": {"expression": "a"}})


# --- answers ------------------------------------------------------------------


def test_valid_answers_are_normalized() -> None:
    panel = _panel()
    answers = validate_answers(panel, _submit(panel, *GOOD_ANSWERS))

    assert answers["q1"].value == "k20" and answers["q1"].display == "K20"
    assert answers["q2"].value == "6.5"
    assert answers["q3"].value == ["9", "8"]
    assert answers["q4"].rows == [
        {"name": "Toán", "credits": 3, "score": "8.5"},
        {"name": None, "credits": 2, "score": "B+"},
    ]
    assert answers["q5"].value == "Chất lượng cao"
    assert answers_summary(answers).startswith("Khoá: K20 · Điểm CK: 6.5 · Điểm TH: 9, 8")

    metadata = answers_metadata(panel, answers)
    assert isinstance(metadata, dict)
    items = metadata["items"]
    assert isinstance(items, list)
    assert items[0] == {
        "question_id": "q1",
        "tab_label": "Khoá",
        "prompt": "Bạn thuộc khoá nào?",
        "kind": "choice",
        "display": "K20",
        "option_id": "k20",
    }
    table = items[3]
    assert isinstance(table, dict)
    assert table["display"] is None and table["rows"]


def test_other_text_has_null_option_id() -> None:
    panel = _panel(CHOICE)
    answers = validate_answers(panel, _submit(panel, {"question_id": "q1", "other_text": "K18"}))
    assert answers["q1"].card_item()["option_id"] is None
    assert answers["q1"].display == "K18"


def _errors(panel: ClarificationPanel, *answers: dict[str, Any]) -> dict[str, str]:
    with pytest.raises(ClarificationInvalid) as error:
        validate_answers(panel, _submit(panel, *answers))
    return {item.question_id: item.reason for item in error.value.errors}


def test_missing_duplicate_and_unknown_answers() -> None:
    panel = _panel(CHOICE, SCORE)
    errors = _errors(
        panel,
        {"question_id": "q1", "option_id": "k20"},
        {"question_id": "q1", "option_id": "k19"},
        {"question_id": "q9", "number": "1"},
    )
    assert errors == {
        "q1": "trả lời trùng",
        "q9": "câu hỏi không có trong panel",
        "q2": "chưa trả lời",
    }


@pytest.mark.parametrize(
    ("question", "answer", "fragment"),
    [
        (CHOICE, {"option_id": "k99"}, "không có trong câu hỏi"),
        (CHOICE, {"number": "1"}, "cần chọn"),
        ({**CHOICE, "allow_other": False}, {"other_text": "K18"}, "không cho nhập"),
        (CHOICE, {"other_text": "   "}, "cần nhập nội dung"),
        (SCORE, {"number": "10.5"}, "từ 0 đến 10"),
        (SCORE, {"number": "6.555"}, "bội số"),
        (SCORE, {"option_id": "x"}, "cần nhập một số"),
        (PRACTICE, {"numbers": []}, "ít nhất một số"),
        ({**PRACTICE, "max_items": 2}, {"numbers": ["1", "2", "3"]}, "tối đa 2"),
        (TEXT, {"text": "x" * 51}, "tối đa 50"),
        (COURSES, {"rows": [{"credits": 3, "score": "8"}] * 31}, ""),
        (COURSES, {"rows": [{"credits": 0, "score": "8"}]}, "dòng 1"),
        (COURSES, {"rows": [{"credits": 3, "score": "G"}]}, "điểm chữ"),
        (COURSES, {"rows": [{"credits": 3, "score": "8.555"}]}, "2 chữ số"),
    ],
)
def test_answer_must_fit_the_stored_question(
    question: dict[str, Any], answer: dict[str, Any], fragment: str
) -> None:
    panel = _panel(question)
    try:
        submit = _submit(panel, {"question_id": question["id"], **answer})
    except ValidationError:
        assert fragment == ""  # rejected by the request schema itself (e.g. > 30 rows)
        return
    with pytest.raises(ClarificationInvalid) as error:
        validate_answers(panel, submit)
    assert fragment in error.value.errors[0].reason


def test_answer_carries_exactly_one_value() -> None:
    panel = _panel(SCORE)
    with pytest.raises(ValidationError):
        _submit(panel, {"question_id": "q2", "number": "1", "text": "1"})
    with pytest.raises(ValidationError):
        _submit(panel, {"question_id": "q2"})


# --- request ------------------------------------------------------------------


def test_request_needs_exactly_one_of_message_or_clarification() -> None:
    panel_id = str(uuid4())
    ChatStreamRequest.model_validate({"conversation_id": "c", "message": "hi"})
    ChatStreamRequest.model_validate(
        {"conversation_id": "c", "clarification": {"action": "cancel", "panel_id": panel_id}}
    )
    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate({"conversation_id": "c"})
    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate(
            {
                "conversation_id": "c",
                "message": "hi",
                "clarification": {"action": "cancel", "panel_id": panel_id},
            }
        )
    with pytest.raises(ValidationError):
        ChatStreamRequest.model_validate(
            {"conversation_id": "c", "clarification": {"action": "drop", "panel_id": panel_id}}
        )
