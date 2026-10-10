from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.schemas.intent import ClassifiedTask

FORM = {
    "type": "ask_user_form",
    "fields": [
        {
            "field": "nganh",
            "label": "Ngành học của bạn",
            "options": [
                {"id": "CNTT", "label": "Công nghệ Thông tin"},
                {"id": "kt", "label": "Kế toán"},
            ],
        },
        {"field": "khoa", "options": [{"id": "k20"}]},  # only one option: dropped
        {
            "field": "he",
            "label": "Hệ đào tạo",
            "options": [{"label": "Đại trà"}, {"label": "Chất lượng cao"}],
        },
    ],
}


def test_ask_form_becomes_choice_questions() -> None:
    questions = advisory_questions([FORM, FORM], confirmed_metadata={})
    assert [q["field"] for q in questions] == [
        "nganh",
        "he",
    ]  # duplicates and 1-option field dropped
    nganh = questions[0]
    assert nganh["kind"] == "choice" and nganh["allow_other"] is True
    assert [(o.id, o.label) for o in nganh["options"]] == [
        ("cntt", "Công nghệ Thông tin"),
        ("kt", "Kế toán"),
    ]
    assert [o.id for o in questions[1]["options"]] == ["dai_tra", "chat_luong_cao"]


def test_confirmed_fields_are_not_asked_again() -> None:
    assert [
        q["field"] for q in advisory_questions([FORM], confirmed_metadata={"nganh": "cntt"})
    ] == ["he"]


def _task(task_id: str) -> TaskQuestions:
    return TaskQuestions(
        task=advisory_task(task_id, [ClassifiedTask(intent="academic_advisory", query="q")]),
        questions=advisory_questions([FORM], confirmed_metadata={}),
    )


def test_build_round_numbers_questions_across_tasks() -> None:
    pending = build_round([_task("T1"), _task("T2")], original_query="q", chain_depth=2)
    assert pending is not None
    assert [(q.id, q.task_id) for q in pending.panel.questions] == [
        ("q1", "T1"),
        ("q2", "T1"),
        ("q3", "T2"),
        ("q4", "T2"),
    ]
    assert pending.chain_depth == 2


def test_build_round_asks_every_question() -> None:
    many = TaskQuestions(task=_task("T1").task, questions=_task("T1").questions * 7)
    pending = build_round([many, _task("T2")], original_query="q", chain_depth=5)
    assert pending is not None
    assert len(pending.panel.questions) == 16
    assert pending.panel.questions[-1].id == "q16"
    assert [task.task_id for task in pending.tasks] == ["T1", "T2"]
    assert pending.chain_depth == 5


def test_only_an_absurd_number_of_questions_is_cut() -> None:
    many = TaskQuestions(task=_task("T1").task, questions=_task("T1").questions * 30)
    pending = build_round([many], original_query="q", chain_depth=1)
    assert pending is not None
    assert len(pending.panel.questions) == 50


def test_nothing_to_ask_gives_no_round() -> None:
    empty = TaskQuestions(task=_task("T1").task, questions=[])
    assert build_round([empty], original_query="q", chain_depth=1) is None
