import pytest

from app.graph.nodes.query_transformation import SubQuery
from app.graph.retrieval_slots import (
    carry_over,
    clean_slots,
    is_follow_up,
    task_slots,
    with_context,
    with_course,
)
from app.rag.prompting.builder import (
    PREVIOUS_RETRIEVAL_SLOTS_LABEL,
    append_recent_history,
    describe_slots,
    last_retrieval_slots,
)
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask
from app.schemas.web_search import WebSlots

_PREVIOUS = {
    "chu_de": "số tín chỉ",
    "hoc_phan": "Cơ lưu chất",
    "nganh_hoc": "Công nghệ kỹ thuật cơ khí",
    "khoa_nhap_hoc": "2025",
}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("còn môn vẽ kỹ thuật", True),
        ("thế khóa luận tốt nghiệp thì sao", True),
        ("ngành kế toán thì sao", True),
        ("hệ chính quy á bạn", True),
        ("khóa 20 thì sao?", True),
        ("điều kiện làm khóa luận tốt nghiệp là gì", False),
        ("học phần tiên quyết của môn Cơ lưu chất ngành cơ khí khóa 21", False),
    ],
)
def test_is_follow_up(message: str, expected: bool) -> None:
    assert is_follow_up(message) is expected


def test_clean_slots_uses_the_registry_names_only() -> None:
    raw = {"nganh": "Cơ khí", "chuyen_nganh": "x", "mon_hoc": "Vẽ kỹ thuật", "khoa": "2025"}

    # "nganh" and "chuyen_nganh" are both nganh_hoc: one field, never two.
    assert clean_slots(raw | {"unknown": "y", "co_so": " "}) == {
        "hoc_phan": "Vẽ kỹ thuật",
        "nganh_hoc": "x",
        "khoa_nhap_hoc": "2025",
    }


def test_task_slots_read_the_classifier_fields_and_the_cohort() -> None:
    task = ClassifiedTask(
        intent="academic_advisory",
        query="q",
        course="Cơ lưu chất",
        web=WebSlots(topics=["chuong_trinh_dao_tao"], subject="số tín chỉ", nganh_hoc="Cơ khí"),
    )

    assert task_slots(task, "Cơ lưu chất ngành cơ khí khóa K21 (nhập học 2025) mấy tín?") == {
        "chu_de": "số tín chỉ",
        "hoc_phan": "Cơ lưu chất",
        "nganh_hoc": "Cơ khí",
        "khoa_nhap_hoc": "2025",
    }


def test_a_follow_up_gets_back_the_major_and_cohort_its_rewrite_dropped() -> None:
    """ "còn môn vẽ kỹ thuật" was rewritten without the major and cohort (25/11/2026)."""

    slots, added = carry_over(
        _PREVIOUS,
        {"chu_de": "số tín chỉ", "hoc_phan": "Vẽ kỹ thuật"},
        "còn môn vẽ kỹ thuật",
        "Học phần Vẽ kỹ thuật có bao nhiêu tín chỉ?",
    )

    assert added == ["ngành Công nghệ kỹ thuật cơ khí", "khóa 21 (nhập học 2025)"]
    assert slots == {
        "chu_de": "số tín chỉ",
        "hoc_phan": "Vẽ kỹ thuật",
        "nganh_hoc": "Công nghệ kỹ thuật cơ khí",
        "khoa_nhap_hoc": "2025",
    }


def test_nothing_is_carried_into_a_new_question_or_what_the_rewrite_already_states() -> None:
    new = {"chu_de": "điều kiện làm khóa luận"}
    assert carry_over(_PREVIOUS, new, "điều kiện làm khóa luận tốt nghiệp là gì", "Điều kiện?") == (
        new,
        [],
    )
    # The rewrite kept them: stored, not added again.
    slots, added = carry_over(
        _PREVIOUS,
        {"hoc_phan": "Vẽ kỹ thuật"},
        "còn môn vẽ kỹ thuật",
        "Học phần Vẽ kỹ thuật ngành Công nghệ kỹ thuật cơ khí khóa K21 có bao nhiêu tín chỉ?",
    )
    assert added == []
    assert slots["nganh_hoc"] == "Công nghệ kỹ thuật cơ khí"
    # A follow-up naming its own value keeps it.
    slots, added = carry_over(
        _PREVIOUS, {"nganh_hoc": "Kế toán"}, "ngành kế toán thì sao", "Cơ lưu chất ngành Kế toán?"
    )
    assert slots["nganh_hoc"] == "Kế toán" and added == ["khóa 21 (nhập học 2025)"]


def test_with_context_writes_into_the_standalone_question_of_every_retrieval_text() -> None:
    sub_query = SubQuery(
        question="còn môn vẽ kỹ thuật",
        retrieval_text="Học phần Vẽ kỹ thuật có bao nhiêu tín chỉ?\n\nHọc phần Vẽ kỹ thuật có ...",
        extra_retrieval_texts=("Học phần Vẽ kỹ thuật có bao nhiêu tín chỉ?\n\nCTĐT | Vẽ kỹ thuật",),
    )

    extended = with_context(sub_query, ["ngành Cơ khí", "khóa 21 (nhập học 2025)"])

    question = "Học phần Vẽ kỹ thuật có bao nhiêu tín chỉ (ngành Cơ khí, khóa 21 (nhập học 2025))?"
    assert extended.retrieval_text == f"{question}\n\nHọc phần Vẽ kỹ thuật có ..."
    assert extended.extra_retrieval_texts == (f"{question}\n\nCTĐT | Vẽ kỹ thuật",)
    assert with_context(sub_query, []) is sub_query


def test_the_classifier_prompt_names_the_previous_turns_slots() -> None:
    history = [
        HistoryMessage(role="USER", content="cơ lưu chất ngành cơ khí khóa 21 mấy tín"),
        HistoryMessage(
            role="ASSISTANT",
            content="...",
            retrieval_questions=["Cơ lưu chất ngành cơ khí khóa K21 mấy tín chỉ?"],
            retrieval_slots=_PREVIOUS,
        ),
    ]

    assert last_retrieval_slots(history) == _PREVIOUS
    assert describe_slots(_PREVIOUS) == (
        "điều đang hỏi: số tín chỉ; học phần: Cơ lưu chất; ngành: Công nghệ kỹ thuật cơ khí; "
        "khóa: K21 (nhập học 2025)"
    )
    prompt = append_recent_history("còn môn vẽ kỹ thuật", history)
    assert prompt.endswith(f"{PREVIOUS_RETRIEVAL_SLOTS_LABEL} {describe_slots(_PREVIOUS)}")
    # A calculation turn after the lookup ends the chain.
    assert (
        last_retrieval_slots(
            [*history, HistoryMessage(role="ASSISTANT", content="=", calculation=True)]
        )
        == {}
    )


def test_with_course_fills_hoc_phan_in_field_order() -> None:
    slots = {"chu_de": "học phần tiên quyết", "nganh_hoc": "Cơ khí", "khoa_nhap_hoc": "2025"}

    assert list(with_course(slots, "Cơ lưu chất")) == [
        "chu_de",
        "hoc_phan",
        "nganh_hoc",
        "khoa_nhap_hoc",
    ]


def test_task_slots_drop_a_major_the_question_does_not_name() -> None:
    """The classifier copied the asker's profile major into `web` for a question naming none
    (25/11/2026): it must not become the conversation's major."""

    task = ClassifiedTask(
        intent="academic_advisory",
        query="q",
        course="Cơ lưu chất",
        web=WebSlots(
            topics=["chuong_trinh_dao_tao"],
            subject="học phần tiên quyết",
            nganh_hoc="Công nghệ Thông tin",
            he_dao_tao="Đại học chính quy",
        ),
    )

    assert task_slots(task, "Học phần tiên quyết của môn Cơ lưu chất là những học phần nào?") == {
        "chu_de": "học phần tiên quyết",
        "hoc_phan": "Cơ lưu chất",
    }


def test_task_slots_keep_a_major_the_question_names_by_its_abbreviation() -> None:
    """ "hoc phi nganh cntt": the classifier wrote "Công nghệ thông tin", the question only
    "cntt" - the major was dropped and the follow-up "K18 thì sao" had none (27/11/2026)."""

    task = ClassifiedTask(
        intent="academic_advisory",
        query="hoc phi nganh cntt",
        web=WebSlots(
            topics=["hoc_phi"], subject="mức thu học phí", nganh_hoc="Công nghệ thông tin"
        ),
    )

    assert task_slots(task, "Học phí ngành CNTT là bao nhiêu?") == {
        "chu_de": "mức thu học phí",
        "nganh_hoc": "Công nghệ thông tin",
    }
