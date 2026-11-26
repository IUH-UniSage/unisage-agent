"""Retrieval slots: what a lookup turn asked about, as named fields stored on the ASSISTANT
message next to `retrieval_questions` (`metadata.retrieval_slots`).

`retrieval_questions` is free text: the next turn's rewrite has to read the major and the
cohort back out of it, and the classifier (no reasoning) dropped them - "còn môn vẽ kỹ thuật"
after "cơ lưu chất ngành cơ khí khóa 21 mấy tín chỉ" became "Học phần Vẽ kỹ thuật có bao
nhiêu tín chỉ?", 3 times out of 3 (25/11/2026). With the fields stored, code carries the ones
a follow-up did not change into its question.

Field names are the registry's (`configs/known_metadata_fields.json`, through
`canonical_field`): `nganh_hoc` also for "ngành", "chuyên ngành", "ngành đào tạo";
`khoa_nhap_hoc` is the intake year. `hoc_phan` and `chu_de` are registered there with scope
`retrieval`. Values come from the question only - never from the answer (a prerequisite the
answer named is not what was asked).
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.graph.cohort_scope import asked_cohort, cohort_label
from app.graph.major_scope import fold
from app.graph.metadata_fields import canonical_field
from app.graph.nodes.query_transformation import SubQuery, extract_standalone_question
from app.schemas.intent import ClassifiedTask

# Stored, in this order.
SLOT_FIELDS = (
    "chu_de",
    "hoc_phan",
    "nganh_hoc",
    "khoa_nhap_hoc",
    "he_dao_tao",
    "bac_dao_tao",
    "co_so_hoc",
)
# Written into a follow-up's question when it does not name them, and how they read there.
# Not `he_dao_tao`: the classifier may fill it ("Đại học chính quy") for a question that
# named none, and written into the question it would drop the other programmes' documents.
_CARRIED = {
    "hoc_phan": "học phần {}",
    "nganh_hoc": "ngành {}",
    "khoa_nhap_hoc": "{}",  # cohort_label
}
# A short message that only names a new value ("còn môn vẽ kỹ thuật", "thế khóa luận tốt
# nghiệp thì sao", "hệ chính quy á bạn") - not one that asks something of its own
# ("điều kiện làm khóa luận tốt nghiệp là gì").
_FOLLOW_UP = re.compile(
    r"^(?:con|the|vay|vay con|the con)\b"
    r"|\b(?:thi sao|thi the nao|the nao|nua|a|ha|nhi|nhe)(?: ban| a)?\s*\??$"
)
_FOLLOW_UP_MAX_WORDS = 10


def is_follow_up(message: str) -> bool:
    folded = fold(message).strip(" ?!.")
    return len(folded.split()) <= _FOLLOW_UP_MAX_WORDS and bool(_FOLLOW_UP.search(folded))


def task_slots(task: ClassifiedTask, question: str) -> dict[str, str]:
    """The fields of one SINGLE task, from the classifier's output and its standalone
    `question` (the cohort is read from it)."""

    web = task.web
    raw: dict[str, str | None] = {
        "chu_de": web.subject if web else None,
        "hoc_phan": task.course,
        "nganh_hoc": web.nganh_hoc if web else None,
        "he_dao_tao": web.he_dao_tao if web else None,
        "bac_dao_tao": web.bac_dao_tao if web else None,
        "co_so_hoc": web.co_so if web else None,
    }
    # The classifier may fill these from the asker's declared profile (the profile's
    # "Công nghệ thông tin" stored for a question naming no major became the next turn's
    # major, 25/11/2026): kept only when the question states them.
    for field in ("nganh_hoc", "he_dao_tao", "bac_dao_tao", "co_so_hoc"):
        value = raw.get(field)
        if value and not _named(value, question):
            raw[field] = None
    year = asked_cohort(question)
    if year is not None:
        raw["khoa_nhap_hoc"] = str(year)
    return clean_slots(raw)


def _named(value: str, question: str) -> bool:
    """`question` names `value`: every word of it, accents ignored ("Cơ khí" in "ngành
    Công nghệ kỹ thuật cơ khí")."""

    words = fold(value).split()
    folded = f" {fold(question)} "
    return bool(words) and all(f" {word} " in folded for word in words)


def clean_slots(raw: Mapping[str, object]) -> dict[str, str]:
    """Registry names, known fields only, non-empty text values, in SLOT_FIELDS order."""

    named = {
        canonical_field(str(key)): value.strip()
        for key, value in raw.items()
        if isinstance(value, str) and value.strip()
    }
    return {field: named[field] for field in SLOT_FIELDS if field in named}


def _phrase(field: str, value: str) -> str:
    if field == "khoa_nhap_hoc":
        return cohort_label(int(value)) if value.isdigit() else f"khóa {value}"
    return _CARRIED[field].format(value)


def _states(field: str, value: str, question: str) -> bool:
    """`question` already names that value (or, for the cohort, any cohort)."""

    if field == "khoa_nhap_hoc":
        return asked_cohort(question) is not None
    return fold(value) in fold(question)


def carry_over(
    previous: Mapping[str, str], current: Mapping[str, str], message: str, question: str
) -> tuple[dict[str, str], list[str]]:
    """The turn's slots and the phrases to add to its question. Only for a follow-up: a
    field the previous turn had, this turn's message did not set and its rewritten
    `question` does not state is carried into the stored slots, and the ones in _CARRIED
    also into the question."""

    if not previous or not is_follow_up(message):
        return dict(current), []
    merged = dict(current)
    added: list[str] = []
    for field in SLOT_FIELDS:
        value = previous.get(field)
        if not value or field in current:
            continue
        merged[field] = value
        if field in _CARRIED and not _states(field, value, question):
            added.append(_phrase(field, value))
    return {field: merged[field] for field in SLOT_FIELDS if field in merged}, added


def with_context(sub_query: SubQuery, added: Sequence[str]) -> SubQuery:
    """`sub_query` with `added` written into its standalone question (the first line of its
    retrieval texts)."""

    if not added:
        return sub_query
    suffix = ", ".join(added)

    def extend(text: str) -> str:
        question = extract_standalone_question(text)
        if not question or not text.startswith(question):
            return text
        extended = f"{question.rstrip(' ?')} ({suffix})?"
        return extended + text[len(question) :]

    return replace(
        sub_query,
        retrieval_text=extend(sub_query.retrieval_text),
        extra_retrieval_texts=tuple(extend(text) for text in sub_query.extra_retrieval_texts),
    )


def with_course(slots: Mapping[str, str], course: str) -> dict[str, str]:
    """`slots` with `hoc_phan` set (kept in SLOT_FIELDS order)."""

    return clean_slots({**slots, "hoc_phan": course})
