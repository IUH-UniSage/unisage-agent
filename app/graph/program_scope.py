"""Programme scope check (chuẩn / tăng cường tiếng Anh / chất lượng cao / liên thông) between
the question and the retrieved training-programme documents, after the cohort check
(app/graph/cohort_scope.py) and before generation.

Asked "k21" without naming a programme, retrieval returned only the K21 "tăng cường tiếng
Anh" programme, and the answer gave its "Anh văn 2, 4(4,0,8)" as if it were the student's -
a standard-programme student would register the wrong course (23-11-2026). "Chính quy" does
not tell them apart: the enhanced-English programme is chính quy too.

The programme is read from a training-programme document's title (no programme word ->
chuẩn) and from the standalone questions, then the profile's `he_dao_tao`. Then:
- the asked programme has a document: other programmes' documents are dropped;
- it has none: generation is told to say whose programme the figures are;
- none asked: the standard programme's documents go first and generation names the
  programme of every figure; when a non-standard one was found it also asks which programme
  the user is in (one `he_dao_tao` ask_user_form, saved to the profile when about the user).
Nothing is defaulted into the profile.
"""

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

STANDARD = "chuan"
# Most specific first: a title or question is matched against these in order.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tang_cuong_ta", re.compile(r"tang cuong (?:tieng anh|ta)\b")),
    ("clc", re.compile(r"chat luong cao|\bclc\b")),
    ("lien_thong", re.compile(r"lien thong|van bang 2|van bang hai")),
    (STANDARD, re.compile(r"chuong trinh chuan|chinh quy chuan|\bdai tra\b")),
)
_LABELS = {
    STANDARD: "chương trình chuẩn",
    "tang_cuong_ta": "chương trình tăng cường tiếng Anh",
    "clc": "chương trình chất lượng cao",
    "lien_thong": "chương trình liên thông, văn bằng 2",
}
# `he_dao_tao` options as the profile field lists them (app/rag/prompting metadata fields).
_FORM_OPTIONS = {
    STANDARD: ("chinh_quy_chuan", "Đại học chính quy (chương trình chuẩn)"),
    "tang_cuong_ta": ("chinh_quy_tang_cuong_tieng_anh", "Đại học chính quy tăng cường tiếng Anh"),
    "clc": ("chinh_quy_chat_luong_cao", "Đại học chính quy chất lượng cao"),
    "lien_thong": ("lien_thong_van_bang_2", "Đại học liên thông, văn bằng 2"),
}
_ORDER = {program: rank for rank, program in enumerate(_LABELS)}


@dataclass(frozen=True)
class ProgramScope:
    chunks: list[RetrievedChunk]
    notice: str = ""
    asked: str | None = None
    available: tuple[str, ...] = ()
    dropped: int = 0


def _fold(text: str) -> str:
    """Lower case, no diacritics, "_" as a space: "Tăng cường Tiếng Anh" -> "tang cuong
    tieng anh", and a stored option id "chinh_quy_chuan" reads like its label."""

    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return plain.lower().replace("_", " ")


def _programs_in(text: str) -> set[str]:
    folded = _fold(text)
    found = {program for program, pattern in _PATTERNS if pattern.search(folded)}
    # A non-standard programme named next to a standard word is that programme.
    return found - {STANDARD} if len(found) > 1 and STANDARD in found else found


def document_program(source: str) -> str | None:
    """The programme of a training-programme document, from its title; None for any other
    document (a fee schedule, a regulation), which stays neutral."""

    title = _fold(source_title(source))
    if "chuong trinh dao tao" not in title and "ctdt" not in title:
        return None
    found = _programs_in(title) - {STANDARD}
    return found.pop() if len(found) == 1 else STANDARD if not found else None


def asked_program(questions: Sequence[str], profile_value: str | None) -> tuple[str | None, bool]:
    """(the programme asked about, whether several are - a comparison is left alone). The
    questions win; the profile's `he_dao_tao` only fills a question that names none."""

    named: set[str] = set()
    for question in questions:
        named |= _programs_in(question)
    if len(named) > 1:
        return None, True
    if named:
        return named.pop(), False
    profile = _programs_in(profile_value) if profile_value else set()
    return (profile.pop(), False) if len(profile) == 1 else (None, False)


def check_program_scope(
    questions: Sequence[str],
    profile_value: str | None,
    chunks: Sequence[RetrievedChunk],
) -> ProgramScope:
    kept = list(chunks)
    programs = [document_program(chunk.source) for chunk in kept]
    available = tuple(sorted({p for p in programs if p is not None}, key=_ORDER.__getitem__))
    if not questions or not available:
        return ProgramScope(chunks=kept, available=available)
    asked, comparison = asked_program(questions, profile_value)
    if comparison:
        return ProgramScope(chunks=kept, available=available)
    if asked is not None and asked in available:
        matching = [c for c, p in zip(kept, programs, strict=True) if p in (None, asked)]
        return ProgramScope(
            chunks=matching, asked=asked, available=available, dropped=len(kept) - len(matching)
        )
    # The standard programme first, then the others, each in retrieval order.
    ranked = sorted(
        zip(kept, programs, strict=True),
        key=lambda pair: _ORDER[pair[1]] if pair[1] is not None else -1,
    )
    ordered = [chunk for chunk, _ in ranked]
    notice = _missing_notice(asked, available) if asked is not None else _unnamed_notice(available)
    return ProgramScope(chunks=ordered, notice=notice, asked=asked, available=available)


def _labels(programs: Sequence[str]) -> str:
    return ", ".join(_LABELS[program] for program in programs)


def _missing_notice(asked: str, available: Sequence[str]) -> str:
    return (
        "\n\n## CHƯƠNG TRÌNH ĐÀO TẠO (bắt buộc)\n"
        f"Người dùng hỏi {_LABELS[asked]} nhưng tài liệu tìm được thuộc: {_labels(available)}.\n"
        f"- Câu đầu: nói rõ thông tin dưới đây thuộc {_labels(available)}, không phải "
        f"{_LABELS[asked]}.\n"
        "- Lưu ý các chương trình có thể khác nhau về tên môn, mã học phần và số tín chỉ; "
        "không trộn mã, số liệu giữa các chương trình.\n"
        f"- Cuối: gợi ý xác nhận với Phòng Đào tạo cho {_LABELS[asked]}."
    )


def _unnamed_notice(available: Sequence[str]) -> str:
    if tuple(available) == (STANDARD,):
        return (
            "\n\n## CHƯƠNG TRÌNH ĐÀO TẠO (bắt buộc)\n"
            "Câu hỏi chưa nêu chương trình; tài liệu tìm được là chương trình chuẩn: ghi rõ "
            '"theo chương trình chuẩn" và thêm một câu: sinh viên chương trình khác (tăng '
            "cường tiếng Anh, chất lượng cao) có thể có tên môn, số tín chỉ khác."
        )
    others = [program for program in available if program != STANDARD]
    if STANDARD in available:
        layout = (
            '- Trình bày chương trình chuẩn TRƯỚC, ghi rõ "theo chương trình chuẩn"; sau đó '
            f"TÁCH phần {_labels(others)}, mỗi phần mở đầu bằng tên chương trình."
        )
    else:
        layout = (
            f'- Câu đầu: nói rõ "thông tin dưới đây thuộc {_labels(others)}".\n'
            "- Lưu ý chương trình chuẩn có thể khác về tên môn, mã học phần và số tín chỉ."
        )
    options = [
        {"id": option_id, "label": label}
        for option_id, label in (_FORM_OPTIONS[p] for p in (STANDARD, *others))
    ]
    form = json.dumps(
        {"field": "he_dao_tao", "label": "Chương trình đào tạo", "options": options},
        ensure_ascii=False,
    )
    return (
        "\n\n## CHƯƠNG TRÌNH ĐÀO TẠO (bắt buộc)\n"
        f"Câu hỏi chưa nêu chương trình đào tạo; tài liệu tìm được thuộc: {_labels(available)}.\n"
        f"{layout}\n"
        "- Không trộn mã, số liệu giữa các chương trình; mỗi phần chỉ trích dẫn [n] của chính "
        "tài liệu chương trình đó.\n"
        "- Cuối: hỏi người được hỏi đang học chương trình nào bằng MỘT khối ask_user_form "
        f"(`about` theo người được hỏi), dùng đúng field và options này: {form}"
    )
