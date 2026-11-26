"""Cohort scope check between the question and the retrieved documents, before generation.

The training programmes of consecutive cohorts are near-identical, so retrieval returns
whichever cohorts' documents it has. Asked about K18 with only K16 and K19 indexed, the
answer gave K16's numbers as K18's, or mixed K16's and K19's course codes in one answer
(23-11-2026) - the prompt rule "nói rõ chưa có văn bản cho đúng khóa" was not enough.

Cohorts are read from the standalone question and from each chunk's document title ("...
khóa 16 năm học 2020-2021"); a document whose title names no cohort (a fee schedule, a
regulation) is neutral and never dropped. Then:
- some document is of the asked cohort: other cohorts' documents are dropped;
- none is: they are all kept, grouped by document, and generation is told to say first that
  the asked cohort has no document and which cohorts it has, then answer per document;
- the question names no cohort but documents do: generation is told to name the cohort of
  every figure it gives.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

# IUH numbers its cohorts from 2004: K18 entered in 2022.
_COHORT_BASE_YEAR = 2004
_KHOA = r"kh(?:óa|oá|oa)"
_INTAKE_YEAR = re.compile(r"nhập học(?:\s+năm)?\s+(20\d{2})", re.IGNORECASE)
_COHORT_NUMBER = re.compile(rf"\b(?:{_KHOA}\s+K?\s?|K)(\d{{1,2}})\b", re.IGNORECASE)
_COHORT_YEAR = re.compile(
    rf"\b{_KHOA}\s+(?:tuyển sinh\s+)?(?:năm học\s+)?(20\d{{2}})\b", re.IGNORECASE
)


@dataclass(frozen=True)
class CohortScope:
    chunks: list[RetrievedChunk]
    # Per-turn instruction for generation ("" when none is needed).
    notice: str = ""
    asked: int | None = None
    available: tuple[int, ...] = ()
    dropped: int = 0


# Shared by the cohort, programme and major notices: each asks for the answer split per
# document, and the model wrote a "không thấy học phần X" part for every document found
# (25/11/2026), and an unasked "related" course read from the wrong row (Tiếng Anh 1 given
# the code of Kinh tế chính trị). Answers from other cohorts/programmes stay.
ONLY_WHAT_WAS_ASKED = (
    "- Trình bày thông tin TRẢ LỜI câu hỏi từ mọi tài liệu có nó (kể cả của khóa, chương trình "
    "khác, ghi rõ của ai). KHÔNG nêu mục khác không được hỏi (VD học phần khác) làm thông tin "
    'liên quan/tham khảo. KHÔNG viết một phần/một câu riêng cho mỗi tài liệu chỉ để nói "không '
    'thấy/không có"; thiếu gì thì gộp vào MỘT câu, không gắn [n] cho câu đó.'
)


def cohort_label(intake_year: int) -> str:
    return f"khóa {intake_year - _COHORT_BASE_YEAR} (nhập học {intake_year})"


def _single(years: set[int]) -> int | None:
    plausible = {year for year in years if 2005 <= year <= 2050}
    return plausible.pop() if len(plausible) == 1 else None


def asked_cohort(question: str) -> int | None:
    """The intake year the question is about; None when it names none, or several (a
    comparison is not one cohort's question)."""

    intake = {int(year) for year in _INTAKE_YEAR.findall(question)}
    if intake:
        return _single(intake)
    years = {int(year) for year in _COHORT_YEAR.findall(question)}
    years |= {_COHORT_BASE_YEAR + int(number) for number in _COHORT_NUMBER.findall(question)}
    return _single(years)


# In a title: "khóa 16", "khóa K22", or a bare two-digit "K15" ("khóa luận tốt nghiệp K15
# ngành ..."). A bare K needs two digits and a word boundary, so "HK2" is not a cohort.
_DOCUMENT_COHORT = re.compile(rf"\b(?:{_KHOA}\s+K?\s?(\d{{1,2}})|K(\d{{2}}))\b", re.IGNORECASE)


def document_cohort(source: str) -> int | None:
    """The intake year a document's title is for ("khóa 16" -> 2020, "K15" -> 2019), or None.

    "K15" was not read (25/11/2026): "Thông báo kế hoạch thực hiện khóa luận tốt nghiệp K15
    ngành Công nghệ thực phẩm" counted as cohort-neutral, so a K18 question got K15's plan
    presented as K18's."""

    title = source_title(source)
    years = {int(year) for year in _COHORT_YEAR.findall(title)}
    years |= {
        _COHORT_BASE_YEAR + int(khoa or bare) for khoa, bare in _DOCUMENT_COHORT.findall(title)
    }
    return _single(years)


def check_cohort_scope(questions: Sequence[str], chunks: Sequence[RetrievedChunk]) -> CohortScope:
    """`questions`: the turn's standalone questions. Sub-queries of one message usually share
    the cohort ("khóa 18 học trước môn gì? mấy tín chỉ?" is two of them); sub-queries naming
    different cohorts (a comparison) are left alone."""

    kept = list(chunks)
    cohorts = [document_cohort(chunk.source) for chunk in kept]
    available = tuple(sorted({cohort for cohort in cohorts if cohort is not None}))
    if not questions or not available:
        return CohortScope(chunks=kept, available=available)
    named = {cohort for cohort in map(asked_cohort, questions) if cohort is not None}
    if len(named) > 1:
        return CohortScope(chunks=kept, available=available)
    if not named:
        return CohortScope(chunks=kept, notice=_unnamed_notice(available), available=available)
    asked = named.pop()
    if asked in available:
        matching = [c for c, cohort in zip(kept, cohorts, strict=True) if cohort in (None, asked)]
        return CohortScope(
            chunks=matching, asked=asked, available=available, dropped=len(kept) - len(matching)
        )
    # Grouped by document, in the order retrieval ranked them, so each one reads as a whole.
    order = {source: rank for rank, source in enumerate(dict.fromkeys(c.source for c in kept))}
    grouped = sorted(kept, key=lambda chunk: order[chunk.source])
    return CohortScope(
        chunks=grouped, notice=_missing_notice(asked, available), asked=asked, available=available
    )


def _labels(cohorts: Sequence[int]) -> str:
    return ", ".join(cohort_label(cohort) for cohort in cohorts)


def _missing_notice(asked: int, available: Sequence[int]) -> str:
    wanted = cohort_label(asked)
    return (
        "\n\n## PHẠM VI TÀI LIỆU (bắt buộc)\n"
        f"Người dùng hỏi {wanted} nhưng KHÔNG có tài liệu của khóa này; tài liệu tìm được "
        f"thuộc: {_labels(available)}.\n"
        f"- Câu đầu: nói rõ tài liệu mình tra cứu được chưa có thông tin của {wanted}, chỉ có "
        f"của {_labels(available)} (thiếu là ở tài liệu tra cứu được, không nói Nhà trường "
        "không có).\n"
        "- Sau đó trình bày thông tin được hỏi từ những tài liệu CÓ nó, TÁCH theo từng tài liệu, "
        "mỗi phần mở đầu bằng khóa của tài liệu đó và chỉ trích dẫn [n] của chính tài liệu đó; "
        "không trộn mã, số liệu giữa các tài liệu; không gọi đó là "
        f"thông tin của {wanted}.\n"
        f"{ONLY_WHAT_WAS_ASKED}\n"
        f"- Cuối: gợi ý xác nhận với Phòng Đào tạo cho {wanted}."
    )


def _unnamed_notice(available: Sequence[int]) -> str:
    return (
        "\n\n## PHẠM VI TÀI LIỆU (bắt buộc)\n"
        f"Câu hỏi không nêu khóa; tài liệu tìm được thuộc: {_labels(available)}. Ghi rõ khóa "
        "của từng thông tin đưa ra, không trộn mã, số liệu giữa các khóa.\n"
        f"{ONLY_WHAT_WAS_ASKED}"
    )
