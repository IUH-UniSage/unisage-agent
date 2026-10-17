"""The current term from today's date, and the student's own term number, for the
`<current_time>` prompt block.

IUH has three terms per academic year. The boundaries below are the school-wide
provisional calendar (2026-2027): faculties may start or end a term on other dates,
so every prompt that shows the term also tells the model to say so to the student.

A student's terms are numbered 1, 2, 3... from their intake: two per academic year
(HK I, HK II); the summer term is not counted, so 4 years are 8 terms.
"""

import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TermSpan:
    name: str
    # (month, day) of the first and last day; a span may cross the new year.
    start: tuple[int, int]
    end: tuple[int, int]
    # Shown to the model, with {y0}/{y1} = the academic year's two calendar years.
    period: str


# In academic-year order. The gaps between them are the Tết and summer breaks.
TERMS: tuple[TermSpan, ...] = (
    TermSpan("Học kỳ I", (9, 7), (1, 15), "07/09/{y0} - nửa đầu tháng 01/{y1}"),
    TermSpan("Học kỳ II", (2, 16), (5, 31), "nửa cuối tháng 02/{y1} - cuối tháng 05/{y1}"),
    TermSpan("Học kỳ III (hè)", (6, 1), (7, 15), "đầu tháng 06/{y1} - giữa tháng 07/{y1}"),
)
_YEAR_START = TERMS[0].start
# IUH numbers its intakes from 2004: "khóa 20" entered in 2024-2025.
COHORT_YEAR_OFFSET = 2004
# The confirmed_metadata field holding the intake (known_metadata_fields.json).
INTAKE_FIELD = "khoa_nhap_hoc"
INTAKE_OPTIONS = 6


@dataclass(frozen=True)
class CurrentTerm:
    """The term today falls in, or - during a break - the next one (`in_break`)."""

    name: str
    academic_year: str
    period: str
    in_break: bool = False
    start_year: int = 0
    # Position in TERMS: 0 = HK I, 1 = HK II, 2 = summer.
    index: int = 0


def current_term(today: date) -> CurrentTerm:
    start_year = today.year if (today.month, today.day) >= _YEAR_START else today.year - 1
    for span in TERMS:
        if today < _on(span.start, start_year):
            return _term(span, start_year, in_break=True)
        if today <= _on(span.end, start_year):
            return _term(span, start_year)
    return _term(TERMS[0], start_year + 1, in_break=True)


def student_term(today: date, intake_year: int) -> int | None:
    """The student's term number now - or, during a break, the next one; during the
    summer term, the term just finished. None before the student's first term."""

    term = current_term(today)
    years = term.start_year - intake_year
    number = 2 * years + (1 if term.index == 0 else 2)
    return number if number >= 1 else None


def cohort_label(intake_year: int) -> str:
    """IUH's name for an intake: 2022 → "K18" - never "K2022"."""

    return f"K{intake_year - COHORT_YEAR_OFFSET}"


def describe_student_term(today: date, intake_year: int | None) -> str:
    if intake_year is None:
        options = "; ".join(
            f"nhập học {year} → học kỳ {number}" for year, number in intake_options(today)
        )
        return f"CHƯA BIẾT (chưa có năm nhập học). Bảng tra lúc này: {options}"
    number = student_term(today, intake_year)
    cohort = f"nhập học {intake_year}, khóa {intake_year - COHORT_YEAR_OFFSET}"
    if number is None:
        return f"chưa bắt đầu học kỳ đầu tiên ({cohort})"
    term = current_term(today)
    if term.index == 2 and not term.in_break:
        return f"học kỳ hè sau học kỳ {number} của khoá học ({cohort})"
    if term.in_break:
        return f"đang nghỉ; học kỳ tới là học kỳ {number} của khoá học ({cohort})"
    return f"học kỳ {number} của khoá học ({cohort})"


def intake_options(today: date) -> list[tuple[int, int]]:
    """(intake year, term number now) for the latest intakes - the options a student
    picks from when asked which intake they are in."""

    start_year = current_term(today).start_year
    options = []
    for intake in range(start_year, start_year - INTAKE_OPTIONS, -1):
        number = student_term(today, intake)
        if number is not None:
            options.append((intake, number))
    return options


def parse_intake_year(value: object, today: date) -> int | None:
    """An intake year from what a student typed or picked: a year ("2022",
    "2022-2023"), an IUH cohort ("K20", "khóa 20" → 2024) or a term number ("kỳ 9",
    "học kỳ 9" → the intake that is in term 9 today)."""

    if isinstance(value, int) and not isinstance(value, bool):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        return None
    if match := re.search(r"\b(20\d{2})\b", text):
        year = int(match.group(1))
    elif match := re.fullmatch(r"(?:k|khóa|khoá)\s*(\d{1,2})", text):
        year = COHORT_YEAR_OFFSET + int(match.group(1))
    elif match := re.fullmatch(r"(?:học kỳ|học kì|kỳ|kì|hk)\s*(\d{1,2})", text):
        return _intake_for_term(today, int(match.group(1)))
    else:
        return None
    return year if 2000 <= year <= current_term(today).start_year + 1 else None


def describe_current_term(today: date) -> str:
    term = current_term(today)
    label = f"{term.name} năm học {term.academic_year} (dự kiến {term.period})"
    return f"đang nghỉ giữa các học kỳ; học kỳ tới: {label}" if term.in_break else label


def _intake_for_term(today: date, number: int) -> int | None:
    if number < 1:
        return None
    term = current_term(today)
    first_of_year = 1 if term.index == 0 else 2
    if (number - first_of_year) % 2:
        # "kỳ 9" during HK II: term 9 was last HK I, so the same intake.
        number += 1
    return term.start_year - (number - first_of_year) // 2


def _term(span: TermSpan, start_year: int, *, in_break: bool = False) -> CurrentTerm:
    return CurrentTerm(
        name=span.name,
        academic_year=f"{start_year}-{start_year + 1}",
        period=span.period.format(y0=start_year, y1=start_year + 1),
        in_break=in_break,
        start_year=start_year,
        index=TERMS.index(span),
    )


def _on(month_day: tuple[int, int], start_year: int) -> date:
    """The date of `month_day` inside the academic year that starts in `start_year`."""

    month, day = month_day
    year = start_year if (month, day) >= _YEAR_START else start_year + 1
    return date(year, month, day)
