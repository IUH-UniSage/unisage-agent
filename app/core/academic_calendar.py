"""The current term from today's date, for the `<current_time>` prompt block.

IUH has three terms per academic year. The boundaries below are the school-wide
provisional calendar (2026-2027): faculties may start or end a term on other dates,
so every prompt that shows the term also tells the model to say so to the student.
"""

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


@dataclass(frozen=True)
class CurrentTerm:
    """The term today falls in, or - during a break - the next one (`in_break`)."""

    name: str
    academic_year: str
    period: str
    in_break: bool = False


def current_term(today: date) -> CurrentTerm:
    start_year = today.year if (today.month, today.day) >= _YEAR_START else today.year - 1
    for span in TERMS:
        if today < _on(span.start, start_year):
            return _term(span, start_year, in_break=True)
        if today <= _on(span.end, start_year):
            return _term(span, start_year)
    return _term(TERMS[0], start_year + 1, in_break=True)


def describe_current_term(today: date) -> str:
    term = current_term(today)
    label = f"{term.name} năm học {term.academic_year} (dự kiến {term.period})"
    return f"đang nghỉ giữa các học kỳ; học kỳ tới: {label}" if term.in_break else label


def _term(span: TermSpan, start_year: int, *, in_break: bool = False) -> CurrentTerm:
    return CurrentTerm(
        name=span.name,
        academic_year=f"{start_year}-{start_year + 1}",
        period=span.period.format(y0=start_year, y1=start_year + 1),
        in_break=in_break,
    )


def _on(month_day: tuple[int, int], start_year: int) -> date:
    """The date of `month_day` inside the academic year that starts in `start_year`."""

    month, day = month_day
    year = start_year if (month, day) >= _YEAR_START else start_year + 1
    return date(year, month, day)
