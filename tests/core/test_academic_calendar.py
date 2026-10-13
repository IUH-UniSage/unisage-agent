from datetime import date

import pytest

from app.core.academic_calendar import current_term, describe_current_term


@pytest.mark.parametrize(
    ("today", "name", "year", "in_break"),
    [
        ("2026-09-07", "Học kỳ I", "2026-2027", False),
        ("2026-10-13", "Học kỳ I", "2026-2027", False),
        ("2027-01-15", "Học kỳ I", "2026-2027", False),
        ("2027-01-16", "Học kỳ II", "2026-2027", True),  # Tết break
        ("2027-02-16", "Học kỳ II", "2026-2027", False),
        ("2027-05-31", "Học kỳ II", "2026-2027", False),
        ("2027-06-01", "Học kỳ III (hè)", "2026-2027", False),
        ("2027-07-15", "Học kỳ III (hè)", "2026-2027", False),
        ("2027-07-16", "Học kỳ I", "2027-2028", True),  # summer break
        ("2026-09-06", "Học kỳ I", "2026-2027", True),
    ],
)
def test_current_term_follows_the_provisional_calendar(
    today: str, name: str, year: str, in_break: bool
) -> None:
    term = current_term(date.fromisoformat(today))
    assert (term.name, term.academic_year, term.in_break) == (name, year, in_break)


def test_description_names_the_period_and_says_when_it_is_a_break() -> None:
    assert describe_current_term(date(2026, 10, 13)) == (
        "Học kỳ I năm học 2026-2027 (dự kiến 07/09/2026 - nửa đầu tháng 01/2027)"
    )
    assert describe_current_term(date(2027, 1, 20)) == (
        "đang nghỉ giữa các học kỳ; học kỳ tới: Học kỳ II năm học 2026-2027"
        " (dự kiến nửa cuối tháng 02/2027 - cuối tháng 05/2027)"
    )
