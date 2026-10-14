from datetime import date

import pytest

from app.core.academic_calendar import (
    cohort_label,
    current_term,
    describe_current_term,
    describe_student_term,
    intake_options,
    parse_intake_year,
    student_term,
)

_TODAY = date(2026, 10, 13)  # Học kỳ I 2026-2027


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


@pytest.mark.parametrize(
    ("today", "intake", "number"),
    [
        ("2026-10-13", 2022, 9),  # HK I of the 5th year
        ("2026-10-13", 2026, 1),
        ("2027-03-01", 2022, 10),  # HK II
        ("2027-01-20", 2022, 10),  # Tết break: the next term
        ("2027-06-10", 2022, 10),  # summer: not numbered, the term just finished
        ("2027-08-01", 2022, 11),  # summer break: the next HK I
        ("2026-10-13", 2027, None),  # not started yet
    ],
)
def test_student_term_counts_two_terms_a_year_from_the_intake(
    today: str, intake: int, number: int | None
) -> None:
    assert student_term(date.fromisoformat(today), intake) == number


def test_student_term_description_names_the_number_and_the_cohort() -> None:
    assert describe_student_term(_TODAY, 2022) == "học kỳ 9 của khoá học (nhập học 2022, khóa 18)"
    assert describe_student_term(date(2027, 6, 10), 2022) == (
        "học kỳ hè sau học kỳ 10 của khoá học (nhập học 2022, khóa 18)"
    )
    unknown = describe_student_term(_TODAY, None)
    assert unknown.startswith("CHƯA BIẾT")
    assert "nhập học 2022 → học kỳ 9" in unknown


def test_intake_options_list_the_latest_intakes_with_their_term_now() -> None:
    assert intake_options(_TODAY) == [
        (2026, 1),
        (2025, 3),
        (2024, 5),
        (2023, 7),
        (2022, 9),
        (2021, 11),
    ]


@pytest.mark.parametrize(
    ("value", "intake"),
    [
        ("2022", 2022),
        (2022, 2022),
        ("2022-2023", 2022),
        ("Nhập học 2022 (đang học kỳ 9)", 2022),
        ("K20", 2024),  # IUH: khóa N = 2004 + N
        ("khóa 18", 2022),
        ("kỳ 9", 2022),
        ("học kỳ 9", 2022),
        ("HK1", 2026),
        ("học kỳ 10", 2021),  # an even term in HK I: the one just finished
        ("abc", None),
        ("K99", None),
        (None, None),
    ],
)
def test_parse_intake_year_reads_a_year_a_cohort_or_a_term_number(
    value: object, intake: int | None
) -> None:
    assert parse_intake_year(value, _TODAY) == intake


def test_cohort_label_counts_intakes_from_2004() -> None:
    assert cohort_label(2022) == "K18"
    assert cohort_label(2024) == "K20"
