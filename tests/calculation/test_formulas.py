from decimal import Decimal

import pytest

from app.calculation import formulas
from app.calculation.formulas import (
    course_score,
    fmt,
    grade_band,
    grade_conversion,
    round_half_up,
)
from app.calculation.result import CalculationInputError

X = formulas.TIMES


def _fields(error: CalculationInputError) -> set[str]:
    return {item.field for item in error.errors}


# --- rounding and display ---------------------------------------------------


@pytest.mark.parametrize(
    ("value", "places", "expected"),
    [
        ("7.45", 1, "7.5"),
        ("7.449", 1, "7.4"),
        ("2.345", 2, "2.35"),
        ("2.344", 2, "2.34"),
        ("8.95", 1, "9.0"),
    ],
)
def test_round_half_up_is_not_bankers_rounding(value: str, places: int, expected: str) -> None:
    assert round_half_up(Decimal(value), places) == Decimal(expected)


def test_decimal_inputs_avoid_float_drift() -> None:
    result = course_score({"tbtx": 0.1, "gk": 0.2, "ck": 0.3, "tclt": 1, "tcth": 0})
    # 0.2*0.1 + 0.3*0.2 + 0.5*0.3 = 0.23 exactly (float math would give 0.22999...) -> 0.2
    assert result.steps[1].substituted.endswith("= 0.02 + 0.06 + 0.15 = 0.23 → 0.2")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("6.95", "6.95"), ("8", "8"), ("8.50", "8.5"), ("7.46666", "≈ 7.47"), ("8.333333", "≈ 8.33")],
)
def test_fmt_shows_at_most_two_decimals(value: str, expected: str) -> None:
    assert fmt(Decimal(value)) == expected


# --- grade scale ------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "letter", "gp4"),
    [
        ("10", "A+", "4.0"),
        ("9.0", "A+", "4.0"),
        ("8.95", "A+", "4.0"),
        ("8.94", "A", "3.8"),
        ("8.5", "A", "3.8"),
        ("8.45", "A", "3.8"),
        ("8.44", "B+", "3.5"),
        ("8.0", "B+", "3.5"),
        ("7.95", "B+", "3.5"),
        ("7.0", "B", "3.0"),
        ("6.95", "B", "3.0"),
        ("6.0", "C+", "2.5"),
        ("5.5", "C", "2.0"),
        ("5.45", "C", "2.0"),
        ("5.0", "D+", "1.5"),
        ("4.0", "D", "1.0"),
        ("3.95", "D", "1.0"),
        ("3.94", "F", "0.0"),
        ("0", "F", "0.0"),
    ],
)
def test_grade_conversion_rounds_then_looks_up(score: str, letter: str, gp4: str) -> None:
    result = grade_conversion({"score10": score})
    outputs = dict(result.outputs)
    assert outputs["Điểm chữ"] == letter
    assert outputs["Thang 4"] == gp4


def test_grade_scale_has_no_gaps_on_rounded_scores() -> None:
    for tenth in range(0, 101):
        score = Decimal(tenth) / 10
        band = grade_band(score)
        assert band.min_score <= score


@pytest.mark.parametrize("bad", ["11", "-0.5", "abc", "8.555", True, None])
def test_grade_conversion_rejects_bad_scores(bad: object) -> None:
    with pytest.raises(CalculationInputError) as error:
        grade_conversion({"score10": bad})
    assert _fields(error.value) == {"score10"}


# --- course score -----------------------------------------------------------


def test_course_score_spec_example() -> None:
    result = course_score({"tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8], "tclt": 2, "tcth": 1})

    assert dict(result.outputs) == {"ĐTKHP": "7.5", "Điểm chữ": "B", "Thang 4": "3.0"}
    labels = [step.label for step in result.steps]
    assert labels == [
        "Điểm thường xuyên",
        "Điểm lý thuyết",
        "Điểm thực hành",
        "Điểm tổng kết học phần",
    ]
    # The band lookup is the result line, not a repeated step.
    assert result.summary == (
        "ĐTKHP **7.5** thuộc khoảng [7.0; 8.0) → điểm chữ **B** → thang 4 **3.0**"
    )
    assert result.steps[0].substituted == "TBtx = 8"
    # Each component is rounded to 0.1 before the next step uses it.
    assert result.steps[1].substituted == (
        f"ĐLT = 0.2 {X} 8.0 + 0.3 {X} 7 + 0.5 {X} 6.5 = 1.6 + 2.1 + 3.25 = 6.95 → 7.0"
    )
    assert result.steps[2].substituted == "ĐTH = (9 + 8) / 2 = 17 / 2 = 8.5"
    assert result.steps[3].substituted == (
        f"ĐTKHP = (7.0 {X} 2 + 8.5 {X} 1) / (2 + 1) = 22.5 / 3 = 7.5"
    )


def test_tx_columns_are_averaged_and_every_component_rounded() -> None:
    result = course_score(
        {"tbtx": [8, 7, 7], "gk": 7, "ck": 6.5, "th": [9, 8, 8], "tclt": 2, "tcth": 1}
    )
    steps = [step.substituted for step in result.steps]
    assert steps[0] == "TBtx = (8 + 7 + 7) / 3 = 22 / 3 ≈ 7.33 → 7.3"
    assert steps[1] == (
        f"ĐLT = 0.2 {X} 7.3 + 0.3 {X} 7 + 0.5 {X} 6.5 = 1.46 + 2.1 + 3.25 = 6.81 → 6.8"
    )
    assert steps[2] == "ĐTH = (9 + 8 + 8) / 3 = 25 / 3 ≈ 8.33 → 8.3"
    assert steps[3] == f"ĐTKHP = (6.8 {X} 2 + 8.3 {X} 1) / (2 + 1) = 21.9 / 3 = 7.3"
    assert dict(result.outputs) == {"ĐTKHP": "7.3", "Điểm chữ": "B", "Thang 4": "3.0"}


def test_a_given_tbtx_is_rounded_too() -> None:
    result = course_score({"tbtx": "7.35", "gk": 7, "ck": 7, "tclt": 1, "tcth": 0})
    assert result.steps[0].substituted == "TBtx = 7.35 → 7.4"


def test_bad_tx_columns_are_reported() -> None:
    with pytest.raises(CalculationInputError) as error:
        course_score({"tbtx": [8, 12], "gk": 7, "ck": 7, "tclt": 1, "tcth": 0})
    assert _fields(error.value) == {"tbtx"}
    with pytest.raises(CalculationInputError):
        course_score({"tbtx": [8] * 21, "gk": 7, "ck": 7, "tclt": 1, "tcth": 0})


def test_rounding_step_never_looks_wrong() -> None:
    # Mean 22.34 / 3 = 7.44667: shown as 7.45 it would visibly "round" to 7.5,
    # but the real value rounds to 7.4 - so the step shows 4 decimals instead.
    result = course_score({"th": ["7.44", "7.45", "7.45"], "tclt": 0, "tcth": 1})
    assert dict(result.outputs)["ĐTKHP"] == "7.4"
    assert (
        result.steps[0].substituted == "ĐTH = (7.44 + 7.45 + 7.45) / 3 = 22.34 / 3 ≈ 7.4467 → 7.4"
    )


def test_before_rounding_falls_back_to_four_decimals() -> None:
    assert formulas.before_rounding(Decimal("7.449"), Decimal("7.4"), 1) == "7.449"
    assert formulas.before_rounding(Decimal("7.44667"), Decimal("7.4"), 1) == "≈ 7.4467"
    assert formulas.before_rounding(Decimal("7.4667"), Decimal("7.5"), 1) == "≈ 7.47"
    assert formulas.before_rounding(Decimal("7.45"), Decimal("7.5"), 1) == "7.45"


def test_theory_only_course_skips_practice() -> None:
    result = course_score({"tbtx": 8, "gk": 8, "ck": 8, "tclt": 3, "tcth": 0})
    assert dict(result.outputs)["ĐTKHP"] == "8.0"
    assert "chỉ có lý thuyết" in result.steps[-1].substituted
    assert all(step.label != "Điểm thực hành" for step in result.steps)


def test_practice_only_course_does_not_need_theory_scores() -> None:
    result = course_score({"th": [9, 10, 8], "tclt": 0, "tcth": 2})
    assert dict(result.outputs)["ĐTKHP"] == "9.0"
    assert all(step.label != "Điểm lý thuyết" for step in result.steps)


def test_zero_total_credits_is_rejected() -> None:
    with pytest.raises(CalculationInputError) as error:
        course_score({"tclt": 0, "tcth": 0})
    assert _fields(error.value) == {"tclt"}


def test_practice_scores_required_only_when_practice_credits() -> None:
    with pytest.raises(CalculationInputError) as error:
        course_score({"tbtx": 8, "gk": 7, "ck": 6.5, "tclt": 2, "tcth": 1})
    assert _fields(error.value) == {"th"}


def test_all_errors_are_reported_at_once() -> None:
    with pytest.raises(CalculationInputError) as error:
        course_score({"tbtx": 11, "ck": "x", "th": [9, 12], "tclt": "1.5", "tcth": 1})
    # tclt is invalid, so missing theory scores (gk) are not reported - only the bad ones.
    assert _fields(error.value) == {"tbtx", "ck", "th", "tclt"}
    reasons = {item.field: item.reason for item in error.value.errors}
    assert "bạn nhập 11" in reasons["tbtx"]


@pytest.mark.parametrize("credits", [-1, 11, "1.5", True])
def test_bad_component_credits(credits: object) -> None:
    with pytest.raises(CalculationInputError) as error:
        course_score({"tbtx": 8, "gk": 8, "ck": 8, "tclt": credits, "tcth": 0})
    assert "tclt" in _fields(error.value)


def test_practice_score_count_is_capped() -> None:
    with pytest.raises(CalculationInputError):
        course_score({"th": [8] * 21, "tclt": 0, "tcth": 1})


def test_comma_decimal_input_is_accepted() -> None:
    result = course_score({"tbtx": "8,5", "gk": "7", "ck": "6,5", "tclt": 1, "tcth": 0})
    assert result.steps[0].value == Decimal("8.5")
    assert result.steps[1].substituted.endswith("= 7.05 → 7.1")


# --- GPA --------------------------------------------------------------------


def test_gpa_counts_failed_courses_and_letter_grades() -> None:
    result = formulas.calculate(
        "gpa",
        {
            "courses": [
                {"name": "Toán", "credits": 3, "score": 8.5},
                {"name": "Lý", "credits": 2, "score": "b+"},
                {"credits": 4, "score": "3.9"},
            ]
        },
    )
    # (3.8*3 + 3.5*2 + 0*4) / 9 = 18.4 / 9 = 2.0444...
    assert dict(result.outputs) == {"GPA": "2.04", "Tổng tín chỉ": "9"}
    assert result.steps[0].substituted == f"8.5 → A → 3.8; 3.8 {X} 3 = 11.4"
    assert result.steps[2].label == "Môn 3"
    assert result.steps[2].substituted.startswith("3.9 → F → 0.0")


def test_gpa_rounds_course_score_before_lookup() -> None:
    result = formulas.calculate("gpa", {"courses": [{"credits": 2, "score": "8.45"}]})
    assert result.steps[0].substituted.startswith("8.45 → 8.5 → A → 3.8")
    assert dict(result.outputs)["GPA"] == "3.80"


def test_gpa_rounds_half_up_to_two_places() -> None:
    # (4.0*1 + 3.5*1 + 3.0*2) / 4 = 13.5 / 4 = 3.375 -> 3.38 (banker's would give 3.38 too;
    # 2.345-style ties are covered by test_round_half_up_is_not_bankers_rounding).
    result = formulas.calculate(
        "gpa",
        {
            "courses": [
                {"credits": 1, "score": "A+"},
                {"credits": 1, "score": "B+"},
                {"credits": 2, "score": "B"},
            ]
        },
    )
    assert dict(result.outputs)["GPA"] == "3.38"


@pytest.mark.parametrize(
    ("courses", "fragment"),
    [
        (None, "còn thiếu"),
        ([], "ít nhất một môn"),
        ([{"credits": 3, "score": 8}] * 31, "tối đa 30"),
        ([{"credits": 0, "score": 8}], "dòng 1"),
        ([{"credits": 3, "score": "G"}], "dòng 1"),
        ([{"credits": 3, "score": 8}, {"credits": 2.5, "score": 8}], "dòng 2"),
    ],
)
def test_gpa_rejects_bad_tables(courses: object, fragment: str) -> None:
    with pytest.raises(CalculationInputError) as error:
        formulas.calculate("gpa", {"courses": courses} if courses is not None else {})
    assert any(fragment in item.reason for item in error.value.errors)


# --- params -----------------------------------------------------------------


def test_missing_params_follow_display_order() -> None:
    names = [spec.name for spec in formulas.missing_params("course_score", {})]
    assert names == ["tclt", "tcth", "tbtx", "gk", "ck", "th"]


def test_practice_scores_not_asked_when_no_practice_credits() -> None:
    names = [spec.name for spec in formulas.missing_params("course_score", {"tclt": 2, "tcth": 0})]
    assert names == ["tbtx", "gk", "ck"]


def test_theory_scores_not_asked_when_no_theory_credits() -> None:
    params = {"tclt": 0, "tcth": 1, "th": [8]}
    assert formulas.missing_params("course_score", params) == []


def test_every_formula_param_spec_is_well_formed() -> None:
    for formula in formulas.FORMULAS.values():
        for spec in formula.params:
            assert len(spec.tab_label) <= 24
            assert formulas.param_spec(formula.formula_id, spec.name) is spec
            if spec.kind in {"number", "number_list"}:
                assert spec.min is not None and spec.max is not None and spec.step is not None
