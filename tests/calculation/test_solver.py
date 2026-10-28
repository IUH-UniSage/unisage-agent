"""Inverse calculation (SPEC-calc-engine, tính ngược): the unknown is searched on its
grid with the forward formula, so every rounding rule applies; the result shows the
check with the adjacent value."""

from decimal import Decimal

import pytest

from app.calculation.formulas import FormulaId, param_spec
from app.calculation.render import render_markdown
from app.calculation.result import CalculationInputError
from app.calculation.solver import (
    Target,
    Unknown,
    parse_target,
    parse_unknown,
    search_step,
    solve,
    solved_result,
    with_value,
)


def _target(formula_id: FormulaId, output: str, op: str, value: object) -> Target:
    target = parse_target(formula_id, {"output": output, "op": op, "value": value})
    assert isinstance(target, Target), target
    return target


# --- parsing ------------------------------------------------------------------


def test_parse_unknown_reads_a_scalar_and_a_list_position() -> None:
    assert parse_unknown("ck") == Unknown("ck")
    assert parse_unknown(" Diem[4] ") == Unknown("diem", 4)
    assert parse_unknown("diem[]") is None
    assert parse_unknown(12) is None


def test_parse_target_reads_letter_grades_on_the_matching_scale() -> None:
    assert _target("course_score", "dtkhp", ">=", "A+").value == Decimal("9.0")
    assert _target("course_score", "thang4", ">=", "b+").value == Decimal("3.5")
    assert _target("admission_score_2026", "xt1", ">=", "24") == Target("xt1", ">=", Decimal(24))


def test_parse_target_rejects_unknown_outputs_operators_and_values() -> None:
    assert parse_target("gpa", {"output": "xt1", "op": ">=", "value": 3}).field == "target"  # type: ignore[union-attr]
    assert parse_target("gpa", {"output": "gpa", "op": ">", "value": 3}).field == "target"  # type: ignore[union-attr]
    assert parse_target("gpa", {"output": "gpa", "op": ">=", "value": "nhiều"}).field == "target"  # type: ignore[union-attr]
    assert parse_target("gpa", "gpa >= 3").field == "target"  # type: ignore[union-attr]


def test_with_value_fills_a_list_position_and_appends_one_past_the_end() -> None:
    assert with_value({"diem": [8, 8, 6]}, Unknown("diem", 4), Decimal(7)) == {"diem": [8, 8, 6, 7]}
    assert with_value({"diem": [8, 8, 6]}, Unknown("diem", 2), Decimal(9)) == {"diem": [8, 9, 6]}
    assert with_value({}, Unknown("ck"), Decimal(5)) == {"ck": Decimal(5)}


# --- the eval's inverse questions --------------------------------------------------


def test_admission_2026_minimum_transcript_total_for_xt1() -> None:
    # d0308: ĐTN 23, ưu tiên 0,25, XT1 ≥ 24 → ĐHB ≥ 25,5.
    solution = solve(
        "admission_score_2026",
        {"dtn": 23, "uu_tien": 0.25},
        Unknown("dhb"),
        _target("admission_score_2026", "xt1", ">=", 24),
    )
    assert solution.value == Decimal("25.5")
    assert solution.output == Decimal("24")
    assert solution.checked is not None and solution.checked[0] == Decimal("25.49")
    assert not solution.maximum


def test_transcript_minimum_first_term_grade_12_score() -> None:
    # d0309: Toán 8,0, Hóa 6,5, Lý lớp 10/11 = 4,5/5,0, ngưỡng 19,5 → HK1 lớp 12 ≥ 5,5.
    params = {
        "m1_10": 7.5, "m1_11": 8, "m1_12": 8.5,
        "m2_10": 6, "m2_11": 6.5, "m2_12": 7,
        "mon3": "Lý", "m3_10": 4.5, "m3_11": 5,
    }  # fmt: skip
    result = solved_result(
        "transcript_score_3hk",
        params,
        Unknown("m3_12"),
        _target("transcript_score_3hk", "tong", ">=", 19.5),
    )
    assert result.values["nghiem"] == Decimal("5.5")
    assert result.title == "Môn 3 (Lý): TB học kỳ 1 lớp 12 tối thiểu để tổng điểm 3 môn ≥ 19.5"
    assert result.steps[0].label == "Giá trị cần tìm"
    assert result.steps[-1].label == "Kiểm tra"
    assert "5.49 → ≈ 19.4967 (không đạt)" in result.steps[-1].substituted


def test_weighted_score_minimum_final_component() -> None:
    # d0313: 10/10/30/50 with 8, 8, 6 → final ≥ 7,1 gives 6,95 → 7,0 (rounded to 0.1).
    solution = solve(
        "weighted_score",
        {"trong_so": [10, 10, 30, 50], "diem": [8, 8, 6]},
        Unknown("diem", 4),
        _target("weighted_score", "diem_hp", ">=", 7.0),
    )
    assert solution.value == Decimal("7.1")
    assert solution.output == Decimal("7.0")


def test_weighted_score_searches_in_halves_when_components_are_rounded_to_half() -> None:
    # d0311: 20/30/50 with 9, 7, exam rounded to 0.5, A (≥ 8,5) → exam ≥ 9,5.
    params = {"trong_so": [20, 30, 50], "diem": [9, 7], "lam_tron_thanh_phan": 0.5}
    spec = param_spec("weighted_score", "diem")
    assert spec is not None and search_step("weighted_score", spec, params) == Decimal("0.5")
    solution = solve(
        "weighted_score",
        params,
        Unknown("diem", 3),
        _target("weighted_score", "diem_hp", ">=", "A"),
    )
    assert solution.value == Decimal("9.5")
    assert solution.checked == (Decimal("9.0"), Decimal("8.4"))


def test_course_score_final_exam_for_a_plus_is_searched_in_halves() -> None:
    params = {"tclt": 3, "tcth": 1, "tbtx": [9, 8, 7], "gk": 9, "th": [6, 9, 9]}
    result = solved_result(
        "course_score", params, Unknown("ck"), _target("course_score", "dtkhp", ">=", "A+")
    )
    assert result.values["nghiem"] == Decimal("10")
    assert "thang 10 = 10 → ĐTKHP 9 (đạt); Điểm cuối kỳ, thang 10 = 9.5 → 8.8 (không đạt)" in (
        result.steps[-1].substituted
    )
    assert result.outputs[0] == ("Điểm cuối kỳ, thang 10", "10")
    assert result.version == "1"


def test_tuition_maximum_credits_under_a_budget_skips_the_refused_zero() -> None:
    # 0 credits is refused by the formula (not a target miss): the search still runs.
    params = {"tc_th": 0, "don_gia_lt": 980000, "don_gia_th": 1600000}
    result = solved_result(
        "tuition_semester",
        params,
        Unknown("tc_lt"),
        _target("tuition_semester", "hoc_phi", "<=", 8_000_000),
    )
    assert result.values["nghiem"] == Decimal(8)
    assert (
        result.title
        == "Số tín chỉ lý thuyết của một học kỳ tối đa để học phí một học kỳ ≤ 8.000.000 đồng"
    )
    assert "TC lý thuyết = 9 → 8.820.000 đồng (không đạt)" in result.steps[-1].substituted


def test_a_falling_parameter_is_solved_for_its_maximum() -> None:
    # ĐĐGNL = 900 * 30 / ĐTK ≥ 25 → ĐTK ≤ 1080; ĐTK below 900 is refused (ĐGNL > ĐTK).
    solution = solve(
        "admission_score_2026",
        {"dtn": 20, "dhb": 24, "dgnl": 900},
        Unknown("dtk"),
        _target("admission_score_2026", "ddgnl", ">=", 25),
    )
    assert solution.value == Decimal("1080")
    assert solution.maximum


def test_an_equality_target_is_met_exactly_or_reported() -> None:
    hit = solve(
        "course_score",
        {"tclt": 3, "tcth": 0, "tbtx": 8, "gk": 7},
        Unknown("ck"),
        _target("course_score", "dtkhp", "==", 7.5),
    )
    assert hit.value == Decimal("7.5")
    miss = solve(
        "course_score",
        {"tclt": 3, "tcth": 0, "tbtx": 8, "gk": 7},
        Unknown("ck"),
        _target("course_score", "dtkhp", "==", 7.55),
    )
    assert miss.value is None


def test_an_unreachable_target_says_so_with_the_boundary_value() -> None:
    result = solved_result(
        "course_score",
        {"tclt": 3, "tcth": 0, "tbtx": 2, "gk": 2},
        Unknown("ck"),
        _target("course_score", "dtkhp", ">=", 9),
    )
    assert "nghiem" not in result.values
    assert result.summary is not None
    assert result.summary.startswith("Không có giá trị Điểm cuối kỳ, thang 10 nào trong [0; 10]")
    assert result.summary.endswith("ĐTKHP chỉ được **6**")
    assert result.warnings[-1] == "Mục tiêu không đạt được với các số liệu đã cho."
    assert "Mục tiêu không đạt được" in render_markdown(result)


def test_errors_of_the_other_params_are_raised_not_hidden() -> None:
    with pytest.raises(CalculationInputError) as info:
        solve(
            "admission_score_2026",
            {"dtn": 40},
            Unknown("dhb"),
            _target("admission_score_2026", "xt1", ">=", 24),
        )
    assert {error.field for error in info.value.errors} == {"dtn"}


def test_the_unknown_must_be_a_numeric_param_of_the_formula() -> None:
    with pytest.raises(CalculationInputError) as info:
        solve("gpa", {}, Unknown("courses"), _target("gpa", "gpa", ">=", 3))
    assert info.value.errors[0].reason == "chỉ tìm được tham số dạng số"
    with pytest.raises(CalculationInputError):
        solve("course_score", {}, Unknown("xyz"), _target("course_score", "dtkhp", ">=", 9))
    with pytest.raises(CalculationInputError):
        solve(
            "weighted_score",
            {"trong_so": [50, 50], "diem": [8]},
            Unknown("diem"),
            _target("weighted_score", "diem_hp", ">=", 8),
        )
