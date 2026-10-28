from decimal import Decimal

import pytest

from app.calculation import formulas
from app.calculation.formulas import (
    calculate,
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
    result = course_score({"tbtx": 0.1, "gk": 0.5, "ck": 1.5, "tclt": 1, "tcth": 0})
    # 0.2*0.1 + 0.3*0.5 + 0.5*1.5 = 0.92 exactly (float math drifts) -> 0.9
    assert result.steps[1].substituted.endswith("= 0.02 + 0.15 + 0.75 = 0.92 → 0.9")


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


@pytest.mark.parametrize(
    ("score", "expected"),
    [("7.24", "7"), ("7.25", "7.5"), ("7.74", "7.5"), ("7.75", "8"), ("9.9", "10"), ("0.1", "0")],
)
def test_component_scores_round_to_the_half_point(score: str, expected: str) -> None:
    assert formulas.round_to_half_point(Decimal(score)) == Decimal(expected)


def test_component_rounding_is_shown_and_used() -> None:
    result = course_score(
        {"tbtx": [9, 8.3, 7], "gk": 6.3, "ck": 7.8, "th": [6, 9.2, 8.8], "tclt": 3, "tcth": 1}
    )
    assert result.steps[0].label == "Làm tròn điểm thành phần"
    assert result.steps[0].substituted == (
        "TX2 8.3 → 8.5; GK 6.3 → 6.5; CK 7.8 → 8.0; TH2 9.2 → 9.0; TH3 8.8 → 9.0"
    )
    assert result.steps[1].substituted == "TBtx = (9 + 8.5 + 7) / 3 = 24.5 / 3 ≈ 8.17 → 8.2"
    assert dict(result.outputs)["ĐTKHP"] == "7.7"


def test_no_rounding_step_when_scores_are_already_half_points() -> None:
    result = course_score({"tbtx": [8, 7.5], "gk": 7, "ck": 6.5, "tclt": 1, "tcth": 0})
    assert result.steps[0].label == "Điểm thường xuyên"


def test_a_given_tbtx_is_an_average_not_a_component() -> None:
    result = course_score({"tbtx": "7.35", "gk": 7, "ck": 7, "tclt": 1, "tcth": 0})
    assert all(step.label != "Làm tròn điểm thành phần" for step in result.steps)


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


# --- tuition -----------------------------------------------------------------

_TUITION = {"tc_lt": 100, "tc_th": 42, "don_gia_lt": 980000, "don_gia_th": "1.600.000 đ"}


def test_tuition_is_theory_and_practice_credits_times_their_unit_price() -> None:
    result = formulas.calculate("tuition_program", _TUITION)

    assert result.title == "Học phí cả khoá học"
    assert result.outputs == (("Học phí cả khoá học", "165.200.000 đồng"),)
    assert [step.substituted for step in result.steps] == [
        f"100 {X} 980.000 đồng = 98.000.000 đồng",
        f"42 {X} 1.600.000 đồng = 67.200.000 đồng",
        "98.000.000 đồng + 67.200.000 đồng = 165.200.000 đồng",
    ]


def test_tuition_title_and_credit_labels_name_the_period_the_student_asked_for() -> None:
    params = {**_TUITION, "pham_vi": "2 năm đầu"}
    result = formulas.calculate("tuition_year", params)
    assert result.title == "Học phí 2 năm đầu"
    assert result.outputs == (("Học phí 2 năm đầu", "165.200.000 đồng"),)

    spec = formulas.param_spec("tuition_year", "tc_lt")
    assert spec is not None
    assert (
        formulas.label_for(spec, {"pham_vi": "2 năm đầu"}) == "Số tín chỉ lý thuyết của 2 năm đầu"
    )
    assert formulas.label_for(spec, {}) == "Số tín chỉ lý thuyết của một năm học"
    assert formulas.calculate("tuition_year", _TUITION).title == "Học phí một năm học"


@pytest.mark.parametrize("period", [None, "", "   ", 3])
def test_tuition_without_a_usable_period_keeps_the_default_title(period: object) -> None:
    result = formulas.calculate("tuition_semester", {**_TUITION, "pham_vi": period})
    assert result.title == "Học phí một học kỳ"


def test_the_period_is_never_asked() -> None:
    names = {spec.name for spec in formulas.missing_params("tuition_year", {})}
    assert "pham_vi" not in names


def test_tuition_needs_theory_and_practice_credits_separately() -> None:
    missing = formulas.missing_params("tuition_semester", {"don_gia_lt": 1, "don_gia_th": 1})
    assert [spec.name for spec in missing] == ["tc_lt", "tc_th"]


def test_tuition_context_params_are_never_asked_on_their_own() -> None:
    names = {spec.name for spec in formulas.missing_params("tuition_year", {})}
    assert names == {"tc_lt", "tc_th", "don_gia_lt", "don_gia_th"}


def test_only_prices_and_a_programme_s_credits_are_looked_up() -> None:
    def looked_up(formula_id: formulas.FormulaId) -> set[str]:
        return {spec.name for spec in formulas.FORMULAS[formula_id].params if spec.lookup}

    assert looked_up("tuition_semester") == {"don_gia_lt", "don_gia_th"}
    assert looked_up("tuition_year") == {"don_gia_lt", "don_gia_th"}
    assert looked_up("tuition_program") == {"tc_lt", "tc_th", "don_gia_lt", "don_gia_th"}


def test_tuition_rejects_bad_inputs_all_at_once() -> None:
    with pytest.raises(CalculationInputError) as error:
        formulas.calculate(
            "tuition_semester", {"tc_lt": 2.5, "tc_th": 301, "don_gia_lt": -1, "don_gia_th": 1}
        )
    assert _fields(error.value) == {"tc_lt", "tc_th", "don_gia_lt"}


def test_tuition_without_any_credit_is_rejected() -> None:
    with pytest.raises(CalculationInputError) as error:
        formulas.calculate(
            "tuition_semester", {"tc_lt": 0, "tc_th": 0, "don_gia_lt": 1, "don_gia_th": 1}
        )
    assert _fields(error.value) == {"tc_lt"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (980000, Decimal(980000)),
        ("980.000", Decimal(980000)),
        ("1,600,000 đồng", Decimal(1600000)),
        ("980000đ", Decimal(980000)),
    ],
)
def test_parse_money_reads_amounts_written_out(raw: object, expected: Decimal) -> None:
    assert formulas.parse_money(raw, "x", []) == expected


@pytest.mark.parametrize("raw", ["1,6 triệu", "980k", "9.80", ""])
def test_parse_money_refuses_shorthand_instead_of_misreading_it(raw: str) -> None:
    errors: list[formulas.FieldError] = []
    assert formulas.parse_money(raw, "x", errors) is None
    assert [error.field for error in errors] == ["x"]


# --- admission score 2026 ------------------------------------------------------


def test_admission_2026_without_aptitude_result_matches_the_published_example() -> None:
    # d0307: ĐTN 24, ĐHB 27, ưu tiên 0,5 → XT1 25,4; XT2 24,5; ĐXT 25,4.
    result = calculate("admission_score_2026", {"dtn": 24, "dhb": 27, "uu_tien": 0.5})
    assert result.values["xt1"] == Decimal("25.4")
    assert result.values["xt2"] == Decimal("24.5")
    assert result.values["dxt"] == Decimal("25.4")
    assert "xt3" not in result.values
    assert dict(result.outputs)["ĐXT"] == "25.4"
    assert result.version == "2026.1" and "2026" in result.source


def test_admission_2026_converts_the_aptitude_score_to_30_before_anything_else() -> None:
    result = calculate(
        "admission_score_2026", {"dtn": 22, "dgnl": 990, "dtk": 1100, "dhb": 25, "diem_cong": 0.5}
    )
    assert result.values["ddgnl"] == Decimal("27")  # 990 * 30 / 1100
    assert result.values["dk"] == Decimal("27")  # Max(22, 27)
    assert result.values["xt1"] == Decimal("0.7") * 27 + Decimal("0.3") * 25 + Decimal("0.5")
    assert result.values["xt3"] == Decimal("27.5")
    assert result.values["dxt"] == max(result.values["xt1"], Decimal("22.5"), Decimal("27.5"))
    assert result.steps[0].label == "Quy đổi ĐGNL về thang 30"


def test_admission_2026_needs_the_top_score_once_an_aptitude_result_is_given() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("admission_score_2026", {"dtn": 22, "dhb": 25, "dgnl": 900})
    assert _fields(info.value) == {"dtk"}
    # The top score is the inconsistent one (it may have been misread from a document): the
    # student's own result is kept and ĐTK is asked again.
    with pytest.raises(CalculationInputError) as info:
        calculate("admission_score_2026", {"dtn": 22, "dhb": 25, "dgnl": 1200, "dtk": 1100})
    assert _fields(info.value) == {"dtk"}
    with pytest.raises(CalculationInputError) as info:
        calculate("admission_score_2026", {"dtn": 22, "dhb": 25, "dgnl": 1100, "dtk": 30})
    assert _fields(info.value) == {"dtk"}  # 30 is below the 1200-point test's possible top score


def test_admission_2026_rejects_scores_outside_the_30_point_scale() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("admission_score_2026", {"dtn": 31, "dhb": -1})
    assert _fields(info.value) == {"dtn", "dhb"}


# --- transcript score (lớp 10, 11, HK1 lớp 12) -----------------------------------

_TRANSCRIPT = {
    "mon1": "Toán",
    "m1_10": 7.5,
    "m1_11": 8,
    "m1_12": 8.5,
    "mon2": "Hóa",
    "m2_10": 6,
    "m2_11": 6.5,
    "m2_12": 7,
    "mon3": "Lý",
    "m3_10": 4.5,
    "m3_11": 5,
    "m3_12": 5.5,
}


def test_transcript_score_averages_each_subject_then_sums() -> None:
    result = calculate("transcript_score_3hk", _TRANSCRIPT)
    assert result.values["mon1"] == Decimal("8")
    assert result.values["mon2"] == Decimal("6.5")
    assert result.values["mon3"] == Decimal("5")
    assert result.values["tong"] == Decimal("19.5")
    assert result.summary is not None and "đủ điều kiện" in result.summary
    assert result.inputs[0] == ("Môn 1 (Toán)", "7.5 - 8 - 8.5")


def test_transcript_score_below_the_threshold_says_so() -> None:
    result = calculate("transcript_score_3hk", {**_TRANSCRIPT, "m3_12": 5})
    assert result.values["tong"] == Decimal("19.5") - Decimal("0.5") / 3
    assert result.summary is not None and "chưa đạt" in result.summary


def test_transcript_score_lists_every_missing_term() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("transcript_score_3hk", {"m1_10": 7, "m1_11": 8})
    assert _fields(info.value) == {"m1_12", "m2_10", "m2_11", "m2_12", "m3_10", "m3_11", "m3_12"}


# --- weighted score ------------------------------------------------------------


def test_weighted_score_follows_the_syllabus_weights() -> None:
    # d0312: 10/10/30/50 with 8, 9, 7, 6 → 6,8.
    result = calculate("weighted_score", {"trong_so": [10, 10, 30, 50], "diem": [8, 9, 7, 6]})
    assert result.values["diem_hp"] == Decimal("6.8")
    assert dict(result.outputs)["Điểm chữ"] == "C+"


def test_weighted_score_accepts_fractions_and_rounds_components_when_asked() -> None:
    result = calculate(
        "weighted_score",
        {"trong_so": [0.2, 0.3, 0.5], "diem": [9, 7, 9.2], "lam_tron_thanh_phan": 0.5},
    )
    assert result.steps[0].substituted == "TP3 9.2 → 9"
    assert result.values["diem_hp"] == Decimal("8.4")


def test_weighted_score_rejects_weights_that_do_not_sum_to_100() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("weighted_score", {"trong_so": [10, 30, 50], "diem": [8, 7, 6]})
    assert _fields(info.value) == {"trong_so"}


def test_weighted_score_needs_one_score_per_weight() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("weighted_score", {"trong_so": [50, 50], "diem": [8, None]})
    assert [error.reason for error in info.value.errors] == ["còn thiếu điểm thành phần thứ 2"]


def test_every_registered_formula_declares_version_source_and_outputs() -> None:
    for formula in formulas.FORMULAS.values():
        assert formula.version and formula.source and formula.outputs
        assert formulas.calculate is not None


# --- date span / time span ---------------------------------------------------------


def test_date_span_counts_calendar_days_with_and_without_both_ends() -> None:
    result = calculate("date_span", {"tu_ngay": "13/08/2026", "den_ngay": "20/08/2026"})
    assert result.values == {"so_ngay": Decimal(7), "ca_hai_dau": Decimal(8)}
    assert result.summary is not None and "**8 ngày**" in result.summary


def test_date_span_adds_a_validity_period_to_the_start_date() -> None:
    # d0319: cấp 20/5/2024, hạn 2 năm, nộp 10/6/2026 → quá hạn 21 ngày.
    result = calculate(
        "date_span", {"tu_ngay": "20/5/2024", "den_ngay": "10/06/2026", "cong_them_nam": 2}
    )
    assert result.values["so_ngay"] == Decimal(21)
    assert result.steps[0].substituted == "20/05/2024 + 2 năm = 20/05/2026"


def test_date_span_rejects_bad_dates_and_reversed_order() -> None:
    with pytest.raises(CalculationInputError) as info:
        calculate("date_span", {"tu_ngay": "31/02/2026", "den_ngay": "hôm qua"})
    assert _fields(info.value) == {"tu_ngay", "den_ngay"}
    with pytest.raises(CalculationInputError) as info:
        calculate("date_span", {"tu_ngay": "20/08/2026", "den_ngay": "13/08/2026"})
    assert _fields(info.value) == {"den_ngay"}


def test_time_span_sums_clock_ranges_in_hours_and_minutes() -> None:
    # d0332: 06h30-19h40 x2 và 06h30-17h40 → 37 giờ 30 phút.
    ranges = "06h30 \u2013 19h40; 06:30-19:40; 06:30 đến 17:40"
    result = calculate("time_span", {"khoang_gio": ranges})
    assert result.values == {"tong_gio": Decimal("37.5"), "tong_phut": Decimal(2250)}
    assert result.summary == "Tổng thời gian **37 giờ 30 phút** (= **37.5 giờ**)"
    assert calculate("time_span", {"khoang_gio": "22:00-02:00"}).values["tong_phut"] == Decimal(240)
    with pytest.raises(CalculationInputError):
        calculate("time_span", {"khoang_gio": "sáng đến chiều"})


def test_admission_2026_asks_every_input_the_student_must_supply() -> None:
    """ĐGNL, ưu tiên and điểm cộng change the result, so the panel asks them explicitly
    ("0 nếu không có") instead of assuming 0; ĐTK only once a ĐGNL result (> 0) is given."""

    from app.calculation.formulas import missing_params

    assert [spec.name for spec in missing_params("admission_score_2026", {})] == [
        "dtn",
        "dhb",
        "dgnl",
        "uu_tien",
        "diem_cong",
    ]
    assert [
        spec.name
        for spec in missing_params(
            "admission_score_2026", {"dtn": 24, "dhb": 27, "dgnl": 0, "uu_tien": 0, "diem_cong": 0}
        )
    ] == []
    assert [
        spec.name
        for spec in missing_params(
            "admission_score_2026",
            {"dtn": 24, "dhb": 27, "dgnl": 900, "uu_tien": 0, "diem_cong": 0},
        )
    ] == ["dtk"]
    # ĐGNL 0 means "did not sit the test": no conversion, ĐK = ĐTN.
    result = calculate(
        "admission_score_2026", {"dtn": 24, "dhb": 27, "dgnl": 0, "uu_tien": 0.5, "diem_cong": 0}
    )
    assert result.values["dk"] == Decimal(24) and "ddgnl" not in result.values
