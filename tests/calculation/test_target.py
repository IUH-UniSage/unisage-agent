"""Target questions on the built-in course score (SPEC-calc-engine.md §solver)."""

from decimal import Decimal

from app.calculation.formulas import TIMES, course_score
from app.calculation.render import render_markdown
from app.calculation.solver import Domain
from app.calculation.target import Target, Unknown, grade_goal, solve_one, solve_two

D = Decimal
# The student's course from the bug report: CK must be 10 (9.75+) for A+, not 9.4.
KNOWN = {"tclt": 3, "tcth": 1, "tbtx": [9, 8, 7], "gk": 9, "th": [6, 9, 9]}
CK = Unknown("ck", "CK", Domain(D(0), D(10), D("0.5")), half_point=True)
GK = Unknown("gk", "GK", Domain(D(0), D(10), D("0.5")), half_point=True)


def _target(letter: str) -> Target:
    goal = grade_goal(letter)
    assert goal is not None
    return Target("ĐTKHP", goal, letter)


def test_final_exam_needed_for_a_plus_respects_every_rounding_step() -> None:
    result = solve_one(course_score, KNOWN, CK, _target("A+"))
    assert result.primary_value == D(10)
    assert result.summary == (
        "Cần CK tối thiểu **10.0** (điểm từ 9.75 trở lên được làm tròn thành 10.0)"
        " để ĐTKHP ≥ 9.0 (A+)."
    )
    text = render_markdown(result)
    assert f"ĐTKHP = (9.3 {TIMES} 3 + 8.0 {TIMES} 1) / (3 + 1) = 35.9 / 4 ≈ 8.98 → 9.0" in text
    assert "CK = 9.5 → ĐTKHP 8.8 (chưa đạt)" in text
    assert [label for label, _ in result.inputs] == [
        "Tín chỉ lý thuyết",
        "Tín chỉ thực hành",
        "Các cột thường xuyên",
        "Điểm giữa kỳ",
        "Điểm thực hành",
    ]


def test_unreachable_goal_says_so_with_the_best_possible_result() -> None:
    result = solve_one(course_score, {**KNOWN, "gk": 0, "tbtx": 0}, CK, _target("A+"))
    assert result.primary_value is None
    assert result.summary is not None and "chỉ được **5.8**, **không đạt**" in result.summary


def test_goal_already_met_with_zero() -> None:
    result = solve_one(course_score, KNOWN, CK, _target("D"))
    assert result.primary_value == D(0)
    assert result.summary is not None and "đã chắc chắn đạt" in result.summary


def test_two_unknowns_give_the_equal_level_and_a_short_trade_off_table() -> None:
    known = {key: value for key, value in KNOWN.items() if key != "gk"}
    result = solve_two(course_score, known, (GK, CK), _target("A"))
    assert result.primary_value == D(9)
    assert result.table[0] == ("GK", "CK cần tối thiểu")
    assert ("10.0", "8.0") in result.table
    assert all(row[1] != "không đạt được" for row in result.table[1:])
    assert "| GK | CK cần tối thiểu |" in render_markdown(result)
