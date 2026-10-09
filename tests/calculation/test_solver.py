"""Generic target solver (SPEC-calc-engine.md §solver)."""

from decimal import Decimal

from app.calculation.solver import Domain, Goal, solve, solve_pair

D = Decimal
SCORES = Domain(D(0), D(10), D("0.5"))


def test_min_value_meeting_the_goal_and_the_value_just_below() -> None:
    solution = solve(lambda x: x * 2, SCORES, Goal(">=", D(13)), "min")
    assert solution.value == D("6.5")
    assert solution.boundary is not None and solution.boundary.value == D(6)


def test_max_value_for_an_upper_goal() -> None:
    credits = Domain(D(1), D(40), D(1))
    solution = solve(lambda tc: tc * 420000, credits, Goal("<=", D(8_000_000)), "max")
    assert solution.value == D(19)
    assert solution.boundary is not None and solution.boundary.value == D(20)


def test_unreachable_goal_reports_the_best_end_tried() -> None:
    solution = solve(lambda x: x / 2, SCORES, Goal(">=", D(9)), "min")
    assert solution.value is None
    assert solution.at_limit is not None and solution.at_limit.value == D(10)


def test_already_met_has_no_boundary() -> None:
    solution = solve(lambda x: x + 5, SCORES, Goal(">=", D(4)), "min")
    assert solution.value == D(0) and solution.boundary is None


def test_uncomputable_values_are_skipped() -> None:
    solution = solve(lambda x: None if x < 3 else x, SCORES, Goal(">=", D(0)), "min")
    assert solution.value == D(3)


def test_large_continuous_domain_is_bisected() -> None:
    money = Domain(D(0), D(1_000_000_000), D("0.01"))
    solution = solve(lambda x: x * 3, money, Goal(">=", D(1000)), "min")
    assert solution.value == D("333.34")
    assert solution.boundary is not None and solution.boundary.value == D("333.33")


def test_pair_gives_the_equal_level_and_trade_off_rows() -> None:
    equal, rows = solve_pair(
        lambda a, b: (a + b) / 2, SCORES, SCORES, Goal(">=", D(8)), "min", [D(6), D(10)]
    )
    assert equal.value == D(8)
    assert [(row.first, row.second) for row in rows] == [(D(6), D(10)), (D(10), D(6))]
