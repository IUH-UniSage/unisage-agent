"""Target questions ("cuối kỳ cần bao nhiêu để được A+") turned into a
`CalculationResult`, for built-in and regulation formulas alike.

The forward formula stays the single source of truth: `solver.py` re-runs it
for each candidate value of the unknown, so every rounding rule is honoured.
This module only picks the domain of each unknown, runs the solver and writes
the answer (the value found, the check that proves it, and why one step lower
is not enough). Spec: docs/specs/SPEC-calc-engine.md §solver.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from app.calculation.formulas import GRADE_SCALE, HALF_POINT_LOW, fixed, plain
from app.calculation.result import CalculationInputError, CalculationResult, Step
from app.calculation.solver import Domain, Goal, Solution, Want, solve, solve_pair

Compute = Callable[[Mapping[str, object]], CalculationResult]
# Rows of the two-unknowns trade-off table.
TRADE_OFF_ROWS = 5


@dataclass(frozen=True)
class Unknown:
    name: str
    symbol: str
    domain: Domain
    # Recorded scores are half-point rounded: a rounded value v is reached from v - 0.25.
    half_point: bool = False


@dataclass(frozen=True)
class Target:
    label: str  # what the result is called: "ĐTKHP"
    goal: Goal
    grade: str | None = None  # "A+" when the goal came from a letter grade
    want: Want = "min"


def grade_goal(letter: str) -> Goal | None:
    """A letter grade as a goal on a 10-scale score: at least the band's lower bound."""

    band = next((band for band in GRADE_SCALE if band.letter == letter.strip().upper()), None)
    return None if band is None else Goal(">=", band.min_score)


def _goal_text(target: Target) -> str:
    sign = "≥" if target.goal.comparator == ">=" else "≤"
    value = fixed(target.goal.value, 1) if target.grade else plain(target.goal.value)
    text = f"{target.label} {sign} {value}"
    return f"{text} ({target.grade})" if target.grade else text


def _value_text(unknown: Unknown, value: Decimal) -> str:
    return fixed(value, 1) if unknown.half_point or unknown.domain.step < 1 else plain(value)


def _raw_threshold(unknown: Unknown, value: Decimal, want: Want) -> str:
    """For a half-point rounded score: the raw scores that round to `value`."""

    if not unknown.half_point:
        return ""
    if want == "min":
        low = max(value - HALF_POINT_LOW, unknown.domain.min)
        return f" (điểm từ {plain(low)} trở lên được làm tròn thành {fixed(value, 1)})"
    high = min(value + HALF_POINT_LOW, unknown.domain.max)
    return f" (điểm dưới {plain(high)} được làm tròn tối đa thành {fixed(value, 1)})"


def _with(
    known: Mapping[str, object], names: Sequence[str], values: Sequence[Decimal]
) -> dict[str, object]:
    return {**known, **{name: plain(value) for name, value in zip(names, values, strict=True)}}


def _forward(
    compute: Compute, known: Mapping[str, object], names: Sequence[str]
) -> Callable[..., Decimal | None]:
    def run(*values: Decimal) -> Decimal | None:
        try:
            return compute(_with(known, names, values)).primary_value
        except CalculationInputError:
            return None

    return run


def _given_inputs(
    compute: Compute, known: Mapping[str, object], unknowns: Sequence[Unknown]
) -> tuple[tuple[str, str], ...]:
    """The inputs the student gave: the formula's input lines, minus those that change
    with the unknowns (found by running it at both ends of their domains)."""

    names = [unknown.name for unknown in unknowns]
    low = compute(_with(known, names, [unknown.domain.min for unknown in unknowns])).inputs
    high = compute(_with(known, names, [unknown.domain.max for unknown in unknowns])).inputs
    return tuple(item for item in low if item in high)


def check_known(compute: Compute, known: Mapping[str, object], unknowns: Sequence[Unknown]) -> None:
    """Raise the input errors of the parameters the student gave (not the unknowns),
    so they are asked again before anything is solved."""

    probe = {**known, **{unknown.name: plain(unknown.domain.min) for unknown in unknowns}}
    try:
        compute(probe)
    except CalculationInputError as exc:
        names = {unknown.name for unknown in unknowns}
        errors = [error for error in exc.errors if error.field not in names]
        if errors:
            raise CalculationInputError(errors) from exc


def _trial_step(
    label: str, unknown: Unknown, value: Decimal, result: Decimal | None, target: Target
) -> Step:
    met = result is not None and target.goal.met(result)
    shown = "không tính được" if result is None else fixed(result, 1)
    verdict = "đạt" if met else "chưa đạt"
    return Step(
        label=label,
        symbolic=f"{unknown.symbol} = {_value_text(unknown, value)}",
        substituted=(
            f"{unknown.symbol} = {_value_text(unknown, value)} → {target.label} {shown} ({verdict})"
        ),
        value=result if result is not None else Decimal(0),
        display=shown,
    )


def solve_one(
    compute: Compute, known: Mapping[str, object], unknown: Unknown, target: Target
) -> CalculationResult:
    """One unknown. The answer shows the full substitution with the value found (the
    proof), then the neighbouring value that falls short (why it is the minimum)."""

    forward = _forward(compute, known, [unknown.name])
    solution = solve(forward, unknown.domain, target.goal, target.want)
    shown_value = (
        solution.value
        if solution.value is not None
        else (solution.at_limit.value if solution.at_limit is not None else unknown.domain.max)
    )
    proof = compute(_with(known, [unknown.name], [shown_value]))
    best = "nhỏ nhất" if target.want == "min" else "lớn nhất"
    text = (*proof.formula_text, f"Tìm {unknown.symbol} {best} sao cho {_goal_text(target)}")
    checks, summary, outputs = _describe(solution, unknown, target)
    label = f"Thay số với {unknown.symbol} = {_value_text(unknown, shown_value)}"
    steps = tuple(
        Step(
            f"{label}: {step.label}" if index == 0 else step.label,
            step.symbolic,
            step.substituted,
            step.value,
            step.display,
            step.note,
        )
        for index, step in enumerate(proof.steps)
    )
    return CalculationResult(
        formula_id="target",
        title=f"{unknown.symbol} cần đạt để {_goal_text(target)}",
        formula_text=text,
        inputs=_given_inputs(compute, known, [unknown]),
        steps=steps + checks,
        outputs=outputs,
        summary=summary,
        primary_value=solution.value,
    )


def _describe(
    solution: Solution, unknown: Unknown, target: Target
) -> tuple[tuple[Step, ...], str, tuple[tuple[str, str], ...]]:
    goal = _goal_text(target)
    if solution.value is None:
        limit = solution.at_limit
        assert limit is not None
        reached = "không tính được" if limit.result is None else f"**{fixed(limit.result, 1)}**"
        steps: tuple[Step, ...] = ()
        summary = (
            f"Dù {unknown.symbol} đạt {_value_text(unknown, limit.value)} thì {target.label} chỉ "
            f"được {reached}, **không đạt** {goal}."
        )
        return steps, summary, ((unknown.symbol, "không đạt được"),)
    value_text = _value_text(unknown, solution.value)
    if solution.boundary is None:
        # The very first value tried (0 for a minimum) already meets the goal.
        summary = (
            f"Với các điểm hiện có, {unknown.symbol} {value_text} vẫn đạt {goal}: "
            f"bạn đã chắc chắn đạt mục tiêu này."
        )
        return (), summary, ((unknown.symbol, value_text),)
    boundary = solution.boundary
    checks = (_trial_step("Mức liền kề", unknown, boundary.value, boundary.result, target),)
    word = "tối thiểu" if target.want == "min" else "tối đa"
    summary = (
        f"Cần {unknown.symbol} {word} **{value_text}**"
        f"{_raw_threshold(unknown, solution.value, target.want)} để {goal}."
    )
    return checks, summary, ((unknown.symbol, value_text),)


def _shown_values(
    forward: Callable[..., Decimal | None], first: Domain, second: Domain, target: Target
) -> list[Decimal]:
    """Up to TRADE_OFF_ROWS evenly spread values of the first unknown, only from the
    range where the goal is still reachable (rows saying "không đạt được" are noise)."""

    other = second.max if target.want == "min" else second.min
    edge = solve(lambda value: forward(value, other), first, target.goal, target.want).value
    if edge is None:
        return []
    low, high = (edge, first.max) if target.want == "min" else (first.min, edge)
    points = Domain(low, high, first.step).points()
    if len(points) <= TRADE_OFF_ROWS:
        return points
    last = len(points) - 1
    picked = {points[round(index * last / (TRADE_OFF_ROWS - 1))] for index in range(TRADE_OFF_ROWS)}
    return sorted(picked)


def solve_two(
    compute: Compute,
    known: Mapping[str, object],
    unknowns: tuple[Unknown, Unknown],
    target: Target,
) -> CalculationResult:
    """Two unknowns: the level needed when both are equal, plus a short table of
    the trade-off (each value of the first and the best value of the second)."""

    first, second = unknowns
    forward = _forward(compute, known, [first.name, second.name])
    equal, rows = solve_pair(
        forward,
        first.domain,
        second.domain,
        target.goal,
        target.want,
        _shown_values(forward, first.domain, second.domain, target),
    )
    goal = _goal_text(target)
    best = "nhỏ nhất" if target.want == "min" else "lớn nhất"
    both = f"{first.symbol} = {second.symbol}"
    probe = compute(_with(known, [first.name, second.name], [first.domain.max, second.domain.max]))
    inputs = _given_inputs(compute, known, unknowns)
    text = (*probe.formula_text, f"Tìm mức {best} khi {both}, và bảng đánh đổi sao cho {goal}")
    table = (
        (first.symbol, f"{second.symbol} cần {'tối thiểu' if target.want == 'min' else 'tối đa'}"),
        *(
            (
                _value_text(first, row.first),
                "không đạt được" if row.second is None else _value_text(second, row.second),
            )
            for row in rows
        ),
    )
    if equal.value is None:
        summary = f"Dù {first.symbol} và {second.symbol} cùng đạt tối đa vẫn **không đạt** {goal}."
        outputs: tuple[tuple[str, str], ...] = ((both, "không đạt được"),)
    else:
        value = _value_text(first, equal.value)
        summary = (
            f"Nếu {first.symbol} và {second.symbol} bằng nhau thì cần mỗi cột "
            f"{'tối thiểu' if target.want == 'min' else 'tối đa'} **{value}**"
            f"{_raw_threshold(first, equal.value, target.want)} để {goal}. "
            f"Các cách kết hợp khác ở bảng dưới."
        )
        outputs = ((both, value),)
    steps: tuple[Step, ...] = ()
    if equal.met_by is not None:
        steps = (
            Step(
                label="Kiểm tra",
                symbolic=both,
                substituted=(
                    f"{first.symbol} = {second.symbol} = {_value_text(first, equal.met_by.value)}"
                    f" → {target.label} {fixed(equal.met_by.result or Decimal(0), 1)} (đạt)"
                ),
                value=equal.met_by.result or Decimal(0),
                display=fixed(equal.met_by.result or Decimal(0), 1),
            ),
        )
    return CalculationResult(
        formula_id="target",
        title=f"{first.symbol} và {second.symbol} cần đạt để {goal}",
        formula_text=text,
        inputs=inputs,
        steps=steps,
        outputs=outputs,
        summary=summary,
        primary_value=equal.value,
        table=table,
    )
