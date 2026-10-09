"""Generic "how much do I need" solver - knows nothing about any formula.

It only gets a forward function (a built-in formula, or a regulation formula
evaluated by `expression.py`), the unknown variable's domain and a goal, and
tries values until the goal is met. Because it re-runs the exact forward
calculation (every rounding step included), its answer is correct by
definition - unlike inverting the formula by hand, which an LLM gets wrong as
soon as there are rounding steps. Spec: docs/specs/SPEC-calc-engine.md §solver.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Comparator = Literal[">=", "<="]
Want = Literal["min", "max"]
# Discrete domains up to this many values are tried one by one; larger ones are bisected.
MAX_GRID_POINTS = 2000
BISECTION_ROUNDS = 60


@dataclass(frozen=True)
class Domain:
    min: Decimal
    max: Decimal
    step: Decimal

    def points(self) -> list[Decimal]:
        count = int((self.max - self.min) / self.step)
        return [self.min + self.step * index for index in range(count + 1)]

    @property
    def is_grid(self) -> bool:
        return (self.max - self.min) / self.step <= MAX_GRID_POINTS


@dataclass(frozen=True)
class Goal:
    comparator: Comparator
    value: Decimal

    def met(self, result: Decimal) -> bool:
        return result >= self.value if self.comparator == ">=" else result <= self.value


Forward = Callable[[Decimal], Decimal | None]  # None = this value is not computable


@dataclass(frozen=True)
class Trial:
    value: Decimal
    result: Decimal | None
    met: bool


@dataclass(frozen=True)
class Solution:
    """`value` is the best value of the unknown, or None when no value in the domain
    meets the goal. `boundary` is the closest value that does NOT meet it (to show
    why `value` is the minimum/maximum); `at_limit` is the domain end tried when
    nothing works."""

    value: Decimal | None
    met_by: Trial | None
    boundary: Trial | None
    at_limit: Trial | None


def _trial(forward: Forward, value: Decimal, goal: Goal) -> Trial:
    result = forward(value)
    return Trial(value, result, result is not None and goal.met(result))


def solve(forward: Forward, domain: Domain, goal: Goal, want: Want) -> Solution:
    """Smallest (`want="min"`) or largest (`want="max"`) value meeting the goal."""

    if domain.is_grid:
        points = domain.points()
        ordered = points if want == "min" else list(reversed(points))
        previous: Trial | None = None
        for value in ordered:
            trial = _trial(forward, value, goal)
            if trial.met:
                return Solution(trial.value, trial, previous, None)
            previous = trial
        return Solution(None, None, None, previous)
    return _bisect(forward, domain, goal, want)


def _bisect(forward: Forward, domain: Domain, goal: Goal, want: Want) -> Solution:
    """Continuous domains (money...): assumes the result moves one way with the
    unknown, which is checked at both ends before searching."""

    best_end = domain.max if want == "min" else domain.min
    worst_end = domain.min if want == "min" else domain.max
    best = _trial(forward, best_end, goal)
    if not best.met:
        return Solution(None, None, None, best)
    worst = _trial(forward, worst_end, goal)
    if worst.met:
        return Solution(worst.value, worst, None, None)
    met, unmet = best, worst
    for _ in range(BISECTION_ROUNDS):
        middle = (met.value + unmet.value) / 2
        snapped = (middle / domain.step).to_integral_value() * domain.step
        if snapped in (met.value, unmet.value):
            break
        trial = _trial(forward, snapped, goal)
        if trial.met:
            met = trial
        else:
            unmet = trial
    return Solution(met.value, met, unmet, None)


@dataclass(frozen=True)
class TradeOffRow:
    first: Decimal
    second: Decimal | None  # best value of the second unknown, None = unreachable


def solve_pair(
    forward: Callable[[Decimal, Decimal], Decimal | None],
    first: Domain,
    second: Domain,
    goal: Goal,
    want: Want,
    shown: Sequence[Decimal],
) -> tuple[Solution, list[TradeOffRow]]:
    """Two unknowns at once: the "both equal" value, and for each `shown` value of the
    first unknown the best value of the second."""

    def with_first(fixed: Decimal) -> Forward:
        return lambda value: forward(fixed, value)

    equal = solve(lambda value: forward(value, value), first, goal, want)
    rows = [
        TradeOffRow(value, solve(with_first(value), second, goal, want).value) for value in shown
    ]
    return equal, rows
