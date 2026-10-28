"""Inverse calculation: the value one parameter must reach for an output to meet a
target ("cuối kỳ cần bao nhiêu để được A+", "ĐHB tối thiểu để XT1 đạt 24", "đăng ký
tối đa mấy tín để học phí không quá 8 triệu").

Python, not the LLM: the unknown parameter is searched on its own grid (`ParamSpec.min`,
`max`, and the coarser of `step` and the rounding the formula applies to it - a score
the regulation rounds to 0.5 is searched in halves) with the registered formula
evaluated at each probe, so the answer obeys every rounding rule the forward
calculation has. Every registered formula is monotonic in each numeric parameter
(sums, weighted sums, means, maxima, prices), which a bisection over the grid relies
on; the direction is measured, not assumed. The result shows the forward calculation
at the solution and a check with the adjacent grid value, the way a student would
verify it by hand.

Spec: docs/specs/SPEC-calc-engine.md (tính ngược).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from app.calculation.formulas import (
    FORMULAS,
    GRADE_SCALE,
    FormulaId,
    OutputSpec,
    ParamSpec,
    calculate,
    fmt,
    label_for,
    money,
    output_spec,
    param_spec,
    plain,
    round_half_up,
    to_decimal,
)
from app.calculation.result import CalculationInputError, CalculationResult, FieldError, Step

Comparison = Literal[">=", "<=", "=="]
COMPARISONS: tuple[Comparison, ...] = (">=", "<=", "==")
_DEFAULT_STEP = Decimal("0.01")
# Outputs on the 10-point scale, where a letter grade names its lower bound.
_SCORE10_OUTPUTS = frozenset({"dtkhp", "diem10", "diem_hp"})
_GP4_OUTPUTS = frozenset({"thang4"})
_UNKNOWN = re.compile(r"^([a-z0-9_]+)(?:\[(\d{1,2})\])?$")
_SIGN = {">=": "≥", "<=": "≤", "==": "="}


@dataclass(frozen=True)
class Unknown:
    """The parameter to solve for: a scalar, or one position (1-based) of a list param."""

    name: str
    index: int | None = None

    def text(self) -> str:
        return self.name if self.index is None else f"{self.name}[{self.index}]"


@dataclass(frozen=True)
class Target:
    output: str
    op: Comparison
    value: Decimal


@dataclass(frozen=True)
class Solution:
    value: Decimal | None  # None = no grid value meets the target
    result: CalculationResult  # forward calculation at the solution (or at the boundary)
    output: Decimal | None
    checked: tuple[Decimal, Decimal | None] | None = None  # (adjacent probe, its output)
    maximum: bool = False  # the solution is the largest feasible value, not the smallest


def parse_unknown(text: object) -> Unknown | None:
    if not isinstance(text, str):
        return None
    match = _UNKNOWN.match(text.strip().lower())
    if match is None:
        return None
    index = match.group(2)
    return Unknown(match.group(1), int(index) if index else None)


def parse_target(formula_id: FormulaId, data: object) -> Target | FieldError:
    """`{"output": "xt1", "op": ">=", "value": 24}`; a letter grade ("A+") as value is
    read as the lower bound of its band on a 10-point output, or its 4-point value."""

    if not isinstance(data, Mapping):
        return FieldError("target", "thiếu mục tiêu (output, op, value)")
    key = str(data.get("output") or "").strip().lower()
    spec = output_spec(formula_id, key)
    if spec is None:
        return FieldError("target", f"công thức không có đầu ra `{key}`")
    op = str(data.get("op") or ">=").strip()
    if op not in COMPARISONS:
        return FieldError("target", f"phép so sánh `{op}` không hợp lệ")
    raw = data.get("value")
    value = _letter_value(key, raw)
    if value is None:
        value = to_decimal(raw)
    if value is None:
        return FieldError("target", "giá trị mục tiêu phải là một số hoặc điểm chữ")
    comparison: Comparison = ">=" if op == ">=" else "<=" if op == "<=" else "=="
    return Target(key, comparison, value)


def _letter_value(output: str, raw: object) -> Decimal | None:
    if not isinstance(raw, str):
        return None
    letter = raw.strip().upper()
    band = next((band for band in GRADE_SCALE if band.letter == letter), None)
    if band is None:
        return None
    if output in _SCORE10_OUTPUTS:
        return band.min_score
    if output in _GP4_OUTPUTS:
        return band.gp4
    return None


def unknown_spec(formula_id: FormulaId, unknown: Unknown) -> ParamSpec | FieldError:
    spec = param_spec(formula_id, unknown.name)
    if spec is None:
        return FieldError(unknown.name, "không phải tham số của công thức này")
    if spec.kind not in ("number", "number_list", "number_or_list"):
        return FieldError(unknown.name, "chỉ tìm được tham số dạng số")
    if spec.kind == "number_list" and unknown.index is None:
        return FieldError(unknown.name, "cần nêu vị trí trong danh sách (VD diem[4])")
    if spec.kind == "number" and unknown.index is not None:
        return FieldError(unknown.name, "tham số này là một số, không có vị trí")
    return spec


def with_value(
    params: Mapping[str, object], unknown: Unknown, value: Decimal | None
) -> dict[str, object]:
    """`params` with the unknown set to `value` (a list position is filled in place; one
    past the end is appended)."""

    filled = dict(params)
    if unknown.index is None:
        filled[unknown.name] = value
        return filled
    current = filled.get(unknown.name)
    items = list(current) if isinstance(current, list | tuple) else []
    position = unknown.index - 1
    while len(items) <= position:
        items.append(None)
    items[position] = value
    filled[unknown.name] = items
    return filled


def search_step(formula_id: FormulaId, spec: ParamSpec, params: Mapping[str, object]) -> Decimal:
    """The grid step: the param's own, or the rounding the formula applies to it when
    that is coarser - a component score rounded to 0.5 is only worth asking in halves."""

    step = spec.step if spec.step is not None and spec.step > 0 else _DEFAULT_STEP
    rounding = FORMULAS[formula_id].rounding_of
    coarser = rounding(params, spec.name) if rounding is not None else None
    return coarser if coarser is not None and coarser > step else step


def _grid(
    formula_id: FormulaId, spec: ParamSpec, params: Mapping[str, object]
) -> tuple[Decimal, Decimal, int]:
    low = spec.min if spec.min is not None else Decimal(0)
    high = spec.max if spec.max is not None else low + 1000
    step = search_step(formula_id, spec, params)
    return low, step, int((high - low) / step)


def _meets(output: Decimal, target: Target) -> bool:
    if target.op == ">=":
        return output >= target.value
    if target.op == "<=":
        return output <= target.value
    return output == target.value


def solve(
    formula_id: FormulaId,
    params: Mapping[str, object],
    unknown: Unknown,
    target: Target,
) -> Solution:
    """The smallest (for a rising output) or largest (falling) grid value of the unknown
    that meets the target - the minimum a student needs, or the maximum they may
    register. Raises `CalculationInputError` for the other params' errors."""

    spec = unknown_spec(formula_id, unknown)
    if isinstance(spec, FieldError):
        raise CalculationInputError([spec])
    low, step, count = _grid(formula_id, spec, params)
    failures: list[CalculationInputError] = []
    cache: dict[int, tuple[CalculationResult | None, Decimal | None]] = {}

    def probe_at(i: int) -> Decimal:
        return low + step * i

    def evaluate(i: int) -> tuple[CalculationResult | None, Decimal | None]:
        if i not in cache:
            try:
                result = calculate(formula_id, with_value(params, unknown, probe_at(i)))
            except CalculationInputError as exc:
                failures.append(exc)
                cache[i] = (None, None)
            else:
                cache[i] = (result, result.values.get(target.output))
        return cache[i]

    def valid(i: int) -> bool:
        return evaluate(i)[0] is not None

    # The formula may refuse one end of the grid (0 credits, a ĐGNL result above the top
    # score): the valid probes form one interval, found by bisection on its ends. Both
    # ends refused = the other params are wrong, and that error is the caller's.
    if not valid(0) and not valid(count):
        raise failures[0]
    first_valid, last_valid = 0, count
    if not valid(0):
        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi) // 2
            if valid(mid):
                hi = mid
            else:
                lo = mid + 1
        first_valid = lo
    elif not valid(count):
        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if valid(mid):
                lo = mid
            else:
                hi = mid - 1
        last_valid = lo
    at_low, at_high = evaluate(first_valid)[1], evaluate(last_valid)[1]
    if at_low is None or at_high is None:
        raise CalculationInputError(
            [FieldError("target", f"công thức không trả về đầu ra `{target.output}`")]
        )
    rising = at_high >= at_low

    def ok(i: int) -> bool:
        # "==" is searched like ">=" (first probe reaching the value) and checked after.
        output = evaluate(i)[1]
        if output is None:
            return False
        if target.op == "<=":
            return output <= target.value
        return output >= target.value

    # Along the grid the predicate flips at most once (monotonic output): the first
    # index that meets the target when the feasible side is the high end, the last
    # one when it is the low end.
    wants_min = target.op != "<=" if rising else target.op == "<="
    lo, hi = first_valid, last_valid
    if wants_min:
        if not ok(hi):
            boundary = evaluate(hi)[0]
            assert boundary is not None
            return Solution(None, boundary, boundary.values.get(target.output))
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        index, adjacent = lo, lo - 1 if lo > first_valid else None
    else:
        if not ok(lo):
            boundary = evaluate(lo)[0]
            assert boundary is not None
            return Solution(None, boundary, boundary.values.get(target.output))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        index, adjacent = lo, lo + 1 if lo < last_valid else None
    result, output = evaluate(index)
    assert result is not None
    if target.op == "==" and (output is None or output != target.value):
        return Solution(None, result, output)
    checked = None
    if adjacent is not None:
        checked = (probe_at(adjacent), evaluate(adjacent)[1])
    return Solution(probe_at(index), result, output, checked, maximum=not wants_min)


def solved_result(
    formula_id: FormulaId,
    params: Mapping[str, object],
    unknown: Unknown,
    target: Target,
) -> CalculationResult:
    """`solve` rendered as one calculation: what was searched, the forward steps at the
    solution, and the check with the adjacent value."""

    spec = unknown_spec(formula_id, unknown)
    if isinstance(spec, FieldError):
        raise CalculationInputError([spec])
    output = output_spec(formula_id, target.output)
    assert output is not None
    solution = solve(formula_id, params, unknown, target)
    label = _unknown_label(spec, unknown, params)
    short = _short_label(spec, unknown, params)
    goal = f"{_short_output(output)} {_SIGN[target.op]} {_show(target.value, output=output)}"
    base = solution.result
    if solution.value is None:
        return _no_solution(base, spec, label, goal, output, solution)
    extreme = "tối đa" if solution.maximum else "tối thiểu"
    step = search_step(formula_id, spec, params)
    found = plain(solution.value)
    search = Step(
        label="Giá trị cần tìm",
        symbolic=f"Tìm {label} {extreme} để {goal}",
        substituted=f"{short} = {found} (thử lần lượt theo bước {plain(step)})",
        value=solution.value,
        display=found,
    )
    check = (
        f"{short} = {found} → {_short_output(output)} {_show(solution.output, output=output)} (đạt)"
    )
    if solution.checked is not None:
        adjacent, adjacent_out = solution.checked
        shown = _show(adjacent_out, beside=solution.output, output=output)
        verdict = (
            "" if adjacent_out is not None and _meets(adjacent_out, target) else " (không đạt)"
        )
        check += f"; {short} = {plain(adjacent)} → {shown}{verdict}"
    steps = (
        search,
        *base.steps,
        Step(
            label="Kiểm tra",
            symbolic="Thay giá trị tìm được và giá trị liền kề vào công thức",
            substituted=check,
            value=solution.value,
            display=found,
        ),
    )
    return replace(
        base,
        title=f"{label} {extreme} để {goal}",
        formula_text=(*base.formula_text, f"Tìm {label} {extreme} sao cho {goal}"),
        steps=steps,
        outputs=((short, found), *base.outputs),
        summary=(
            f"{label} {extreme} **{found}** → {_short_output(output)}"
            f" **{_show(solution.output, output=output)}**"
        ),
        values={**base.values, "nghiem": solution.value},
    )


def _no_solution(
    base: CalculationResult,
    spec: ParamSpec,
    label: str,
    goal: str,
    output: OutputSpec,
    solution: Solution,
) -> CalculationResult:
    low = spec.min if spec.min is not None else Decimal(0)
    high = spec.max if spec.max is not None else low
    boundary = f"[{plain(low)}; {plain(high)}]"
    return replace(
        base,
        title=f"{label} để {goal}",
        formula_text=(*base.formula_text, f"Tìm {label} sao cho {goal}"),
        steps=(
            Step(
                label="Giá trị cần tìm",
                symbolic=f"Tìm {label} trong {boundary} để {goal}",
                substituted=f"không có giá trị nào trong {boundary} đạt được",
                value=Decimal(0),
                display="",
            ),
            *base.steps,
        ),
        summary=(
            f"Không có giá trị {label} nào trong {boundary} làm {goal}: thử giá trị biên"
            f" thì {_short_output(output)} chỉ được **{_show(solution.output, output=output)}**"
        ),
        warnings=(*base.warnings, "Mục tiêu không đạt được với các số liệu đã cho."),
    )


def _unknown_label(spec: ParamSpec, unknown: Unknown, params: Mapping[str, object]) -> str:
    if unknown.index is not None:
        return f"{spec.tab_label} thành phần thứ {unknown.index}"
    return label_for(spec, params)


def _short_label(spec: ParamSpec, unknown: Unknown, params: Mapping[str, object]) -> str:
    if unknown.index is not None:
        return f"{spec.tab_label} {unknown.index}"
    label = label_for(spec, params)
    return label if len(label) <= 32 else spec.tab_label


def _short_output(output: OutputSpec) -> str:
    return output.label.split(" (")[0].split(" = ")[0]


def _show(
    value: Decimal | None, *, beside: Decimal | None = None, output: OutputSpec | None = None
) -> str:
    """`fmt` (or whole đồng for a money output), with more digits when two decimals
    would make the adjacent probe look equal to the solution's output ("≈ 24" next to
    "24")."""

    if value is None:
        return "không tính được"
    if output is not None and output.money:
        return money(value)
    shown = fmt(value)
    if beside is not None and shown.lstrip("≈ ") == fmt(beside).lstrip("≈ ") and value != beside:
        return f"≈ {plain(round_half_up(value, 4))}"
    return shown
