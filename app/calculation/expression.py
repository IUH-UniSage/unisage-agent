"""Safe evaluator for a formula an LLM copied out of a regulation chunk.

The expression string is untrusted. It is parsed with `ast.parse(mode="eval")`
and every node is checked against an allowlist before anything is computed;
`eval`/`exec`/`compile` are never used. Only + - * / unary minus, numeric
literals, declared variables and round/min/max are allowed - no power (so no
`9**9**9`), no attributes, subscripts, comparisons or strings. Arithmetic is
`Decimal` with overflow/invalid/division traps and a bound on every
intermediate value.

Spec: docs/specs/SPEC-calc-engine.md §expression.py.
"""

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

from app.calculation.formulas import TIMES, equals, fmt, plain, round_half_up, to_decimal
from app.calculation.result import CalculationInputError, CalculationResult, FieldError, Step

MAX_EXPRESSION_LENGTH = 200
MAX_NODES = 50
MAX_DEPTH = 10
MAX_VARIABLES = 10
MAX_CONSTANT = Decimal("1e9")
MAX_CONSTANT_PLACES = 6
MAX_VALUE = Decimal("1e12")
MAX_ROUND_PLACES = 4
MIN_MAX_ARGS = (2, 10)
VARIABLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
FUNCTIONS = frozenset({"round", "min", "max"})

_PRECEDENCE: dict[type[ast.operator], int] = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2}
_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: TIMES,
    ast.Div: "/",
}


@dataclass(frozen=True)
class FormulaVariable:
    name: str
    label: str
    unit: str | None = None
    min: Decimal | None = None
    max: Decimal | None = None


@dataclass(frozen=True)
class RetrievedFormula:
    expression: str
    variables: tuple[FormulaVariable, ...]
    result_label: str


class FormulaRejected(Exception):
    """The expression is not something we are willing to evaluate."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _constant_value(node: ast.Constant) -> Decimal:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FormulaRejected("chỉ chấp nhận hằng số dạng số")
    number = Decimal(str(value))
    if not number.is_finite() or abs(number) > MAX_CONSTANT:
        raise FormulaRejected("hằng số quá lớn")
    exponent = number.normalize().as_tuple().exponent
    if isinstance(exponent, int) and -exponent > MAX_CONSTANT_PLACES:
        raise FormulaRejected("hằng số có quá nhiều chữ số thập phân")
    return number


class _Checker:
    def __init__(self, declared: frozenset[str]) -> None:
        self.declared = declared
        self.used: set[str] = set()
        self.nodes = 0

    def check(self, node: ast.AST, depth: int) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise FormulaRejected("biểu thức quá dài")
        if depth > MAX_DEPTH:
            raise FormulaRejected("biểu thức lồng quá sâu")

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _PRECEDENCE:
                raise FormulaRejected("phép toán không được hỗ trợ")
            self.check(node.left, depth + 1)
            self.check(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub | ast.UAdd):
                raise FormulaRejected("phép toán không được hỗ trợ")
            self.check(node.operand, depth + 1)
        elif isinstance(node, ast.Constant):
            _constant_value(node)
        elif isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load) or node.id not in self.declared:
                raise FormulaRejected(f"biến không được khai báo: {node.id}")
            self.used.add(node.id)
        elif isinstance(node, ast.Call):
            self._check_call(node, depth)
        else:
            raise FormulaRejected(f"cú pháp không được hỗ trợ: {type(node).__name__}")

    def _check_call(self, node: ast.Call, depth: int) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise FormulaRejected("chỉ cho phép round, min, max")
        if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            raise FormulaRejected("hàm không nhận đối số dạng từ khoá")
        if node.func.id == "round":
            places = node.args[1] if len(node.args) == 2 else None
            if (
                not isinstance(places, ast.Constant)
                or isinstance(places.value, bool)
                or not isinstance(places.value, int)
                or not 0 <= places.value <= MAX_ROUND_PLACES
            ):
                raise FormulaRejected("round(x, n) cần n là số nguyên từ 0 đến 4")
            self.check(node.args[0], depth + 1)
            return
        low, high = MIN_MAX_ARGS
        if not low <= len(node.args) <= high:
            raise FormulaRejected(f"{node.func.id} cần từ {low} đến {high} đối số")
        for arg in node.args:
            self.check(arg, depth + 1)


def validate(formula: RetrievedFormula) -> ast.expr:
    """Raises `FormulaRejected`; returns the checked expression tree."""

    if len(formula.expression) > MAX_EXPRESSION_LENGTH:
        raise FormulaRejected("biểu thức quá dài")
    if not 1 <= len(formula.variables) <= MAX_VARIABLES:
        raise FormulaRejected(f"cần từ 1 đến {MAX_VARIABLES} biến")
    names = [variable.name for variable in formula.variables]
    if len(set(names)) != len(names):
        raise FormulaRejected("tên biến bị trùng")
    for name in names:
        if not VARIABLE_NAME.match(name) or name in FUNCTIONS:
            raise FormulaRejected(f"tên biến không hợp lệ: {name}")
    try:
        tree = ast.parse(formula.expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise FormulaRejected("biểu thức sai cú pháp") from exc

    checker = _Checker(frozenset(names))
    checker.check(tree.body, depth=1)
    unused = set(names) - checker.used
    if unused:
        raise FormulaRejected(f"biến khai báo nhưng không dùng: {', '.join(sorted(unused))}")
    return tree.body


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > MAX_VALUE:
        raise CalculationInputError([FieldError("expression", "kết quả trung gian quá lớn")])
    return value


def _denominator_field(node: ast.expr) -> str:
    return node.id if isinstance(node, ast.Name) else "expression"


def _eval(node: ast.expr, values: Mapping[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Constant):
        return _constant_value(node)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, values)
        return _bounded(-operand if isinstance(node.op, ast.USub) else operand)
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, values)
        right = _eval(node.right, values)
        if isinstance(node.op, ast.Add):
            return _bounded(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded(left * right)
        if right == 0:
            raise CalculationInputError(
                [FieldError(_denominator_field(node.right), "mẫu số bằng 0")]
            )
        return _bounded(left / right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "round":
            places = node.args[1]
            assert isinstance(places, ast.Constant) and isinstance(places.value, int)
            return round_half_up(_eval(node.args[0], values), places.value)
        args = [_eval(arg, values) for arg in node.args]
        return min(args) if node.func.id == "min" else max(args)
    raise FormulaRejected("cú pháp không được hỗ trợ")  # unreachable after validate()


def _show(node: ast.expr, values: Mapping[str, Decimal] | None, parent: int = 0) -> str:
    """Infix text of the tree - variable names, or their values when given."""

    if isinstance(node, ast.Constant):
        return plain(_constant_value(node))
    if isinstance(node, ast.Name):
        return node.id if values is None else plain(values[node.id])
    if isinstance(node, ast.UnaryOp):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return f"{sign}{_show(node.operand, values, 3)}"
    if isinstance(node, ast.BinOp):
        precedence = _PRECEDENCE[type(node.op)]
        left = _show(node.left, values, precedence)
        # Right operand of - and / needs parentheses at equal precedence too.
        right = _show(node.right, values, precedence + int(isinstance(node.op, ast.Sub | ast.Div)))
        text = f"{left} {_SYMBOLS[type(node.op)]} {right}"
        return f"({text})" if precedence < parent else text
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return f"{node.func.id}({', '.join(_show(arg, values) for arg in node.args)})"
    return "?"


def evaluate(formula: RetrievedFormula, values: Mapping[str, object]) -> CalculationResult:
    """Raises `FormulaRejected` (bad expression) or `CalculationInputError` (bad values)."""

    tree = validate(formula)

    errors: list[FieldError] = []
    numbers: dict[str, Decimal] = {}
    for variable in formula.variables:
        raw = values.get(variable.name)
        if raw is None:
            errors.append(FieldError(variable.name, "còn thiếu"))
            continue
        number = to_decimal(raw)
        if number is None:
            errors.append(FieldError(variable.name, "phải là một số"))
        elif (variable.min is not None and number < variable.min) or (
            variable.max is not None and number > variable.max
        ):
            errors.append(
                FieldError(
                    variable.name,
                    f"bạn nhập {plain(number)}, phải từ {plain(variable.min or Decimal(0))}"
                    f" đến {plain(variable.max) if variable.max is not None else '…'}",
                )
            )
        else:
            numbers[variable.name] = number
    if errors:
        raise CalculationInputError(errors)

    try:
        with localcontext() as context:
            context.prec = 28
            context.Emax = 24
            context.Emin = -24
            context.traps[Overflow] = True
            context.traps[InvalidOperation] = True
            context.traps[DivisionByZero] = True
            result = _eval(tree, numbers)
            numerator = denominator = None
            if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Div):
                numerator = _eval(tree.left, numbers)
                denominator = _eval(tree.right, numbers)
    except (Overflow, InvalidOperation) as exc:
        raise CalculationInputError([FieldError("expression", "không tính được")]) from exc

    label = formula.result_label
    steps = [
        Step(
            label="Thay số",
            symbolic=f"{label} = {_show(tree, None)}",
            substituted=f"{label} = {_show(tree, numbers)}",
            value=result,
            display=fmt(result),
        )
    ]
    if numerator is not None and denominator is not None and isinstance(tree, ast.BinOp):
        steps += [
            Step(
                label="Tử số",
                symbolic=_show(tree.left, None),
                substituted=f"{_show(tree.left, numbers)} {equals(numerator)}",
                value=numerator,
                display=fmt(numerator),
            ),
            Step(
                label="Mẫu số",
                symbolic=_show(tree.right, None),
                substituted=f"{_show(tree.right, numbers)} {equals(denominator)}",
                value=denominator,
                display=fmt(denominator),
            ),
        ]
    steps.append(
        Step(
            label="Kết quả",
            symbolic=label,
            substituted=f"{label} {equals(result)}",
            value=result,
            display=fmt(result),
        )
    )

    legend = tuple(
        f"{variable.name}: {variable.label}" + (f" ({variable.unit})" if variable.unit else "")
        for variable in formula.variables
    )
    return CalculationResult(
        formula_id="retrieved",
        title=label,
        formula_text=(f"{label} = {_show(tree, None)}", *legend),
        inputs=tuple(
            (variable.label, plain(numbers[variable.name])) for variable in formula.variables
        ),
        steps=tuple(steps),
        outputs=((label, fmt(result)),),
    )
