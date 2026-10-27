"""Pure arithmetic as a capability of its own (26/10/2026): a sum of credits, hours or
amounts the student (or a cited document) already states, computed by code from a
whitelisted expression - never by the LLM in its head, and never as a business
conclusion.

The LLM only writes the expression (`+ - * /`, their typographic forms, parentheses and
decimals) with a label and a
unit; this module parses it with a hand-written recursive-descent parser (no `eval`, no
`ast` on model text), evaluates it in `Decimal`, and turns it into the same
`CalculationResult` the registry formulas produce, so the panel, the review and the
trace treat it like any other calculation. Who may ask for it, and whether every
operand really comes from the question or a cited document, is decided by the callers
(`calculation-node` for the student's own numbers, `calc_requests` for the advisory
answer) - grounding never grants that permission by itself.

Spec: docs/specs/SPEC-calc-engine.md (số học thuần).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation

from app.calculation.formulas import fixed, fmt, money, plain, round_half_up
from app.calculation.result import CalculationResult, Step

logger = logging.getLogger(__name__)

ARITHMETIC_ID = "arithmetic"
ARITHMETIC_VERSION = "1"
ARITHMETIC_SOURCE_OWN = "số liệu do người hỏi nêu trong câu hỏi"
ARITHMETIC_TITLE = "Phép tính số học"
EXPRESSION_MAX_LENGTH = 200
OPERANDS_MAX = 20
DEPTH_MAX = 8
LABEL_MAX = 80
UNIT_MAX = 20
_MONEY_UNITS = frozenset({"đồng", "vnđ", "vnd", "đ"})
_HOUR_UNITS = frozenset({"giờ", "h", "tiếng"})
_NORMALISE = str.maketrans(
    {
        "\u00d7": "*",
        "\u00b7": "*",
        "x": "*",
        "X": "*",
        "\u00f7": "/",
        ":": "/",
        "\u2212": "-",
        "\u2013": "-",
    }
)
_TOKEN = re.compile(r"\s*(\d+(?:[.,]\d+)*|[-+*/()])")
_THOUSANDS = re.compile(r"^\d{1,3}(?:\.\d{3})+$")


class ArithmeticError(ValueError):
    """The expression is not a plain arithmetic expression we accept."""


@dataclass(frozen=True)
class ArithmeticRequest:
    expression: str
    label: str
    unit: str | None = None
    # [n] markers of the documents the operands were read from (advisory requests).
    sources: tuple[int, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    expression: str  # normalised, as shown to the student (typographic signs, spaced)
    operands: tuple[Decimal, ...]
    value: Decimal


def parse_request(data: object) -> ArithmeticRequest | None:
    """A `{"expression", "label", "unit", "sources"}` object from the model, or None."""

    if not isinstance(data, Mapping):
        return None
    expression = data.get("expression")
    label = data.get("label")
    if not isinstance(expression, str) or not expression.strip():
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    unit = data.get("unit")
    unit_text = (
        " ".join(unit.split())[:UNIT_MAX] if isinstance(unit, str) and unit.strip() else None
    )
    raw_sources = data.get("sources")
    sources: tuple[int, ...] = ()
    if isinstance(raw_sources, list):
        sources = tuple(
            item for item in raw_sources if isinstance(item, int) and not isinstance(item, bool)
        )
    return ArithmeticRequest(
        expression=expression.strip()[:EXPRESSION_MAX_LENGTH],
        label=" ".join(label.replace("*", "").split())[:LABEL_MAX],
        unit=unit_text,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Tokeniser + recursive-descent parser (whitelist: numbers + - * / parentheses)
# ---------------------------------------------------------------------------


def _number(token: str) -> Decimal:
    """ "1.200.000" (dotted thousands) → 1200000; "7.5" or "7,5" → 7.5; anything else refused."""

    if _THOUSANDS.match(token):
        return Decimal(token.replace(".", ""))
    if token.count(".") + token.count(",") <= 1:
        try:
            return Decimal(token.replace(",", "."))
        except InvalidOperation as exc:
            raise ArithmeticError(f"không đọc được số `{token}`") from exc
    raise ArithmeticError(f"số `{token}` không rõ dấu thập phân hay phân cách hàng nghìn")


def _tokens(expression: str) -> list[str]:
    if len(expression) > EXPRESSION_MAX_LENGTH:
        raise ArithmeticError("biểu thức quá dài")
    text = expression.translate(_NORMALISE)
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            rest = text[position:].strip()
            raise ArithmeticError(f"ký tự không được phép trong biểu thức: `{rest[:12]}`")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise ArithmeticError("biểu thức rỗng")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._index = 0
        self.operands: list[Decimal] = []
        self._depth = 0

    def _peek(self) -> str | None:
        return self._tokens[self._index] if self._index < len(self._tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ArithmeticError("biểu thức kết thúc đột ngột")
        self._index += 1
        return token

    def parse(self) -> Decimal:
        value = self._expr()
        if self._peek() is not None:
            raise ArithmeticError(f"thừa `{self._peek()}` trong biểu thức")
        return value

    def _expr(self) -> Decimal:
        value = self._term()
        while self._peek() in ("+", "-"):
            operator = self._take()
            right = self._term()
            value = value + right if operator == "+" else value - right
        return value

    def _term(self) -> Decimal:
        value = self._factor()
        while self._peek() in ("*", "/"):
            operator = self._take()
            right = self._factor()
            if operator == "*":
                value = value * right
            else:
                if right == 0:
                    raise ArithmeticError("chia cho 0")
                value = value / right
        return value

    def _factor(self) -> Decimal:
        token = self._take()
        if token == "-":
            return -self._factor()
        if token == "+":
            return self._factor()
        if token == "(":
            self._depth += 1
            if self._depth > DEPTH_MAX:
                raise ArithmeticError("quá nhiều ngoặc lồng nhau")
            value = self._expr()
            if self._take() != ")":
                raise ArithmeticError("thiếu dấu đóng ngoặc")
            self._depth -= 1
            return value
        if token in ("*", "/", ")"):
            raise ArithmeticError(f"`{token}` đứng sai chỗ")
        number = _number(token)
        self.operands.append(number)
        if len(self.operands) > OPERANDS_MAX:
            raise ArithmeticError(f"quá {OPERANDS_MAX} số hạng")
        return number


def _display_token(token: str) -> str:
    if token == "*":
        return "\u00d7"
    if token == "/":
        return "\u00f7"
    if token in "+-()":
        return token
    return plain(_number(token))


def evaluate(expression: str) -> Evaluation:
    """Raises `ArithmeticError` for anything outside the whitelist, a division by zero or
    an unreadable number."""

    tokens = _tokens(expression)
    parser = _Parser(tokens)
    try:
        value = parser.parse()
    except (DivisionByZero, InvalidOperation) as exc:
        raise ArithmeticError("phép tính không thực hiện được") from exc
    if not value.is_finite():
        raise ArithmeticError("kết quả không hữu hạn")
    # A division's unending expansion (10/60) is cut at 9 places so 13 h 10 x 2 + 11 h 10
    # is exactly 37.5, not 37.4999…; the student sees at most 2 decimals anyway.
    value = round_half_up(value, 9)
    shown = " ".join(_display_token(token) for token in tokens)
    shown = shown.replace("( ", "(").replace(" )", ")")
    return Evaluation(shown, tuple(parser.operands), value)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def display(value: Decimal, unit: str | None) -> str:
    key = (unit or "").strip().lower()
    if key in _MONEY_UNITS:
        return money(value)
    text = fmt(value)
    if unit:
        text = f"{text} {unit}"
    if key in _HOUR_UNITS and value != value.to_integral_value():
        hours = int(value)
        minutes = int(((value - hours) * 60).to_integral_value())
        text += f" (= {hours} giờ {minutes:02d} phút)"
    return text


def _short_value(value: Decimal) -> str:
    return fixed(value, 2).rstrip("0").rstrip(".")


def arithmetic_result(
    request: ArithmeticRequest, evaluation: Evaluation, *, source: str
) -> CalculationResult:
    """The block the student sees: one substituted line and the result with its unit.
    `source` names where the operands came from (the question, or the cited documents)."""

    shown = display(evaluation.value, request.unit)
    return CalculationResult(
        formula_id=ARITHMETIC_ID,
        title=f"{ARITHMETIC_TITLE}: {request.label}",
        formula_text=(
            "Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào",
        ),
        inputs=tuple(
            (f"Số hạng {index}", plain(operand))
            for index, operand in enumerate(evaluation.operands, start=1)
        ),
        steps=(
            Step(
                label=request.label,
                symbolic=evaluation.expression,
                substituted=f"{evaluation.expression} = {_short_value(evaluation.value)}",
                value=evaluation.value,
                display=shown,
            ),
        ),
        outputs=((request.label, shown),),
        summary=f"{request.label}: **{shown}**",
        values={"ket_qua": evaluation.value},
        version=ARITHMETIC_VERSION,
        source=source,
    )
