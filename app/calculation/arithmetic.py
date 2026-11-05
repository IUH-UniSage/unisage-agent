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

import datetime as dt
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation

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
# `round(x; step)` rounds half up to a multiple of step (04/11/2026, d0314: "6,2 → 6,0;
# 7,6 → 7,5" under the 0.5-point rule). Arguments are separated by ";" or ", " (a comma
# right before a digit is a decimal comma).
_ROUND = ("round", "lam_tron")
_FUNCTION = r"round|lam_tron"
_SEPARATOR = r";|,(?=\s)"
_TOKEN = re.compile(rf"\s*({_FUNCTION}|\d+(?:[.,]\d+)*|[-+*/()]|{_SEPARATOR})")
_THOUSANDS = re.compile(r"^\d{1,3}(?:\.\d{3})+$")
# A date literal (03/11/2026, d0329, q0055, d0335): "20/08/2026 - 13/08/2026 + 1" counts
# days, "20/6/2018 - 20" shifts a date. Day first as documents write it; year first as
# the model sometimes does ("2018/06/20").
_DATE_LITERAL = r"\d{1,2}/\d{1,2}/\d{4}|\d{4}/\d{1,2}/\d{1,2}"
_TOKEN_WITH_DATES = re.compile(
    rf"\s*({_DATE_LITERAL}|{_FUNCTION}|\d+(?:[.,]\d+)*|[-+*/()]|{_SEPARATOR})"
)
_HAS_DATE = re.compile(rf"(?<![\d/])(?:{_DATE_LITERAL})(?![\d/])")
DATE_TITLE = "Phép tính ngày"
DATE_SPAN_TEXT = (
    "Ngày sau trừ ngày trước = số ngày cách nhau (tính cả ngày đầu và ngày cuối thì cộng 1)"
)
DATE_SHIFT_TEXT = "Ngày cộng hoặc trừ một số ngày = ngày mới; không áp dụng quy định hay hệ số nào"
DATE_FORMULA_TEXT = (DATE_SPAN_TEXT, DATE_SHIFT_TEXT)
# Shown when a "± 1" was dropped from a date shift: counting both ends belongs to a day
# count, "20 ngày trước 20/6/2018" is 31/5/2018, not 01/6 (04/11/2026, d0335).
DATE_SHIFT_NOTE = (
    "Lùi/cộng số ngày vào một ngày không tính thêm ngày đầu, nên không cộng hay trừ thêm 1."
)
COMPARISONS = {
    ">=": "\u2265",
    "\u2265": "\u2265",
    ">": ">",
    "<=": "\u2264",
    "\u2264": "\u2264",
    "<": "<",
}
CONDITION_TEXT_MAX = 40


class ArithmeticError(ValueError):
    """The expression is not a plain arithmetic expression we accept."""


@dataclass(frozen=True)
class Condition:
    """A threshold the result is compared with ("điểm tổng kết từ 5.0 trở lên thì đạt"),
    so the verdict comes from code, not from the model (04/11/2026, q0059)."""

    comparison: str  # one of the COMPARISONS symbols: ≥ > ≤ <
    threshold: Decimal
    when_true: str
    when_false: str

    def holds(self, value: Decimal) -> bool:
        if self.comparison == "\u2265":
            return value >= self.threshold
        if self.comparison == ">":
            return value > self.threshold
        if self.comparison == "\u2264":
            return value <= self.threshold
        return value < self.threshold

    def verdict(self, value: Decimal) -> str:
        holds = self.holds(value)
        sign = self.comparison if holds else _NEGATED[self.comparison]
        text = self.when_true if holds else self.when_false
        return f"{_short_value(value)} {sign} {plain(self.threshold)} \u2192 **{text}**"


_NEGATED = {"\u2265": "<", ">": "\u2264", "\u2264": ">", "<": "\u2265"}


@dataclass(frozen=True)
class ArithmeticRequest:
    expression: str
    label: str
    unit: str | None = None
    # [n] markers of the documents the operands were read from (advisory requests).
    sources: tuple[int, ...] = ()
    condition: Condition | None = None


@dataclass(frozen=True)
class Evaluation:
    expression: str  # normalised, as shown to the student (typographic signs, spaced)
    operands: tuple[Decimal, ...]
    value: Decimal


@dataclass(frozen=True)
class DateEvaluation:
    """An expression over date literals: a day count (Decimal) or a date."""

    expression: str
    operands: tuple[Decimal, ...]  # the plain numbers (days added, the + 1)
    dates: tuple[dt.date, ...]
    value: Decimal | dt.date
    # A trailing "± 1" was dropped from a date shift (see DATE_SHIFT_NOTE).
    dropped_inclusive: bool = False


_Value = Decimal | dt.date


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
        condition=parse_condition(data.get("condition")),
    )


def parse_condition(data: object) -> Condition | None:
    """`{"compare": ">=", "value": 5.0, "true": "Đạt", "false": "Không đạt"}`, or None when
    anything is missing or not understood (the result is then shown without a verdict)."""

    if not isinstance(data, Mapping):
        return None
    comparison = COMPARISONS.get(str(data.get("compare") or data.get("operator") or "").strip())
    raw = data.get("value", data.get("threshold"))
    when_true, when_false = data.get("true"), data.get("false")
    if comparison is None or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    if not isinstance(when_true, str) or not isinstance(when_false, str):
        return None
    if not when_true.strip() or not when_false.strip():
        return None
    try:
        threshold = _number(str(raw).strip())
    except ArithmeticError:
        return None
    return Condition(
        comparison=comparison,
        threshold=threshold,
        when_true=" ".join(when_true.replace("*", "").split())[:CONDITION_TEXT_MAX],
        when_false=" ".join(when_false.replace("*", "").split())[:CONDITION_TEXT_MAX],
    )


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Half up to a multiple of `step`: 7.6 to 0.5 → 7.5, 8.8 → 9.0, 6.25 → 6.5."""

    if step <= 0:
        raise ArithmeticError("bước làm tròn phải lớn hơn 0")
    return (value / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step


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


def _date(token: str) -> dt.date:
    parts = [int(part) for part in token.split("/")]
    year, month, day = parts if parts[0] > 31 else parts[::-1]
    try:
        return dt.date(year, month, day)
    except ValueError as exc:
        raise ArithmeticError(f"không có ngày `{token}`") from exc


def _is_date(token: str) -> bool:
    return token.count("/") == 2


def has_date(expression: str) -> bool:
    """The expression writes a date literal: evaluate it with `evaluate_dates`."""

    return _HAS_DATE.search(expression) is not None


def _apply(operator: str, left: _Value, right: _Value) -> _Value:
    """+ - * / on numbers; on dates only date - date (days) and date ± whole days."""

    if isinstance(left, Decimal) and isinstance(right, Decimal):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if right == 0:
            raise ArithmeticError("chia cho 0")
        return left / right
    if operator in ("*", "/"):
        raise ArithmeticError("ngày tháng chỉ cộng, trừ được")
    if isinstance(left, dt.date) and isinstance(right, dt.date):
        if operator == "-":
            return Decimal((left - right).days)
        raise ArithmeticError("không cộng hai ngày với nhau")
    days = right if isinstance(right, Decimal) else left
    if not isinstance(days, Decimal) or days != days.to_integral_value():
        raise ArithmeticError("chỉ cộng, trừ một số ngày nguyên vào một ngày")
    if isinstance(left, dt.date):
        shift = dt.timedelta(days=int(days))
        return left + shift if operator == "+" else left - shift
    if operator == "-":
        raise ArithmeticError("không lấy một số trừ đi một ngày")
    assert isinstance(right, dt.date)
    return right + dt.timedelta(days=int(days))


def _tokens(expression: str, pattern: re.Pattern[str] = _TOKEN) -> list[str]:
    if len(expression) > EXPRESSION_MAX_LENGTH:
        raise ArithmeticError("biểu thức quá dài")
    text = expression.translate(_NORMALISE)
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = pattern.match(text, position)
        if match is None:
            rest = text[position:].strip()
            raise ArithmeticError(f"ký tự không được phép trong biểu thức: `{rest[:12]}`")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise ArithmeticError("biểu thức rỗng")
    return tokens


class _Parser:
    """Values are Decimal; a date token (only `evaluate_dates` tokenises one) is a date."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._index = 0
        self.operands: list[Decimal] = []
        self.dates: list[dt.date] = []
        self._depth = 0

    def _peek(self) -> str | None:
        return self._tokens[self._index] if self._index < len(self._tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ArithmeticError("biểu thức kết thúc đột ngột")
        self._index += 1
        return token

    def parse(self) -> _Value:
        value = self._expr()
        if self._peek() is not None:
            raise ArithmeticError(f"thừa `{self._peek()}` trong biểu thức")
        return value

    def _expr(self) -> _Value:
        value = self._term()
        while self._peek() in ("+", "-"):
            operator = self._take()
            value = _apply(operator, value, self._term())
        return value

    def _term(self) -> _Value:
        value = self._factor()
        while self._peek() in ("*", "/"):
            operator = self._take()
            value = _apply(operator, value, self._factor())
        return value

    def _factor(self) -> _Value:
        token = self._take()
        if token == "-":
            value = self._factor()
            if not isinstance(value, Decimal):
                raise ArithmeticError("`-` đứng trước một ngày")
            return -value
        if token == "+":
            return self._factor()
        if _is_date(token):
            date = _date(token)
            self.dates.append(date)
            return date
        if token == "(":
            self._depth += 1
            if self._depth > DEPTH_MAX:
                raise ArithmeticError("quá nhiều ngoặc lồng nhau")
            value = self._expr()
            if self._take() != ")":
                raise ArithmeticError("thiếu dấu đóng ngoặc")
            self._depth -= 1
            return value
        if token in _ROUND:
            return self._round()
        if token in ("*", "/", ")", ";", ","):
            raise ArithmeticError(f"`{token}` đứng sai chỗ")
        number = _number(token)
        self.operands.append(number)
        if len(self.operands) > OPERANDS_MAX:
            raise ArithmeticError(f"quá {OPERANDS_MAX} số hạng")
        return number

    def _round(self) -> Decimal:
        if self._take() != "(":
            raise ArithmeticError("round cần dạng round(số; bước)")
        self._depth += 1
        if self._depth > DEPTH_MAX:
            raise ArithmeticError("quá nhiều ngoặc lồng nhau")
        value = self._expr()
        if self._take() not in (";", ","):
            raise ArithmeticError("round cần dạng round(số; bước)")
        step = self._expr()
        if self._take() != ")":
            raise ArithmeticError("thiếu dấu đóng ngoặc")
        self._depth -= 1
        if not isinstance(value, Decimal) or not isinstance(step, Decimal):
            raise ArithmeticError("không làm tròn một ngày")
        return round_to_step(value, step)


def _display_token(token: str) -> str:
    if token in _ROUND:
        return "làm tròn"
    if token in (";", ","):
        return ";"
    if _is_date(token):
        return _date(token).strftime("%d/%m/%Y")
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
    parser, value = _parse(tokens)
    assert isinstance(value, Decimal)  # no date token without `evaluate_dates`
    if not value.is_finite():
        raise ArithmeticError("kết quả không hữu hạn")
    # A division's unending expansion (10/60) is cut at 9 places so 13 h 10 x 2 + 11 h 10
    # is exactly 37.5, not 37.4999…; the student sees at most 2 decimals anyway.
    value = round_half_up(value, 9)
    return Evaluation(_shown(tokens), tuple(parser.operands), value)


def evaluate_dates(expression: str) -> DateEvaluation:
    """`evaluate` with date literals: date - date is a day count, date ± n days a date.
    The result is a whole number of days or a date; anything else is refused."""

    tokens = _tokens(expression, _TOKEN_WITH_DATES)
    parser, value = _parse(tokens)
    if not parser.dates:
        raise ArithmeticError("biểu thức không có ngày")
    if isinstance(value, Decimal) and value != value.to_integral_value():
        raise ArithmeticError("số ngày không nguyên")
    dropped = False
    if (
        isinstance(value, dt.date)
        and len(tokens) > 2
        and tokens[-1] == "1"
        and tokens[-2] in ("+", "-")
        and len(parser.operands) >= 2
    ):
        # "20/06/2018 - 20 + 1": the both-ends "+ 1" of a day count, misapplied to a shift.
        tokens = tokens[:-2]
        parser, value = _parse(tokens)
        dropped = True
    return DateEvaluation(
        _shown(tokens), tuple(parser.operands), tuple(parser.dates), value, dropped
    )


def _parse(tokens: list[str]) -> tuple[_Parser, _Value]:
    parser = _Parser(tokens)
    try:
        return parser, parser.parse()
    except (DivisionByZero, InvalidOperation, OverflowError) as exc:
        raise ArithmeticError("phép tính không thực hiện được") from exc


def _shown(tokens: list[str]) -> str:
    shown = " ".join(_display_token(token) for token in tokens)
    shown = shown.replace("( ", "(").replace(" )", ")").replace(" ;", ";")
    return shown.replace("làm tròn (", "làm tròn(")


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
    request: ArithmeticRequest,
    evaluation: Evaluation,
    *,
    source: str,
    condition: Condition | None = None,
) -> CalculationResult:
    """The block the student sees: one substituted line and the result with its unit.
    `source` names where the operands came from (the question, or the cited documents).
    `condition`: a grounded threshold the result is compared with - the verdict is added
    to the result line (the caller drops a condition whose threshold is not grounded)."""

    shown = display(evaluation.value, request.unit)
    summary = f"{request.label}: **{shown}**"
    outputs: tuple[tuple[str, str], ...] = ((request.label, shown),)
    if condition is not None:
        verdict = condition.verdict(evaluation.value)
        summary += f"; kết luận: {verdict}"
        outputs += (("Kết luận", verdict),)
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
        outputs=outputs,
        summary=summary,
        values={"ket_qua": evaluation.value},
        version=ARITHMETIC_VERSION,
        source=source,
    )


def date_result(
    request: ArithmeticRequest, evaluation: DateEvaluation, *, source: str
) -> CalculationResult:
    """The block for a date expression: a day count with its unit, or a date."""

    if isinstance(evaluation.value, Decimal):
        shown = display(evaluation.value, request.unit or "ngày")
        short = _short_value(evaluation.value)
        values = {"ket_qua": evaluation.value}
        formula_text: tuple[str, ...] = (DATE_SPAN_TEXT,)
    else:
        shown = short = evaluation.value.strftime("%d/%m/%Y")
        values = {}
        formula_text = (DATE_SHIFT_TEXT,)
    return CalculationResult(
        formula_id=ARITHMETIC_ID,
        title=f"{DATE_TITLE}: {request.label}",
        formula_text=formula_text,
        warnings=(DATE_SHIFT_NOTE,) if evaluation.dropped_inclusive else (),
        inputs=(
            *(
                (f"Ngày {index}", date.strftime("%d/%m/%Y"))
                for index, date in enumerate(evaluation.dates, start=1)
            ),
            *(
                (f"Số hạng {index}", plain(operand))
                for index, operand in enumerate(evaluation.operands, start=1)
            ),
        ),
        steps=(
            Step(
                label=request.label,
                symbolic=evaluation.expression,
                substituted=f"{evaluation.expression} = {short}",
                value=values.get("ket_qua", Decimal(0)),
                display=shown,
            ),
        ),
        outputs=((request.label, shown),),
        summary=f"{request.label}: **{shown}**",
        values=values,
        version=ARITHMETIC_VERSION,
        source=source,
    )


def arithmetic_results(
    pairs: Sequence[tuple[ArithmeticRequest, Evaluation]], *, source: str
) -> CalculationResult:
    """One block for every quantity the question asks ("tổng tiền phải trả và tổng tiền
    được giảm", 29/10/2026): a line and an output per quantity."""

    if len(pairs) == 1:
        return arithmetic_result(*pairs[0], source=source)
    single = [
        arithmetic_result(request, evaluation, source=source) for request, evaluation in pairs
    ]
    labels = ", ".join(request.label for request, _ in pairs)
    return CalculationResult(
        formula_id=ARITHMETIC_ID,
        title=f"{ARITHMETIC_TITLE}: {labels}",
        formula_text=single[0].formula_text,
        inputs=tuple(
            (f"Số hạng {index}", plain(operand))
            for index, operand in enumerate(
                (o for _, evaluation in pairs for o in evaluation.operands), start=1
            )
        ),
        steps=tuple(step for result in single for step in result.steps),
        outputs=tuple(output for result in single for output in result.outputs),
        summary="; ".join(result.summary for result in single),
        values={
            f"ket_qua_{index}": evaluation.value
            for index, (_, evaluation) in enumerate(pairs, start=1)
        },
        version=ARITHMETIC_VERSION,
        source=source,
    )
