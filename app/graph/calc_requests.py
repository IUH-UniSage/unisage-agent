"""`calc_request` blocks of the advisory answer: the model asks for a plain arithmetic
result over figures it cited instead of computing it in its head (26/10/2026).

Two independent decisions, as the policy asks: (1) permission - only a plain `+ - * /`
expression with a label and a unit, at most `REQUESTS_MAX` per answer; (2) evidence -
every operand must be a number the question or one of the cited documents states.
A request that fails either is not computed: the answer keeps the figures it quoted,
a fixed line tells the student the sum was not computed, and AI admins get a warning.
The computed block is Python-rendered (`render_markdown`) and appended after the
answer, so no number in it comes from the model.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.calculation.arithmetic import (
    ArithmeticError,
    ArithmeticRequest,
    DateEvaluation,
    Evaluation,
    arithmetic_result,
    date_result,
    evaluate,
    evaluate_dates,
    has_date,
    parse_request,
)
from app.calculation.formulas import plain
from app.calculation.render import render_markdown
from app.calculation.result import CalculationResult
from app.graph.grounding import stated_values, states_grounded
from app.graph.streaming_state import AdminWarning
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REQUESTS_MAX = 3
REJECTED_LINE = (
    '_Mình chưa tính được "{label}" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số'
    " giúp mình nhé._"
)
_REJECTED = "Yêu cầu tính của câu trả lời bị từ chối: "
# A date as documents write it: 13/08/2026, 20/6/2018, 01-04-2019, 01.04.2019.
_BARE_NUMBER = re.compile(r"\s*\(?\s*\d+(?:[.,]\d+)*\s*\)?\s*")
_DATE_IN_TEXT = re.compile(r"(?<!\d)(\d{1,2})([/.-])(\d{1,2})\2(\d{4})(?!\d)")


@dataclass(frozen=True)
class CalcBlocks:
    markdown: str  # to append after the answer; empty when nothing was requested
    warnings: list[AdminWarning]
    values: list[Decimal]  # results computed by code (the grounding check accepts them)


def arithmetic_blocks(
    requests: Sequence[dict[str, Any]],
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[AdminWarning]]:
    """Markdown to append after the answer (empty when nothing was requested) and the
    AI-admin warnings for rejected requests."""

    blocks = calc_blocks(requests, question=question, chunks=chunks)
    return blocks.markdown, blocks.warnings


def calc_blocks(
    requests: Sequence[dict[str, Any]],
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    answer: str = "",
) -> CalcBlocks:
    """`arithmetic_blocks` with the computed values. A request may use the result of an
    earlier one as an operand ("374 - 45" after "16 + 358", 29/10/2026). `answer`: the
    answer text, so a refused request whose result it already states (and the documents
    ground) does not get a refusal line contradicting it (03/11/2026, q0055)."""

    parts: list[str] = []
    warnings: list[AdminWarning] = []
    values: list[Decimal] = []
    for raw in list(requests)[:REQUESTS_MAX]:
        request = parse_request(raw)
        if request is None:
            warnings.append(_warning("khối calc_request thiếu expression/label"))
            continue
        if _BARE_NUMBER.fullmatch(request.expression):
            # "7.5": a result the model worked out itself, not a calculation - no block
            # ("6 = 6") and no refusal line; the answer text is checked like any other.
            warnings.append(_warning(f"`{request.expression}` chỉ là một số, không có phép tính"))
            continue
        evaluation: Evaluation | DateEvaluation
        try:
            if has_date(request.expression):
                evaluation = evaluate_dates(request.expression)
            else:
                evaluation = evaluate(request.expression)
        except ArithmeticError as exc:
            parts.append(REJECTED_LINE.format(label=request.label))
            warnings.append(_warning(f"`{request.expression}`: {exc}"))
            continue
        cited = [
            chunks[index - 1] for index in request.sources if 1 <= index <= len(chunks)
        ] or list(chunks)
        texts = [question, *(c.content for c in cited)]
        missing: list[Decimal | dt.date] = []
        if isinstance(evaluation, DateEvaluation):
            # "+ 1" counts both ends of a span: a constant, not a figure to cite.
            numbers = [o for o in evaluation.operands if o != 1]
            missing += missing_operands(numbers, texts, known=values)
            stated = dates_in(texts)
            missing += [date for date in evaluation.dates if date not in stated]
        else:
            missing += missing_operands(evaluation.operands, texts, known=values)
        if missing:
            shown = ", ".join(_plain(m) for m in missing)
            message = (
                f"`{request.expression}`: số {shown} không có trong câu hỏi hay tài liệu được trích"
            )
            if _answer_states_result(evaluation, answer, texts, question):
                # Nothing to add: the line would deny a figure the answer already states.
                message += "; câu trả lời đã nêu kết quả có trong tài liệu, không hiện dòng từ chối"
            else:
                parts.append(REJECTED_LINE.format(label=request.label))
            warnings.append(_warning(message))
            logger.warning(
                "calc_request.rejected expression=%r missing=%s", request.expression, missing
            )
            continue
        markers = [index for index in request.sources if 1 <= index <= len(chunks)]
        source = (
            "số liệu trích từ tài liệu " + " ".join(f"[{index}]" for index in markers)
            if markers
            else "số liệu trong câu hỏi và tài liệu đã trích"
        )
        logger.info(
            "calc_request.computed expression=%r operands=%s value=%s sources=%s",
            evaluation.expression,
            [plain(o) for o in evaluation.operands],
            _plain(evaluation.value),
            markers,
        )
        result: CalculationResult
        if isinstance(evaluation, DateEvaluation):
            result = date_result(request, evaluation, source=source)
        else:
            condition = request.condition
            if condition is not None and missing_operands([condition.threshold], texts):
                # A verdict needs the threshold a document (or the question) states.
                warnings.append(
                    _warning(
                        f"`{request.expression}`: ngưỡng {plain(condition.threshold)} không có"
                        " trong câu hỏi hay tài liệu được trích, bỏ phần kết luận"
                    )
                )
                condition = None
            result = arithmetic_result(request, evaluation, source=source, condition=condition)
        parts.append(render_markdown(result))
        if isinstance(evaluation.value, Decimal):
            values.append(evaluation.value)  # a date is not a figure the check reads
    if not parts:
        return CalcBlocks("", warnings, values)
    return CalcBlocks("\n\n" + "\n\n".join(parts), warnings, values)


def _warning(message: str) -> AdminWarning:
    return AdminWarning(code="CALC_REQUEST_REJECTED", message=f"{_REJECTED}{message}")


def _plain(value: Decimal | dt.date) -> str:
    return plain(value) if isinstance(value, Decimal) else value.strftime("%d/%m/%Y")


def _answer_states_result(
    evaluation: Evaluation | DateEvaluation, answer: str, texts: Sequence[str], question: str
) -> bool:
    value = evaluation.value
    if not answer or not isinstance(value, Decimal):
        return False
    return states_grounded(answer, float(value), texts[1:], question=question)


def dates_in(texts: Sequence[str]) -> set[dt.date]:
    """Dates `texts` write (day first), as date literals of a request must be."""

    out: set[dt.date] = set()
    for text in texts:
        for day, _, month, year in _DATE_IN_TEXT.findall(text or ""):
            try:
                out.add(dt.date(int(year), int(month), int(day)))
            except ValueError:
                continue
    return out


def missing_operands(
    operands: Sequence[Decimal], texts: Sequence[str], *, known: Sequence[Decimal] = ()
) -> list[Decimal]:
    """Operands none of `texts` states (as written: 1.200.000 and 1200000 both count) and
    that are not one of `known` (results code already computed this turn). The evidence
    half of the policy - shared by the node (the student's own numbers) and the advisory
    requests (cited documents)."""

    stated: set[Decimal] = set(known)
    for text in texts:
        for value in stated_values(text or "", percent_fractions=True):
            stated.add(Decimal(str(value)))
            if not value.is_integer():
                # The whole part of a stated score: "7.0 + 0.5" rounds 7,6 (03/11/2026, d0314).
                stated.add(Decimal(math.floor(value)))
    return [operand for operand in operands if not any(_same(operand, v) for v in stated)]


def _same(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= Decimal("0.005") or (b != 0 and abs(a - b) / abs(b) <= Decimal("0.000001"))


def request_label(request: ArithmeticRequest) -> str:
    return request.label
