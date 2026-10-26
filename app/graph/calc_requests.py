"""`calc_request` blocks of the advisory answer: the model asks for a plain arithmetic
result over figures it cited instead of computing it in its head (26/10/2026).

Two independent decisions, as the policy asks: (1) permission - only a plain `+ - × ÷`
expression with a label and a unit, at most `REQUESTS_MAX` per answer; (2) evidence -
every operand must be a number the question or one of the cited documents states.
A request that fails either is not computed: the answer keeps the figures it quoted,
a fixed line tells the student the sum was not computed, and AI admins get a warning.
The computed block is Python-rendered (`render_markdown`) and appended after the
answer, so no number in it comes from the model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from app.calculation.arithmetic import (
    ArithmeticError,
    ArithmeticRequest,
    arithmetic_result,
    evaluate,
    parse_request,
)
from app.calculation.formulas import plain
from app.calculation.render import render_markdown
from app.graph.grounding import numbers_in
from app.graph.streaming_state import AdminWarning
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REQUESTS_MAX = 3
REJECTED_LINE = (
    '_Mình chưa tính được "{label}" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số'
    " giúp mình nhé._"
)
_REJECTED = "Yêu cầu tính của câu trả lời bị từ chối: "


def arithmetic_blocks(
    requests: Sequence[dict[str, Any]],
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[AdminWarning]]:
    """Markdown to append after the answer (empty when nothing was requested) and the
    AI-admin warnings for rejected requests."""

    parts: list[str] = []
    warnings: list[AdminWarning] = []
    for raw in list(requests)[:REQUESTS_MAX]:
        request = parse_request(raw)
        if request is None:
            warnings.append(_warning("khối calc_request thiếu expression/label"))
            continue
        try:
            evaluation = evaluate(request.expression)
        except ArithmeticError as exc:
            parts.append(REJECTED_LINE.format(label=request.label))
            warnings.append(_warning(f"`{request.expression}`: {exc}"))
            continue
        cited = [
            chunks[index - 1] for index in request.sources if 1 <= index <= len(chunks)
        ] or list(chunks)
        missing = missing_operands(evaluation.operands, [question, *(c.content for c in cited)])
        if missing:
            parts.append(REJECTED_LINE.format(label=request.label))
            warnings.append(
                _warning(
                    f"`{request.expression}`: số {', '.join(plain(m) for m in missing)} không có"
                    " trong câu hỏi hay tài liệu được trích"
                )
            )
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
            plain(evaluation.value),
            markers,
        )
        parts.append(render_markdown(arithmetic_result(request, evaluation, source=source)))
    if not parts:
        return "", warnings
    return "\n\n" + "\n\n".join(parts), warnings


def _warning(message: str) -> AdminWarning:
    return AdminWarning(code="CALC_REQUEST_REJECTED", message=f"{_REJECTED}{message}")


def missing_operands(operands: Sequence[Decimal], texts: Sequence[str]) -> list[Decimal]:
    """Operands none of `texts` states (as written: 1.200.000 and 1200000 both count).
    The evidence half of the policy - shared by the node (the student's own numbers) and
    the advisory requests (cited documents)."""

    stated: set[Decimal] = set()
    for text in texts:
        for value in numbers_in(text or "").values():
            stated.add(Decimal(str(value)))
    return [operand for operand in operands if not any(_same(operand, v) for v in stated)]


def _same(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= Decimal("0.005") or (b != 0 and abs(a - b) / abs(b) <= Decimal("0.000001"))


def request_label(request: ArithmeticRequest) -> str:
    return request.label
