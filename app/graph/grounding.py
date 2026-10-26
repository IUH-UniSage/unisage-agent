"""Post-generation grounding checks: do the numbers and citations in an answer come from
somewhere the model was shown?

Retrieval and rerank decide what the model sees; nothing before this point checks what
it then *writes*. With no LLM rerank in the loop (`CHAT_RERANK_MODE=none|cross_encoder`,
UNISAGE-106) the prompt's "answer only from the context" rule is the last line, so this
module adds a deterministic check behind it:

- `cited_markers`: which `[n]` the answer uses, and which point outside the sources shown.
- `ungrounded_numbers`: numbers in the answer that appear neither in the sources it cites
  (or, with no citation, in any source shown), nor in the question / known values, and
  cannot be reached from those numbers with one arithmetic step or a sum. Those are the
  numbers a reader cannot verify against a document.

The answer has already been streamed when this runs, so the graph cannot take it back: it
appends a short caution and raises an AI-admin warning (`ANSWER_UNGROUNDED_NUMBERS`,
`ANSWER_NO_CITATION`, `CALC_UNGROUNDED_INPUT`). The checks are heuristics tuned to catch
invented figures, not to prove correctness; `tests/graph/test_grounding.py` pins the rules.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.rag.prompting.citations import cited_indexes

# A number as Vietnamese documents write it: 1.234.567 / 1,5 / 38.350.000 / 7,9 / 2025.
_NUMBER = re.compile(r"(?<![\w/])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w/])")
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
# Small integers are list numbers, ordinals, counts ("2 học phần") far more often than
# invented figures; a fabricated fee or score is almost never below this.
SMALL_INTEGER_MAX = 12
# Years are read from questions ("năm 2025") and headings; they are not figures to verify.
_YEAR_MIN, _YEAR_MAX = 1990, 2100
_SUM_MAX_TERMS = 6


@dataclass(frozen=True)
class CitationCheck:
    cited: list[int]  # 1-based markers that exist among the sources
    invalid: list[int]  # markers pointing outside the sources shown


@dataclass(frozen=True)
class GroundingReport:
    ungrounded: list[str]  # numbers as written, first appearance order
    cited: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    no_citation: bool = False

    @property
    def clean(self) -> bool:
        return not self.ungrounded and not self.invalid_markers


def check_citations(answer: str, source_count: int) -> CitationCheck:
    cited = cited_indexes(answer, source_count)
    invalid: list[int] = []
    for match in _MARKER.finditer(answer):
        for raw in match.group(1).split(","):
            index = int(raw)
            if not 1 <= index <= source_count and index not in invalid:
                invalid.append(index)
    return CitationCheck(cited=cited, invalid=invalid)


def _to_float(token: str) -> float | None:
    text = token
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", text):
        text = text.replace(".", "").replace(",", ".")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def numbers_in(text: str) -> dict[str, float]:
    """Numeric tokens of `text` (as written -> value), dates and citation markers removed."""

    cleaned = _MARKER.sub(" ", _DATE.sub(" ", text))
    found: dict[str, float] = {}
    for match in _NUMBER.finditer(cleaned):
        token = match.group(1)
        value = _to_float(token)
        if value is not None and token not in found:
            found[token] = value
    return found


def _verifiable(token: str, value: float) -> bool:
    if value.is_integer() and value <= SMALL_INTEGER_MAX:
        return False
    if value.is_integer() and _YEAR_MIN <= value <= _YEAR_MAX and len(token) == 4:
        return False
    return True


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.051, abs(b) * 0.0005)


def _derivable(value: float, grounded: Sequence[float]) -> bool:
    """One arithmetic step from two grounded numbers (reuse allowed), a sum of up to
    `_SUM_MAX_TERMS` of them, or a percentage of one - the shapes a calculation answer
    legitimately produces."""

    pool = sorted(set(grounded))
    for a, b in itertools.combinations_with_replacement(pool, 2):
        candidates = [a + b, a - b, b - a, a * b, a * b / 100]
        if b:
            candidates.append(a / b)
        if a:
            candidates.append(b / a)
        if any(_close(value, c) for c in candidates):
            return True
    terms = [g for g in pool if 0 < g <= value * 1.0001][:30]
    for n in range(3, _SUM_MAX_TERMS + 1):
        for combo in itertools.combinations_with_replacement(terms, n):
            if _close(value, sum(combo)):
                return True
    return False


def ungrounded_numbers(
    answer: str,
    sources: Iterable[str],
    *,
    question: str = "",
    known: Mapping[str, object] | None = None,
) -> list[str]:
    """Numbers written in `answer` that no source, the question or a known value contains
    and that cannot be derived from them. Order of first appearance."""

    source_values: set[float] = set()
    for text in [*sources, question, *(str(v) for v in (known or {}).values())]:
        source_values.update(numbers_in(text).values())
    grounded = sorted(source_values)
    out: list[str] = []
    # Two passes: a number derived from sources (a subtotal) can ground the next one
    # (a remainder) whichever order the answer states them in.
    pending = {
        token: value for token, value in numbers_in(answer).items() if _verifiable(token, value)
    }
    for _ in range(2):
        for token, value in list(pending.items()):
            if any(_close(value, g) for g in grounded) or _derivable(value, grounded):
                grounded.append(value)
                pending.pop(token)
    for token in numbers_in(answer):
        if token in pending:
            out.append(token)
    return out


def check_answer(
    answer: str,
    source_texts: Sequence[str],
    *,
    question: str = "",
    known: Mapping[str, object] | None = None,
) -> GroundingReport:
    """`source_texts[i]` is source `[i+1]` as shown to the model. Numbers are checked
    against the cited sources when the answer cites any, else against every source."""

    citations = check_citations(answer, len(source_texts))
    scope = (
        [source_texts[i - 1] for i in citations.cited] if citations.cited else list(source_texts)
    )
    return GroundingReport(
        ungrounded=ungrounded_numbers(answer, scope, question=question, known=known),
        cited=citations.cited,
        invalid_markers=citations.invalid,
        no_citation=not citations.cited and bool(source_texts),
    )


REFUSAL_HINTS = (
    "không tìm thấy",
    "không có thông tin",
    "chưa có thông tin",
    "không đủ thông tin",
    "không đủ dữ liệu",
    "không thể trả lời",
    "tạo ticket",
    "gửi yêu cầu hỗ trợ",
)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(hint in lowered for hint in REFUSAL_HINTS)


def caution_line(ungrounded: Sequence[str]) -> str:
    shown = ", ".join(ungrounded[:5]) + (" ..." if len(ungrounded) > 5 else "")
    return (
        f"\n\n> Lưu ý: con số {shown} không tìm thấy trong tài liệu được trích dẫn, "
        "bạn kiểm tra lại với đơn vị phụ trách trước khi dùng."
    )
