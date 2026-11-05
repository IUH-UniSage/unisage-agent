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
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.rag.prompting.citations import cited_indexes

# A number as Vietnamese documents write it: 1.234.567 / 1,5 / 38.350.000 / 7,9 / 2025.
# Not a number: digits glued to letters or "/" (K18, 3177/QĐ), or a document code written
# with a hyphen ("3177-QĐ-ĐHCN", 04/11/2026 d0335: flagged while the question cites it).
_NUMBER = re.compile(r"(?<![\w/])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w/]|-[A-ZĐ])")
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_UNIT_SUFFIX = re.compile(r"(\d)(?:đ|%|k|₫)(?![\w])")
# A unit glued to its number is split off, not dropped (03/11/2026, q0076: "lề trái 3,5cm"
# was never read): "5tr" still scales below.
_GLUED_UNIT = re.compile(r"(\d)(cm|mm|km|kg|m|g|ha|tr)(?![\w])", re.IGNORECASE)
# "5 triệu" is 5.000.000 (03/11/2026, q0075); a range scales both ends ("5 - 7 triệu").
_SCALE = re.compile(
    r"(?:\s*[-\u2013]\s*\d+(?:[.,]\d+)?)?\s*(triệu|tr|tỷ|tỉ|nghìn|ngàn)(?![\w])", re.IGNORECASE
)
_SCALES = {"triệu": 1e6, "tr": 1e6, "tỷ": 1e9, "tỉ": 1e9, "nghìn": 1e3, "ngàn": 1e3}
# A cohort code is an identifier, not a figure to verify or compute with (03/11/2026,
# q0073, q0071: "K18" in the question, "khóa 18" in the answer). "K18" is never read
# (a letter precedes it); "khóa 18" / "Khoá K14" are removed.
_COHORT = re.compile(r"\bkh(?:óa|oá)\s+(?:k\s*)?\d{1,2}(?![\w/]|[.,]\d)", re.IGNORECASE)
_PERCENT = re.compile(r"(?<![\w/.,])(\d+(?:[.,]\d+)?)\s*%")
# Phone numbers and extensions are identifiers, not figures to verify: "028 3588 2679",
# "0283 894 0390", "Ext: 3894", "máy lẻ 112".
_PHONE = re.compile(
    r"\b0\d{1,3}[ .]?\d{3,4}[ .]?\d{3,4}\b|(?:ext|máy lẻ|số lẻ)\s*:?\s*\d{2,5}", re.IGNORECASE
)
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


_DATE_FULL = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_TIME = re.compile(r"\b(\d{1,2})[h:](\d{2})\b")


def _date_time_spans(texts: Iterable[str], *, multiples: bool = True) -> list[float]:
    """Day counts between any two dates and hour/minute spans between any two clock times
    written in `texts`: what "quá hạn 21 ngày", "730 ngày", "13 giờ 10 phút" come from.
    `multiples=False` (strict grounding) keeps the date and clock spans - no code computes
    those - but not the calendar multiples (2 năm → 730 ngày), which are arithmetic."""

    import datetime as dt

    dates: list[dt.date] = []
    minutes: list[int] = []
    for text in texts:
        for d, m, y in _DATE_FULL.findall(text):
            try:
                dates.append(dt.date(int(y), int(m), int(d)))
            except ValueError:
                continue
        for h, mi in _TIME.findall(text):
            if int(h) < 24 and int(mi) < 60:
                minutes.append(int(h) * 60 + int(mi))
    out: list[float] = []
    # "thời hạn 02 năm" -> 730 ngày / 24 tháng; a deadline = a date shifted by n years.
    small = {
        int(v)
        for text in texts
        for v in numbers_in(text).values()
        if v.is_integer() and 1 <= v <= 10
    }
    if multiples:
        for n in small:
            out += [n * 365.0, n * 366.0, n * 12.0, n * 30.0, n * 7.0]
    shifted = []
    for date in set(dates):
        for n in small:
            for sign in (1, -1):
                try:
                    shifted.append(date.replace(year=date.year + sign * n))
                except ValueError:
                    continue
    dates = [*dates, *shifted]
    for first, second in itertools.combinations(sorted(set(dates)), 2):
        days = (second - first).days
        out += [float(days), float(days + 1), float(days - 1), days / 7, days / 30, days / 365]
    for start, end in itertools.combinations(sorted(set(minutes)), 2):
        span = end - start
        out += [float(span), span / 60, float(span // 60), float(span % 60)]
    return out


def numbers_in(text: str) -> dict[str, float]:
    """Numeric tokens of `text` (as written -> value), dates and citation markers removed."""

    found: dict[str, float] = {}
    for token, value, _ in _tokens(text):
        if token not in found:
            found[token] = value
    return found


def _clean(text: str) -> str:
    cleaned = _identifiers_removed(text)
    cleaned = _GLUED_UNIT.sub(r"\1 \2", cleaned)  # 3,5cm -> 3,5 cm ; 5tr -> 5 tr
    return _UNIT_SUFFIX.sub(r"\1 ", cleaned)  # 250.000đ -> 250.000 ; 50% -> 50


def _identifiers_removed(text: str) -> str:
    """Dates, [n] markers, phone numbers and cohort codes blanked out."""

    return _PHONE.sub(" ", _MARKER.sub(" ", _DATE.sub(" ", _COHORT.sub(" ", text))))


def _tokens(text: str) -> list[tuple[str, float, float | None]]:
    """(token as written, value, bare figure of a scaled token or None): "5 triệu" ->
    ("5 triệu", 5000000, 5)."""

    cleaned = _clean(text)
    out: list[tuple[str, float, float | None]] = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group(1)
        value = _to_float(token)
        if value is None:
            continue
        scale = _SCALE.match(cleaned, match.end())
        if scale is None:
            out.append((token, value, None))
            continue
        word = scale.group(1)
        out.append((f"{token} {word}", value * _SCALES[word.lower()], value))
    return out


def stated_values(text: str, *, percent_fractions: bool = False) -> set[float]:
    """Every value `text` states, for a source: its numbers and the bare figure of a scaled
    one ("5" of "5 triệu"). `percent_fractions`: also the fraction a percentage is ("80%"
    states 0,8 - an operand of "15 * 0.8", 03/11/2026, q0073, q0059); not for the answer
    check, where 0,5 next to every grounded figure would make half-steps derivable."""

    values: set[float] = set()
    for _, value, bare in _tokens(text):
        values.add(value)
        if bare is not None:
            values.add(bare)
    if not percent_fractions:
        return values
    for match in _PERCENT.finditer(_identifiers_removed(text)):
        percent = _to_float(match.group(1))
        if percent is not None:
            values.add(percent / 100)
    return values


def _verifiable(token: str, value: float) -> bool:
    if value.is_integer() and value <= SMALL_INTEGER_MAX:
        return False
    if value.is_integer() and _YEAR_MIN <= value <= _YEAR_MAX and len(token) == 4:
        return False
    return True


def _close(a: float, b: float) -> bool:
    # Half a rounding step for grades (7,9 vs 7,85) plus float noise; never a relative
    # window - 0,05 % of 500 triệu is 250 nghìn, wide enough to "derive" any invented fee.
    return abs(a - b) <= 0.051 or abs(a - b) <= abs(b) * 1e-6


class _Lookup:
    """Membership with the `_close` tolerance in O(1): keys rounded to 0,01 and probed at
    the value and ±0,05."""

    def __init__(self, values: Sequence[float]) -> None:
        self._keys = {round(v, 2) for v in values}

    def has(self, value: float, *, exact: bool = False) -> bool:
        """`exact`: no rounding window - for multiplicative relations, where a ±0,05 window
        lets 609.878.000 / 11.957.000 = 51,006 pass as "51 x a fee" in a number-rich table."""

        if value != value or abs(value) > 1e15:  # NaN / absurd
            return False
        if exact:
            return round(value, 2) in self._keys and any(
                abs(value - k) <= abs(k) * 1e-6 + 1e-9 for k in (round(value, 2),)
            )
        return any(round(value + d, 2) in self._keys for d in (0.0, 0.05, -0.05, 0.01, -0.01))


def _derivable(value: float, grounded: Sequence[float], in_answer: Sequence[float]) -> bool:
    """One arithmetic step from two grounded numbers (a+b, a-b, a*b, a/b, a% of b), or a
    sum of 3-`_SUM_MAX_TERMS` numbers that the *answer itself* states and that are grounded
    ("3 + 4 + 2 + 3 = 12"). Sums are not drawn from the whole source pool: a fee table holds
    hundreds of figures and almost any amount is some sum of six of them. O(pool) per value
    thanks to the lookup, so a table-heavy context stays cheap."""

    pool = sorted(set(grounded))
    lookup = _Lookup(pool)
    for a in pool:
        if lookup.has(value - a) or lookup.has(a - value):
            return True
        if a and (lookup.has(value / a, exact=True) or lookup.has(value * 100 / a, exact=True)):
            return True
    terms = sorted({g for g in in_answer if 0 < g <= value * 1.0001})[:12]
    for n in range(3, _SUM_MAX_TERMS + 1):
        for combo in itertools.combinations_with_replacement(terms, n):
            if _close(value, sum(combo)):
                return True
    return False


_STEPS = (0.1, 0.25, 0.5)


def _step_roundings(question: str, texts: Iterable[str]) -> list[float]:
    """The question's numbers rounded to the nearest step a source or the question writes
    ("làm tròn đến 0,5": 7,6 -> 7,5; 8,8 -> 9,0; 29/10/2026). Only those values: a figure
    rounded the wrong way stays ungrounded. No code tool rounds, so strict mode keeps it."""

    written = {v for text in texts for v in numbers_in(text).values()}
    steps = [step for step in _STEPS if any(_close(step, v) for v in written)]
    out: list[float] = []
    for value in numbers_in(question).values():
        for step in steps:
            # Half up, the way the rules round ("từ 0,25 đến dưới 0,75 -> 0,5").
            out.append(math.floor(value / step + 0.5 + 1e-9) * step)
    return out


def _parts(question: str) -> list[float]:
    """Whole and fractional parts of the question's decimals: "7,6 có phần lẻ 0,6"
    (03/11/2026, d0314). Direct matches only, so no window opens for derivations."""

    out: list[float] = []
    for value in numbers_in(question).values():
        if not value.is_integer():
            whole = math.floor(value)
            out += [float(whole), round(value - whole, 6)]
    return out


def _source_pool(texts: Sequence[str], question: str, *, allow_derived: bool) -> set[float]:
    pool: set[float] = set()
    for text in texts:
        pool.update(stated_values(text))
    pool.update(_date_time_spans(texts, multiples=allow_derived))
    pool.update(_step_roundings(question, texts))
    return pool


def states_grounded(
    answer: str, value: float, sources: Sequence[str], *, question: str = ""
) -> bool:
    """`answer` writes `value` and the strict check finds that value itself in the
    sources (or a date span / rounding of them) - not derived, not merely unchecked.
    Lets a refused calc request stay silent when the answer already states its result
    (03/11/2026, q0055)."""

    if not any(_close(value, v) for v in numbers_in(answer).values()):
        return False
    pool = _source_pool([*sources, question], question, allow_derived=False)
    return any(_close(value, g) for g in pool)


def ungrounded_numbers(
    answer: str,
    sources: Iterable[str],
    *,
    question: str = "",
    known: Mapping[str, object] | None = None,
    allow_derived: bool = True,
) -> list[str]:
    """Numbers written in `answer` that no source, the question or a known value contains
    and that cannot be derived from them. Order of first appearance. `allow_derived=False`
    (strict): a sum or difference the model worked out itself is ungrounded too - for a
    turn whose arithmetic belongs to code (`calc_request`, a deferred calculation task),
    where the model was told not to state results of its own (26/10/2026)."""

    texts = [*sources, question, *(str(v) for v in (known or {}).values())]
    grounded = sorted(_source_pool(texts, question, allow_derived=allow_derived))
    parts = _parts(question)
    out: list[str] = []
    # Two passes: a number derived from sources (a subtotal) can ground the next one
    # (a remainder) whichever order the answer states them in.
    pending = {
        token: value for token, value in numbers_in(answer).items() if _verifiable(token, value)
    }
    answer_values = list(numbers_in(answer).values())
    for _ in range(2):
        for token, value in list(pending.items()):
            in_answer = [v for v in answer_values if any(_close(v, g) for g in grounded)]
            if any(_close(value, g) for g in [*grounded, *parts]) or (
                allow_derived and _derivable(value, grounded, in_answer)
            ):
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
    strict: bool = False,
) -> GroundingReport:
    """`source_texts[i]` is source `[i+1]` as shown to the model. Numbers are checked
    against the cited sources when the answer cites any, else against every source.
    `strict`: numbers the model derived itself count as ungrounded (see
    `ungrounded_numbers`)."""

    citations = check_citations(answer, len(source_texts))
    scope = (
        [source_texts[i - 1] for i in citations.cited] if citations.cited else list(source_texts)
    )
    return GroundingReport(
        ungrounded=ungrounded_numbers(
            answer, scope, question=question, known=known, allow_derived=not strict
        ),
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
