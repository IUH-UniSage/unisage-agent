"""Every academic business rule UniSage calculates lives in this file.

The grade scale, the theory weights, the rounding places and the input limits
are declared here once; nothing else in the codebase may hard-code them. All
arithmetic is `Decimal` (inputs go through `Decimal(str(value))`), rounding is
half-up via `round_half_up` - never Python's banker's `round()`.

Spec: docs/specs/SPEC-calc-engine.md.
"""

import datetime as _dt
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Literal

from app.calculation.result import CalculationInputError, CalculationResult, FieldError, Step

SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("10")
SCORE_MAX_PLACES = 2
COMPONENT_CREDIT_MAX = 10
PRACTICE_SCORES_MAX = 20

COURSE_SCORE_PLACES = 1
# TBtx, ĐLT and ĐTH are rounded to this before the next step uses them.
COMPONENT_PLACES = 1
# Recorded component scores (each TX/TH column, GK, CK) are rounded to the half
# point first (quy chế): fraction < 0.25 -> .0, 0.25..< 0.75 -> .5, >= 0.75 -> 1.
HALF_POINT_LOW = Decimal("0.25")
HALF_POINT_HIGH = Decimal("0.75")
GPA_PLACES = 2
DISPLAY_PLACES = 2

TIMES = "\u00d7"  # multiplication sign, kept out of literals for ruff's RUF001


@dataclass(frozen=True)
class TheoryWeights:
    tx: Decimal
    gk: Decimal
    ck: Decimal


THEORY_WEIGHTS = TheoryWeights(tx=Decimal("0.2"), gk=Decimal("0.3"), ck=Decimal("0.5"))


@dataclass(frozen=True)
class GradeBand:
    min_score: Decimal
    letter: str
    gp4: Decimal


# Looked up on a score already rounded to 0.1, so the bands have no gaps.
GRADE_SCALE: tuple[GradeBand, ...] = (
    GradeBand(Decimal("9.0"), "A+", Decimal("4.0")),
    GradeBand(Decimal("8.5"), "A", Decimal("3.8")),
    GradeBand(Decimal("8.0"), "B+", Decimal("3.5")),
    GradeBand(Decimal("7.0"), "B", Decimal("3.0")),
    GradeBand(Decimal("6.0"), "C+", Decimal("2.5")),
    GradeBand(Decimal("5.5"), "C", Decimal("2.0")),
    GradeBand(Decimal("5.0"), "D+", Decimal("1.5")),
    GradeBand(Decimal("4.0"), "D", Decimal("1.0")),
    GradeBand(Decimal("0"), "F", Decimal("0.0")),
)


# ---------------------------------------------------------------------------
# Numbers and display
# ---------------------------------------------------------------------------


def round_to_half_point(value: Decimal) -> Decimal:
    whole = value.to_integral_value(rounding="ROUND_FLOOR")
    fraction = value - whole
    if fraction < HALF_POINT_LOW:
        return whole
    if fraction < HALF_POINT_HIGH:
        return whole + Decimal("0.5")
    return whole + 1


def round_half_up(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


def plain(value: Decimal) -> str:
    """Exact value without trailing zeros or exponent: 6.950 -> "6.95", 8 -> "8"."""

    text = format(value.normalize(), "f")
    return text


def fmt(value: Decimal) -> str:
    """Display of an intermediate value: exact when it has at most 2 decimals,
    otherwise rounded half-up to 2 decimals with a leading `≈`."""

    rounded = round_half_up(value, DISPLAY_PLACES)
    if rounded == value:
        return plain(value)
    return f"≈ {plain(rounded)}"


def fixed(value: Decimal, places: int) -> str:
    """A rounded result shown with exactly `places` decimals: 3 -> "3.0"."""

    return format(round_half_up(value, places), "f")


def short(value: Decimal) -> str:
    """`fmt` without the `≈` - for an intermediate reused inside a later expression."""

    return plain(round_half_up(value, DISPLAY_PLACES))


def equals(value: Decimal) -> str:
    """Tail of a substituted line: "= 6.95", or "≈ 7.47" when the display is cut."""

    display = fmt(value)
    return display if display.startswith("≈") else f"= {display}"


def before_rounding(raw: Decimal, rounded: Decimal, places: int) -> str:
    """`raw` shown precisely enough that rounding it visibly gives `rounded`.

    Two decimals normally suffice, but 7.449 would show as 7.45 and then "round"
    to 7.4 - so fall back to 4 decimals whenever the short form would mislead.
    """

    # Always at least one digit more than the result, or "≈ 2.04 → 2.04" says nothing.
    for digits in (max(DISPLAY_PLACES, places + 1), 4):
        shown = round_half_up(raw, digits)
        if round_half_up(shown, places) == rounded:
            prefix = "" if shown == raw else "≈ "
            return f"{prefix}{plain(shown)}"
    return plain(raw)


# ---------------------------------------------------------------------------
# Input parsing (collects every error instead of stopping at the first)
# ---------------------------------------------------------------------------


def to_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int | float | str):
        try:
            number = Decimal(str(value).strip().replace(",", "."))
        except InvalidOperation:
            return None
    else:
        return None
    return number if number.is_finite() else None


def _decimal_places(value: Decimal) -> int:
    exponent = value.normalize().as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def parse_score(value: object, field: str, errors: list[FieldError]) -> Decimal | None:
    number = to_decimal(value)
    if number is None:
        errors.append(FieldError(field, "phải là một số"))
        return None
    if not SCORE_MIN <= number <= SCORE_MAX:
        errors.append(FieldError(field, f"bạn nhập {plain(number)}, điểm phải từ 0 đến 10"))
        return None
    if _decimal_places(number) > SCORE_MAX_PLACES:
        errors.append(FieldError(field, "điểm có tối đa 2 chữ số thập phân"))
        return None
    return number


def parse_credits(
    value: object, field: str, errors: list[FieldError], *, minimum: int, maximum: int
) -> int | None:
    number = to_decimal(value)
    if number is None or number != number.to_integral_value():
        errors.append(FieldError(field, "số tín chỉ phải là số nguyên"))
        return None
    credits = int(number)
    if not minimum <= credits <= maximum:
        errors.append(FieldError(field, f"số tín chỉ phải từ {minimum} đến {maximum}"))
        return None
    return credits


def _missing(field: str, errors: list[FieldError]) -> None:
    errors.append(FieldError(field, "còn thiếu"))


# ---------------------------------------------------------------------------
# Grade conversion
# ---------------------------------------------------------------------------


def grade_band(score10: Decimal) -> GradeBand:
    """Band of a score that is already rounded to 0.1."""

    for band in GRADE_SCALE:
        if score10 >= band.min_score:
            return band
    return GRADE_SCALE[-1]


def _band_range(band: GradeBand) -> str:
    index = GRADE_SCALE.index(band)
    lower = fixed(band.min_score, 1) if band.min_score else "0"
    if index == 0:
        return f"[{lower}; 10]"
    upper = fixed(GRADE_SCALE[index - 1].min_score, 1)
    return f"[{lower}; {upper})"


def _conversion_summary(label: str, rounded: Decimal, band: GradeBand) -> str:
    """The single result line of a grade conversion - shown once, never as a step too."""

    return (
        f"{label} **{fixed(rounded, 1)}** thuộc khoảng {_band_range(band)} → "
        f"điểm chữ **{band.letter}** → thang 4 **{fixed(band.gp4, 1)}**"
    )


def grade_conversion(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    raw = params.get("score10")
    score = None if raw is None else parse_score(raw, "score10", errors)
    if raw is None:
        _missing("score10", errors)
    if errors or score is None:
        raise CalculationInputError(errors)

    rounded = round_half_up(score, COURSE_SCORE_PLACES)
    band = grade_band(rounded)
    steps: tuple[Step, ...] = ()
    if rounded != score:
        steps = (
            Step(
                label="Làm tròn",
                symbolic="Làm tròn điểm thang 10 đến 0.1",
                substituted=f"{plain(score)} → {fixed(rounded, 1)}",
                value=rounded,
                display=fixed(rounded, 1),
                note="làm tròn đến 0.1",
            ),
        )
    return CalculationResult(
        formula_id="grade_conversion",
        title="Quy đổi điểm thang 10 sang điểm chữ và thang 4",
        formula_text=("Điểm thang 10 (làm tròn 0.1) → điểm chữ → điểm thang 4 theo bảng quy đổi",),
        inputs=(("Điểm thang 10", plain(score)),),
        steps=steps,
        outputs=(
            ("Điểm thang 10", fixed(rounded, 1)),
            ("Điểm chữ", band.letter),
            ("Thang 4", fixed(band.gp4, 1)),
        ),
        summary=_conversion_summary("Điểm", rounded, band),
        values={"diem10": rounded, "thang4": band.gp4},
    )


# ---------------------------------------------------------------------------
# Course score (học phần tích hợp lý thuyết + thực hành)
# ---------------------------------------------------------------------------

COURSE_SCORE_FORMULA_LATEX = (
    r"\text{TBtx} = \frac{TX_1 + TX_2 + \dots + TX_n}{n}",
    r"\text{ĐLT} = 0{,}2 \times \text{TBtx} + 0{,}3 \times \text{GK} + 0{,}5 \times \text{CK}",
    r"\text{ĐTH} = \frac{TH_1 + TH_2 + \dots + TH_n}{n}",
    r"\text{ĐTKHP} = \frac{\text{ĐLT} \times \text{TCLT} + \text{ĐTH} \times \text{TCTH}}"
    r"{\text{TCLT} + \text{TCTH}}",
)
COURSE_SCORE_FORMULA_TEXT = (
    "Điểm từng cột TX, GK, CK, TH làm tròn đến 0.5 (lẻ dưới 0.25 → 0; từ 0.25 đến dưới 0.75 → 0.5;"
    " từ 0.75 → 1)",
    "TBtx = (TX1 + TX2 + … + TXn) / n, làm tròn đến 0.1 (hoặc TBtx bạn nhập sẵn)",
    f"ĐLT = 20% {TIMES} TBtx + 30% {TIMES} GK + 50% {TIMES} CK, làm tròn đến 0.1",
    "ĐTH = (TH1 + TH2 + … + THn) / n, làm tròn đến 0.1",
    f"ĐTKHP = (ĐLT {TIMES} TCLT + ĐTH {TIMES} TCTH) / (TCLT + TCTH), làm tròn đến 0.1",
)


def _rounded(raw: Decimal) -> tuple[Decimal, str]:
    """A component rounded to 0.1 before the next step uses it, and the tail of its
    substituted line: "= 8.5", or "= 6.95 → 7.0", or "≈ 7.47 → 7.5"."""

    final = round_half_up(raw, COMPONENT_PLACES)
    if raw == final:
        return final, f"= {fixed(final, 1)}"
    shown = before_rounding(raw, final, COMPONENT_PLACES)
    lead = shown if shown.startswith("≈") else f"= {shown}"
    return final, f"{lead} → {fixed(final, 1)}"


def _scores(
    value: object, field: str, label: str, errors: list[FieldError]
) -> list[Decimal] | None:
    """A non-empty list of 0..10 scores (≤ 20), or None after recording the error."""

    if not isinstance(value, list | tuple) or not value:
        errors.append(FieldError(field, f"cần ít nhất một cột {label}"))
        return None
    if len(value) > PRACTICE_SCORES_MAX:
        errors.append(FieldError(field, f"tối đa {PRACTICE_SCORES_MAX} cột {label}"))
        return None
    scores = [parse_score(item, field, errors) for item in value]
    return None if any(score is None for score in scores) else [s for s in scores if s is not None]


def _mean_line(symbol: str, scores: list[Decimal]) -> tuple[Decimal, str]:
    total = sum(scores, Decimal(0))
    final, tail = _rounded(total / len(scores))
    terms = " + ".join(plain(score) for score in scores)
    return final, f"{symbol} = ({terms}) / {len(scores)} = {plain(total)} / {len(scores)} {tail}"


def course_score(params: Mapping[str, object]) -> CalculationResult:
    """ĐTKHP of a theory + practice course. Each recorded component score (TX/TH
    column, GK, CK) is first rounded to the half point; TBtx, ĐLT and ĐTH are each
    rounded to 0.1 before the next step uses them; ĐTKHP is rounded to 0.1 at the end. `tbtx` is
    either the student's own average or the list of TX columns (equal weights)."""

    errors: list[FieldError] = []

    def credits_of(field: str) -> int | None:
        if params.get(field) is None:
            _missing(field, errors)
            return None
        return parse_credits(params[field], field, errors, minimum=0, maximum=COMPONENT_CREDIT_MAX)

    tclt = credits_of("tclt")
    tcth = credits_of("tcth")
    if tclt is not None and tcth is not None and tclt + tcth == 0:
        errors.append(FieldError("tclt", "tổng tín chỉ lý thuyết và thực hành phải lớn hơn 0"))

    # An invalid credit count (None here) leaves it open whether the part is
    # needed: still validate what was given, but don't report the rest missing.
    tx_columns: list[Decimal] | None = None
    tbtx_given: Decimal | None = None
    theory: dict[str, Decimal] = {}
    if tclt is None or tclt > 0:
        raw_tbtx = params.get("tbtx")
        if raw_tbtx is None:
            if tclt:
                _missing("tbtx", errors)
        elif isinstance(raw_tbtx, list | tuple):
            tx_columns = _scores(raw_tbtx, "tbtx", "điểm thường xuyên", errors)
        else:
            tbtx_given = parse_score(raw_tbtx, "tbtx", errors)
        for field in ("gk", "ck"):
            if params.get(field) is None:
                if tclt:
                    _missing(field, errors)
                continue
            score = parse_score(params[field], field, errors)
            if score is not None:
                theory[field] = score

    practice: list[Decimal] | None = None
    if tcth is None or tcth > 0:
        raw_practice = params.get("th")
        if raw_practice is None:
            if tcth:
                _missing("th", errors)
        else:
            practice = _scores(raw_practice, "th", "điểm thực hành", errors)

    if errors or tclt is None or tcth is None:
        raise CalculationInputError(errors)

    steps: list[Step] = []
    inputs: list[tuple[str, str]] = [
        ("Tín chỉ lý thuyết", str(tclt)),
        ("Tín chỉ thực hành", str(tcth)),
    ]

    # Half-point rounding of the recorded component scores, shown only if it changes one.
    changes: list[str] = []

    def half(label: str, value: Decimal) -> Decimal:
        rounded = round_to_half_point(value)
        if rounded != value:
            changes.append(f"{label} {plain(value)} → {fixed(rounded, 1)}")
        return rounded

    if tx_columns is not None:
        tx_columns = [half(f"TX{i}", v) for i, v in enumerate(tx_columns, start=1)]
    theory = {field: half(field.upper(), value) for field, value in theory.items()}
    if practice is not None:
        practice = [half(f"TH{i}", v) for i, v in enumerate(practice, start=1)]
    if changes:
        steps.append(
            Step(
                label="Làm tròn điểm thành phần",
                symbolic=COURSE_SCORE_FORMULA_TEXT[0],
                substituted="; ".join(changes),
                value=Decimal(0),
                display="",
                note="làm tròn đến 0.5",
            )
        )

    dlt = Decimal(0)
    if tclt:
        if tx_columns is not None:
            tbtx, line = _mean_line("TBtx", tx_columns)
            inputs.append(("Các cột thường xuyên", ", ".join(plain(s) for s in tx_columns)))
        else:
            assert tbtx_given is not None
            tbtx, _ = _rounded(tbtx_given)
            line = f"TBtx = {plain(tbtx_given)}" + (
                f" → {fixed(tbtx, 1)}" if tbtx != tbtx_given else ""
            )
            inputs.append(("Điểm thường xuyên (TBtx)", plain(tbtx_given)))
        steps.append(
            Step(
                label="Điểm thường xuyên",
                symbolic=COURSE_SCORE_FORMULA_TEXT[1],
                substituted=line,
                value=tbtx,
                display=fixed(tbtx, 1),
            )
        )
        weights = THEORY_WEIGHTS
        gk, ck = theory["gk"], theory["ck"]
        parts = (weights.tx * tbtx, weights.gk * gk, weights.ck * ck)
        dlt, tail = _rounded(sum(parts, Decimal(0)))
        inputs += [("Điểm giữa kỳ", plain(gk)), ("Điểm cuối kỳ", plain(ck))]
        steps.append(
            Step(
                label="Điểm lý thuyết",
                symbolic=COURSE_SCORE_FORMULA_TEXT[2],
                substituted=(
                    f"ĐLT = {plain(weights.tx)} {TIMES} {fixed(tbtx, 1)}"
                    f" + {plain(weights.gk)} {TIMES} {plain(gk)}"
                    f" + {plain(weights.ck)} {TIMES} {plain(ck)} = "
                    + " + ".join(plain(part) for part in parts)
                    + f" {tail}"
                ),
                value=dlt,
                display=fixed(dlt, 1),
            )
        )

    dth = Decimal(0)
    if tcth:
        assert practice is not None
        dth, line = _mean_line("ĐTH", practice)
        inputs.append(("Điểm thực hành", ", ".join(plain(score) for score in practice)))
        steps.append(
            Step(
                label="Điểm thực hành",
                symbolic=COURSE_SCORE_FORMULA_TEXT[3],
                substituted=line,
                value=dth,
                display=fixed(dth, 1),
            )
        )

    if tclt and tcth:
        numerator = dlt * tclt + dth * tcth
        final, tail = _rounded(numerator / (tclt + tcth))
        substituted = (
            f"ĐTKHP = ({fixed(dlt, 1)} {TIMES} {tclt} + {fixed(dth, 1)} {TIMES} {tcth})"
            f" / ({tclt} + {tcth}) = {plain(numerator)} / {tclt + tcth} {tail}"
        )
    elif tclt:
        final = dlt
        substituted = f"ĐTKHP = ĐLT = {fixed(dlt, 1)} (học phần chỉ có lý thuyết)"
    else:
        final = dth
        substituted = f"ĐTKHP = ĐTH = {fixed(dth, 1)} (học phần chỉ có thực hành)"
    steps.append(
        Step(
            label="Điểm tổng kết học phần",
            symbolic=COURSE_SCORE_FORMULA_TEXT[4],
            substituted=substituted,
            value=final,
            display=fixed(final, 1),
            note="làm tròn đến 0.1",
        )
    )
    band = grade_band(final)

    return CalculationResult(
        formula_id="course_score",
        title="Điểm tổng kết học phần (lý thuyết + thực hành)",
        formula_text=COURSE_SCORE_FORMULA_TEXT,
        formula_latex=COURSE_SCORE_FORMULA_LATEX,
        inputs=tuple(inputs),
        steps=tuple(steps),
        outputs=(
            ("ĐTKHP", fixed(final, 1)),
            ("Điểm chữ", band.letter),
            ("Thang 4", fixed(band.gp4, 1)),
        ),
        summary=_conversion_summary("ĐTKHP", final, band),
        values={"dtkhp": final, "thang4": band.gp4},
    )


# ---------------------------------------------------------------------------
# GPA (thang 4)
# ---------------------------------------------------------------------------

GPA_COURSES_MAX = 30
COURSE_CREDIT_MIN = 1
COURSE_CREDIT_MAX = 10
GPA_FORMULA_LATEX = (
    r"\text{GPA} = \frac{\sum_i \text{điểm thang 4}_i \times \text{TC}_i}{\sum_i \text{TC}_i}",
)
GPA_FORMULA_TEXT = (
    f"Điểm chất lượng của môn = điểm hệ 4 {TIMES} số tín chỉ",
    f"GPA = Σ (điểm hệ 4 {TIMES} số tín chỉ) / Σ số tín chỉ, làm tròn đến 0.01",
)


def _letter_band(value: object) -> GradeBand | None:
    if not isinstance(value, str):
        return None
    letter = value.strip().upper()
    return next((band for band in GRADE_SCALE if band.letter == letter), None)


def gpa(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    raw_courses = params.get("courses")
    if raw_courses is None:
        _missing("courses", errors)
        raise CalculationInputError(errors)
    if not isinstance(raw_courses, list | tuple) or not raw_courses:
        raise CalculationInputError([FieldError("courses", "cần ít nhất một môn")])
    if len(raw_courses) > GPA_COURSES_MAX:
        raise CalculationInputError(
            [FieldError("courses", f"tối đa {GPA_COURSES_MAX} môn mỗi lần tính")]
        )

    rows: list[tuple[str, int, str, GradeBand]] = []
    for index, raw in enumerate(raw_courses, start=1):
        if not isinstance(raw, Mapping):
            errors.append(FieldError("courses", f"dòng {index} không hợp lệ"))
            continue
        name = str(raw.get("name") or f"Môn {index}").strip()[:80]
        row_errors: list[FieldError] = []
        credits = parse_credits(
            raw.get("credits"),
            "courses",
            row_errors,
            minimum=COURSE_CREDIT_MIN,
            maximum=COURSE_CREDIT_MAX,
        )
        score_raw = raw.get("score")
        band = _letter_band(score_raw)
        score_text = band.letter if band else ""
        if band is None:
            score = parse_score(score_raw, "courses", row_errors)
            if score is not None:
                rounded = round_half_up(score, COURSE_SCORE_PLACES)
                band = grade_band(rounded)
                rounding = "" if rounded == score else f" → {fixed(rounded, 1)}"
                score_text = f"{plain(score)}{rounding} → {band.letter}"
        if row_errors or credits is None or band is None:
            errors += [
                FieldError("courses", f"dòng {index} ({name}): {error.reason}")
                for error in row_errors
            ]
            continue
        rows.append((name, credits, score_text, band))
    if errors:
        raise CalculationInputError(errors)

    steps: list[Step] = []
    total_quality = Decimal(0)
    total_credits = 0
    for name, credits, score_text, band in rows:
        quality = band.gp4 * credits
        total_quality += quality
        total_credits += credits
        steps.append(
            Step(
                label=name,
                symbolic=GPA_FORMULA_TEXT[0],
                substituted=(
                    f"{score_text} → {fixed(band.gp4, 1)}; "
                    f"{fixed(band.gp4, 1)} {TIMES} {credits} = {plain(quality)}"
                ),
                value=quality,
                display=plain(quality),
            )
        )

    raw_gpa = total_quality / total_credits
    final = round_half_up(raw_gpa, GPA_PLACES)
    qualities = " + ".join(step.display for step in steps)
    credit_sum = " + ".join(str(credits) for _, credits, _, _ in rows)
    if final == raw_gpa:
        tail = f"= {fixed(final, GPA_PLACES)}"
    else:
        shown = before_rounding(raw_gpa, final, GPA_PLACES)
        lead = shown if shown.startswith("≈") else f"= {shown}"
        tail = f"{lead} → {fixed(final, GPA_PLACES)}"
    steps.append(
        Step(
            label="Điểm trung bình",
            symbolic=GPA_FORMULA_TEXT[1],
            substituted=(
                f"GPA = ({qualities}) / ({credit_sum}) = {plain(total_quality)} / {total_credits}"
                f" {tail}"
            ),
            value=final,
            display=fixed(final, GPA_PLACES),
            note="làm tròn đến 0.01",
        )
    )
    return CalculationResult(
        formula_id="gpa",
        title="Điểm trung bình (GPA) thang 4",
        formula_text=GPA_FORMULA_TEXT,
        formula_latex=GPA_FORMULA_LATEX,
        inputs=tuple(
            (name, f"{credits} TC, điểm {score_text.split(' → ')[0]}")
            for name, credits, score_text, _ in rows
        ),
        steps=tuple(steps),
        outputs=(("GPA", fixed(final, 2)), ("Tổng tín chỉ", str(total_credits))),
        values={"gpa": final, "tong_tin_chi": Decimal(total_credits)},
    )


# ---------------------------------------------------------------------------
# Tuition (học phí theo tín chỉ)
# ---------------------------------------------------------------------------

TUITION_CREDITS_MAX = 300
TUITION_PRICE_MAX = Decimal(100_000_000)
TuitionScope = Literal["tuition_semester", "tuition_year", "tuition_program"]
TUITION_SCOPES: dict[TuitionScope, str] = {
    "tuition_semester": "một học kỳ",
    "tuition_year": "một năm học",
    "tuition_program": "cả khoá học",
}
TUITION_FORMULA_LATEX = (
    r"\text{Học phí} = \text{TC}_{LT} \times \text{đơn giá}_{LT}"
    r" + \text{TC}_{TH} \times \text{đơn giá}_{TH}",
)
TUITION_FORMULA_TEXT = (
    f"Học phí = TC lý thuyết {TIMES} đơn giá 1 TC lý thuyết"
    f" + TC thực hành {TIMES} đơn giá 1 TC thực hành",
    "Đơn giá theo khoá tuyển sinh và giữ nguyên suốt khoá học; chỉ khoá mới có đơn giá khác",
)
TUITION_WARNINGS = ("Chưa gồm học phần Giáo dục quốc phòng - an ninh (đơn giá riêng), nếu có.",)
TUITION_PERIOD_MAX = 60


def money(value: Decimal) -> str:
    """Whole đồng with dot thousands: 38350000 -> "38.350.000 đồng"."""

    whole = int(round_half_up(value, 0))
    return f"{whole:,}".replace(",", ".") + " đồng"


_MONEY_TEXT = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(đồng|đ|vnđ|vnd)?", re.IGNORECASE)


def parse_money(value: object, field: str, errors: list[FieldError]) -> Decimal | None:
    """An amount in whole đồng. Text must be written out ("980000", "980.000 đ"): a dot
    or comma is a thousands separator, and "1,6 triệu" is refused rather than misread."""

    if isinstance(value, str):
        match = _MONEY_TEXT.fullmatch(value.strip())
        number = Decimal(re.sub(r"\D", "", match.group(1))) if match else None
    else:
        number = to_decimal(value)
    if number is None or number != number.to_integral_value():
        errors.append(FieldError(field, "số tiền phải là số nguyên (đồng)"))
        return None
    if not Decimal(0) <= number <= TUITION_PRICE_MAX:
        errors.append(FieldError(field, f"số tiền phải từ 0 đến {money(TUITION_PRICE_MAX)}"))
        return None
    return number


def tuition_period(scope: TuitionScope, params: Mapping[str, object]) -> str:
    """The period in the student's own words ("2 năm đầu") when the extractor copied
    one, else the formula's default ("một năm học"). No formula covers "N năm", so the
    student enters the credits of the whole period and the title has to say so."""

    raw = params.get("pham_vi")
    if not isinstance(raw, str):
        return TUITION_SCOPES[scope]
    # `*` would break the bold title it is rendered into.
    text = " ".join(raw.replace("*", "").split())[:TUITION_PERIOD_MAX]
    return text or TUITION_SCOPES[scope]


def _tuition(scope: TuitionScope) -> Callable[[Mapping[str, object]], CalculationResult]:
    def compute(params: Mapping[str, object]) -> CalculationResult:
        errors: list[FieldError] = []
        credits: dict[str, int | None] = {}
        prices: dict[str, Decimal | None] = {}
        for name in ("tc_lt", "tc_th"):
            raw = params.get(name)
            if raw is None:
                _missing(name, errors)
                credits[name] = None
            else:
                credits[name] = parse_credits(
                    raw, name, errors, minimum=0, maximum=TUITION_CREDITS_MAX
                )
        for name in ("don_gia_lt", "don_gia_th"):
            raw = params.get(name)
            if raw is None:
                _missing(name, errors)
                prices[name] = None
            else:
                prices[name] = parse_money(raw, name, errors)
        lt, th = credits["tc_lt"], credits["tc_th"]
        price_lt, price_th = prices["don_gia_lt"], prices["don_gia_th"]
        if not errors and lt == 0 and th == 0:
            errors.append(FieldError("tc_lt", "tổng tín chỉ lý thuyết và thực hành phải lớn hơn 0"))
        if errors or lt is None or th is None or price_lt is None or price_th is None:
            raise CalculationInputError(errors)

        theory = lt * price_lt
        practice = th * price_th
        total = theory + practice
        title = f"Học phí {tuition_period(scope, params)}"
        return CalculationResult(
            formula_id=scope,
            title=title,
            formula_text=TUITION_FORMULA_TEXT,
            formula_latex=TUITION_FORMULA_LATEX,
            inputs=(
                ("TC lý thuyết", str(lt)),
                ("TC thực hành", str(th)),
                ("Đơn giá 1 TC lý thuyết", money(price_lt)),
                ("Đơn giá 1 TC thực hành", money(price_th)),
            ),
            steps=(
                Step(
                    label="Học phí lý thuyết",
                    symbolic=f"TC lý thuyết {TIMES} đơn giá lý thuyết",
                    substituted=f"{lt} {TIMES} {money(price_lt)} = {money(theory)}",
                    value=theory,
                    display=money(theory),
                ),
                Step(
                    label="Học phí thực hành",
                    symbolic=f"TC thực hành {TIMES} đơn giá thực hành",
                    substituted=f"{th} {TIMES} {money(price_th)} = {money(practice)}",
                    value=practice,
                    display=money(practice),
                ),
                Step(
                    label="Tổng",
                    symbolic="Học phí lý thuyết + học phí thực hành",
                    substituted=f"{money(theory)} + {money(practice)} = {money(total)}",
                    value=total,
                    display=money(total),
                ),
            ),
            outputs=((title, money(total)),),
            warnings=TUITION_WARNINGS,
            values={"hoc_phi": total},
        )

    return compute


# ---------------------------------------------------------------------------
# Parameters per formula, and dispatch
# ---------------------------------------------------------------------------

ParamKind = Literal["number", "number_list", "number_or_list", "course_table", "text"]
FormulaId = Literal[
    "gpa",
    "course_score",
    "grade_conversion",
    "tuition_semester",
    "tuition_year",
    "tuition_program",
    "admission_score_2026",
    "transcript_score_3hk",
    "weighted_score",
    "date_span",
    "time_span",
]


def _always(_: Mapping[str, object]) -> bool:
    return True


def _never(_: Mapping[str, object]) -> bool:
    return False


def _unless_zero(credit_field: str) -> Callable[[Mapping[str, object]], bool]:
    """Needed unless that credit count is known to be 0 - an unknown count asks
    for the scores in the same round rather than costing a second one."""

    def required(params: Mapping[str, object]) -> bool:
        credits = to_decimal(params.get(credit_field))
        return credits is None or credits != 0

    return required


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    tab_label: str
    kind: ParamKind
    required: Callable[[Mapping[str, object]], bool] = _always
    min: Decimal | None = None
    max: Decimal | None = None
    step: Decimal | None = None
    unit: str | None = None
    max_items: int | None = None
    # What to search the documents for before the student is asked; None = the
    # student's own value (their scores, the credits they registered).
    lookup: str | None = None
    # The label once other params are known (see `label_for`); None = always `label`.
    label_from: Callable[[Mapping[str, object]], str] | None = None


def _score_spec(name: str, label: str, tab_label: str, credit_field: str) -> ParamSpec:
    return ParamSpec(
        name=name,
        label=label,
        tab_label=tab_label,
        kind="number",
        required=_unless_zero(credit_field),
        min=SCORE_MIN,
        max=SCORE_MAX,
        step=Decimal("0.01"),
    )


def _credit_spec(name: str, label: str, tab_label: str) -> ParamSpec:
    return ParamSpec(
        name=name,
        label=label,
        tab_label=tab_label,
        kind="number",
        min=Decimal(0),
        max=Decimal(COMPONENT_CREDIT_MAX),
        step=Decimal(1),
        unit="TC",
    )


def _tuition_params(scope: TuitionScope) -> tuple[ParamSpec, ...]:
    """Theory and practice credits are always two separate questions. Credits of a whole
    programme come from its curriculum document; those of a semester or a year are the
    student's own. The unit prices come from the fee schedule. `khoa_nhap_hoc` and
    `khoi_nganh` are not computed with: they pick the fee schedule row, and are asked
    only when the documents hold several rows. `pham_vi` is never asked nor computed
    with: it only names the period in the labels and the title."""

    credit_lookup = (
        "tổng số tín chỉ lý thuyết, thực hành của chương trình đào tạo"
        if scope == "tuition_program"
        else None
    )

    def credit_label(kind: str) -> Callable[[Mapping[str, object]], str]:
        return lambda known: f"Số tín chỉ {kind} của {tuition_period(scope, known)}"

    period = TUITION_SCOPES[scope]
    credits = [
        ParamSpec(
            name=name,
            label=f"Số tín chỉ {kind} của {period}",
            label_from=credit_label(kind),
            tab_label=f"TC {kind}",
            kind="number",
            min=Decimal(0),
            max=Decimal(TUITION_CREDITS_MAX),
            step=Decimal(1),
            unit="TC",
            lookup=credit_lookup,
        )
        for name, kind in (("tc_lt", "lý thuyết"), ("tc_th", "thực hành"))
    ]
    prices = [
        ParamSpec(
            name=name,
            label=f"Đơn giá 1 tín chỉ {kind} (đồng)",
            tab_label=f"Đơn giá {short_kind}",
            kind="number",
            min=Decimal(0),
            max=TUITION_PRICE_MAX,
            step=Decimal(1),
            unit="đồng",
            lookup="mức thu học phí 01 tín chỉ môn lý thuyết, môn thực hành",
        )
        for name, kind, short_kind in (
            ("don_gia_lt", "lý thuyết", "LT"),
            ("don_gia_th", "thực hành", "TH"),
        )
    ]
    context = [
        ParamSpec(
            name="khoa_nhap_hoc",
            label="Bạn nhập học năm nào (VD 2025)",
            tab_label="Năm nhập học",
            kind="text",
            required=_never,
        ),
        ParamSpec(
            name="khoi_nganh",
            label="Ngành của bạn thuộc khối nào (VD Khối Công nghệ)",
            tab_label="Khối ngành",
            kind="text",
            required=_never,
        ),
        ParamSpec(
            name="pham_vi",
            label="Phạm vi học phí đúng như người dùng nói (VD 2 năm đầu)",
            tab_label="Phạm vi",
            kind="text",
            required=_never,
        ),
    ]
    return (*credits, *prices, *context)


@dataclass(frozen=True)
class OutputSpec:
    """One output of a formula - what a target question names ("để ĐXT đạt 24")."""

    key: str
    label: str
    money: bool = False  # shown as whole đồng (học phí), not as a score


@dataclass(frozen=True)
class Formula:
    """A registered, versioned calculation. `version` is "<năm văn bản>.<lần sửa>" for a
    formula bound to one document and "1", "2"... for a standing rule; `source` names
    the document the rule was taken from. Both are shown under the formula lines and
    recorded in the trace, so an answer can always be tied to the rule it used."""

    formula_id: FormulaId
    version: str
    source: str
    title: str
    description: str
    params: tuple[ParamSpec, ...]
    outputs: tuple[OutputSpec, ...]
    compute: Callable[[Mapping[str, object]], CalculationResult]
    # The rounding the formula applies to a param before using it (0.5 for a recorded
    # score), given the other params; None = none. The solver searches on that grid.
    rounding_of: Callable[[Mapping[str, object], str], Decimal | None] | None = None
    # Public link to the source document; empty = the source is shown as plain text.
    source_url: str = ""


def _when_positive(field: str) -> Callable[[Mapping[str, object]], bool]:
    """Needed only once another param is a positive number (ĐTK once a ĐGNL result is;
    "0" there means the student did not sit the test)."""

    def required(params: Mapping[str, object]) -> bool:
        value = to_decimal(params.get(field))
        return value is not None and value > 0

    return required


def _optional_number(
    params: Mapping[str, object],
    name: str,
    errors: list[FieldError],
    *,
    minimum: Decimal,
    maximum: Decimal,
    default: Decimal | None,
) -> Decimal | None:
    """A number in [minimum, maximum]; `default` when not given (None = required)."""

    raw = params.get(name)
    if raw is None:
        if default is None:
            _missing(name, errors)
        return default
    number = to_decimal(raw)
    if number is None:
        errors.append(FieldError(name, "phải là một số"))
        return None
    if not minimum <= number <= maximum:
        errors.append(
            FieldError(
                name,
                f"bạn nhập {plain(number)}, giá trị phải từ {plain(minimum)} đến {plain(maximum)}",
            )
        )
        return None
    if _decimal_places(number) > SCORE_MAX_PLACES:
        errors.append(FieldError(name, "tối đa 2 chữ số thập phân"))
        return None
    return number


# Sources shown under every result and kept in the trace (`Formula.source`).
COURSE_SCORE_SOURCE = (
    "Quy chế đào tạo đại học chính quy theo tín chỉ của Trường (điểm học phần, GPA, quy đổi"
    " điểm; quy tắc chốt 09/10/2026)"
)
TUITION_SOURCE = (
    "Quyết định mức thu học phí theo năm học của Trường (biểu phí theo khoá, khối ngành)"
)


# ---------------------------------------------------------------------------
# Admission score 2026 (điểm xét tuyển đại học chính quy, thang 30)
# ---------------------------------------------------------------------------

ADMISSION_2026_VERSION = "2026.1"
ADMISSION_2026_SOURCE = "Công thức tính điểm xét tuyển đại học chính quy năm 2026 của Trường"
ADMISSION_2026_SOURCE_URL = (
    "https://tuyensinh.iuh.edu.vn/HoSoTuyenSinh/cong-thuc-xet-tuyen-iuh-2026.pdf"
)
ADMISSION_2026_TITLE = "Điểm xét tuyển đại học chính quy 2026 (thang 30)"
ADMISSION_SCALE = Decimal(30)
ADMISSION_WEIGHT_DK = Decimal("0.7")
ADMISSION_WEIGHT_DHB = Decimal("0.3")
APTITUDE_SCORE_MAX = Decimal(1200)
# The top score of the 1200-point test is never below this: a smaller "ĐTK" read from a
# document (the "30" of "thang điểm 30") is a misread, not a top score.
APTITUDE_TOP_MIN = Decimal(600)
BONUS_MAX = Decimal(10)
ADMISSION_2026_FORMULA_LATEX = (
    r"\text{ĐĐGNL} = \frac{\text{Kết quả thi ĐGNL} \times 30}{\text{ĐTK}}",
    r"\text{ĐK} = \max(\text{ĐTN},\ \text{ĐĐGNL})",
    r"\text{XT1} = 0{,}7 \times \text{ĐK} + 0{,}3 \times \text{ĐHB} + \text{Đ(Kv;Đt)} + \text{ĐC}",
    r"\text{XT2} = \text{ĐTN} + \text{Đ(Kv;Đt)} + \text{ĐC}",
    r"\text{XT3} = \text{ĐĐGNL} + \text{Đ(Kv;Đt)} + \text{ĐC}",
    r"\text{ĐXT} = \max(\text{XT1},\ \text{XT2},\ \text{XT3})",
)
ADMISSION_2026_FORMULA_TEXT = (
    f"ĐĐGNL = kết quả thi ĐGNL {TIMES} 30 / ĐTK (ĐTK: điểm thủ khoa kỳ thi ĐGNL năm 2026)",
    "ĐK = Max(ĐTN, ĐĐGNL); ĐTN: tổng điểm 3 môn thi tốt nghiệp THPT trong tổ hợp xét tuyển",
    f"XT1 = 0.7 {TIMES} ĐK + 0.3 {TIMES} ĐHB + Đ(Kv;Đt) + ĐC;"
    " ĐHB: tổng điểm học bạ 3 môn trong tổ hợp",
    "XT2 = ĐTN + Đ(Kv;Đt) + ĐC; XT3 = ĐĐGNL + Đ(Kv;Đt) + ĐC",
    "ĐXT = Max(XT1, XT2, XT3), thang điểm 30",
)
ADMISSION_2026_FORMULA_NOTES = (
    "ĐTN: tổng điểm 3 môn thi tốt nghiệp THPT trong tổ hợp; ĐHB: tổng điểm học bạ 3 môn trong"
    " tổ hợp; ĐTK: điểm thủ khoa kỳ thi ĐGNL năm 2026; Đ(Kv;Đt): điểm ưu tiên khu vực, đối tượng;"
    " ĐC: điểm cộng",
    "Thang điểm 30",
)
ADMISSION_2026_WARNINGS = (
    "Đ(Kv;Đt) là điểm ưu tiên khu vực/đối tượng theo quy định của Bộ GD&ĐT (từ 22,5 điểm trở"
    " lên mức ưu tiên giảm dần) - kết quả dùng đúng mức bạn đã nhập.",
    "ĐC là điểm cộng (điểm thưởng, điểm xét thưởng, điểm khuyến khích) theo quy định của Trường.",
)
ADMISSION_2026_OUTPUTS = (
    OutputSpec("ddgnl", "điểm ĐGNL quy đổi về thang 30"),
    OutputSpec("dk", "ĐK = Max(ĐTN, ĐĐGNL)"),
    OutputSpec("xt1", "XT1 (kết hợp điểm thi và học bạ)"),
    OutputSpec("xt2", "XT2 (điểm thi tốt nghiệp)"),
    OutputSpec("xt3", "XT3 (điểm ĐGNL quy đổi)"),
    OutputSpec("dxt", "điểm xét tuyển ĐXT = Max(XT1, XT2, XT3)"),
)


def _score30_spec(name: str, label: str, tab_label: str) -> ParamSpec:
    return ParamSpec(
        name=name,
        label=label,
        tab_label=tab_label,
        kind="number",
        min=SCORE_MIN,
        max=ADMISSION_SCALE,
        step=Decimal("0.01"),
    )


ADMISSION_2026_PARAMS: tuple[ParamSpec, ...] = (
    _score30_spec("dtn", "Tổng điểm 3 môn thi tốt nghiệp THPT trong tổ hợp (ĐTN)", "ĐTN"),
    _score30_spec("dhb", "Tổng điểm học bạ 3 môn trong tổ hợp (ĐHB)", "ĐHB"),
    ParamSpec(
        name="dgnl",
        label="Kết quả thi đánh giá năng lực (nhập 0 nếu không dự thi)",
        tab_label="ĐGNL",
        kind="number",
        min=SCORE_MIN,
        max=APTITUDE_SCORE_MAX,
        step=Decimal("0.01"),
    ),
    ParamSpec(
        name="dtk",
        label="Điểm thủ khoa kỳ thi đánh giá năng lực năm 2026 (ĐTK, thang 1200)",
        tab_label="ĐTK",
        kind="number",
        required=_when_positive("dgnl"),
        lookup="điểm thủ khoa kỳ thi đánh giá năng lực năm 2026 điểm cao nhất",
        min=APTITUDE_TOP_MIN,
        max=APTITUDE_SCORE_MAX,
        step=Decimal("0.01"),
    ),
    ParamSpec(
        name="uu_tien",
        label=(
            "Điểm ưu tiên khu vực/đối tượng Đ(Kv;Đt), đã quy đổi theo quy định của Bộ"
            " (nhập 0 nếu không có)"
        ),
        tab_label="Ưu tiên",
        kind="number",
        min=SCORE_MIN,
        max=BONUS_MAX,
        step=Decimal("0.01"),
    ),
    ParamSpec(
        name="diem_cong",
        label="Điểm cộng ĐC (thưởng, xét thưởng, khuyến khích; nhập 0 nếu không có)",
        tab_label="Điểm cộng",
        kind="number",
        min=SCORE_MIN,
        max=BONUS_MAX,
        step=Decimal("0.01"),
    ),
)


def admission_score_2026(params: Mapping[str, object]) -> CalculationResult:
    """ĐXT 2026: the aptitude result is first converted to the 30-point scale, the
    better of exam total and converted aptitude score feeds XT1, and ĐXT is the best
    of XT1/XT2/XT3. No rounding rule is published, so the values are exact and shown
    with ≤ 2 decimals (`fmt`)."""

    errors: list[FieldError] = []
    dtn = _optional_number(
        params, "dtn", errors, minimum=SCORE_MIN, maximum=ADMISSION_SCALE, default=None
    )
    dhb = _optional_number(
        params, "dhb", errors, minimum=SCORE_MIN, maximum=ADMISSION_SCALE, default=None
    )
    dgnl = _optional_number(
        params, "dgnl", errors, minimum=SCORE_MIN, maximum=APTITUDE_SCORE_MAX, default=Decimal(0)
    )
    dgnl = None if dgnl is not None and dgnl <= 0 else dgnl  # 0 = did not sit the test
    dtk: Decimal | None = None
    if dgnl is not None:
        dtk = _optional_number(
            params,
            "dtk",
            errors,
            minimum=APTITUDE_TOP_MIN,
            maximum=APTITUDE_SCORE_MAX,
            default=None,
        )
        if dtk is not None and dgnl > dtk:
            errors.append(
                FieldError(
                    "dtk",
                    f"điểm thủ khoa phải lớn hơn hoặc bằng kết quả thi {plain(dgnl)} bạn đã nêu",
                )
            )
    uu_tien = _optional_number(
        params, "uu_tien", errors, minimum=SCORE_MIN, maximum=BONUS_MAX, default=Decimal(0)
    )
    diem_cong = _optional_number(
        params, "diem_cong", errors, minimum=SCORE_MIN, maximum=BONUS_MAX, default=Decimal(0)
    )
    if errors or dtn is None or dhb is None or uu_tien is None or diem_cong is None:
        raise CalculationInputError(errors)

    text = ADMISSION_2026_FORMULA_TEXT
    extra = uu_tien + diem_cong
    extra_text = f" + {plain(uu_tien)} + {plain(diem_cong)}"
    inputs: list[tuple[str, str]] = [("ĐTN", plain(dtn)), ("ĐHB", plain(dhb))]
    steps: list[Step] = []
    values: dict[str, Decimal] = {}

    ddgnl: Decimal | None = None
    if dgnl is not None and dtk is not None:
        ddgnl = dgnl * ADMISSION_SCALE / dtk
        inputs += [("Kết quả ĐGNL", plain(dgnl)), ("ĐTK", plain(dtk))]
        steps.append(
            Step(
                label="Quy đổi ĐGNL về thang 30",
                symbolic=text[0],
                substituted=f"ĐĐGNL = {plain(dgnl)} {TIMES} 30 / {plain(dtk)} {equals(ddgnl)}",
                value=ddgnl,
                display=fmt(ddgnl),
            )
        )
        values["ddgnl"] = ddgnl
        dk = max(dtn, ddgnl)
        dk_line = f"ĐK = Max({plain(dtn)}; {short(ddgnl)}) = {short(dk)}"
    else:
        dk = dtn
        dk_line = f"ĐK = ĐTN = {plain(dtn)} (không dự thi ĐGNL)"
    inputs += [("Đ(Kv;Đt)", plain(uu_tien)), ("ĐC", plain(diem_cong))]
    steps.append(
        Step(
            label="Điểm thi dùng để xét",
            symbolic=text[1],
            substituted=dk_line,
            value=dk,
            display=short(dk),
        )
    )
    values["dk"] = dk

    part_dk = ADMISSION_WEIGHT_DK * dk
    part_dhb = ADMISSION_WEIGHT_DHB * dhb
    xt1 = part_dk + part_dhb + extra
    steps.append(
        Step(
            label="XT1",
            symbolic=text[2],
            substituted=(
                f"XT1 = 0.7 {TIMES} {short(dk)} + 0.3 {TIMES} {plain(dhb)}{extra_text}"
                f" = {short(part_dk)} + {short(part_dhb)}{extra_text} {equals(xt1)}"
            ),
            value=xt1,
            display=fmt(xt1),
        )
    )
    xt2 = dtn + extra
    steps.append(
        Step(
            label="XT2",
            symbolic=text[3],
            substituted=f"XT2 = {plain(dtn)}{extra_text} {equals(xt2)}",
            value=xt2,
            display=fmt(xt2),
        )
    )
    values.update({"xt1": xt1, "xt2": xt2})
    candidates = [("XT1", xt1), ("XT2", xt2)]
    if ddgnl is not None:
        xt3 = ddgnl + extra
        steps.append(
            Step(
                label="XT3",
                symbolic=text[3],
                substituted=f"XT3 = {short(ddgnl)}{extra_text} {equals(xt3)}",
                value=xt3,
                display=fmt(xt3),
            )
        )
        values["xt3"] = xt3
        candidates.append(("XT3", xt3))
    best_label, dxt = max(candidates, key=lambda item: item[1])
    steps.append(
        Step(
            label="Điểm xét tuyển",
            symbolic=text[4],
            substituted=(
                "ĐXT = Max("
                + "; ".join(short(value) for _, value in candidates)
                + f") = {short(dxt)} ({best_label})"
            ),
            value=dxt,
            display=fmt(dxt),
        )
    )
    values["dxt"] = dxt
    outputs = (*((label, fmt(value)) for label, value in candidates), ("ĐXT", fmt(dxt)))
    return CalculationResult(
        formula_id="admission_score_2026",
        title=ADMISSION_2026_TITLE,
        formula_text=ADMISSION_2026_FORMULA_NOTES,
        formula_latex=ADMISSION_2026_FORMULA_LATEX,
        inputs=tuple(inputs),
        steps=tuple(steps),
        outputs=outputs,
        warnings=ADMISSION_2026_WARNINGS,
        summary=f"ĐXT **{fmt(dxt)}** / 30 (lấy theo {best_label})",
        values=values,
    )


# ---------------------------------------------------------------------------
# Transcript-based admission score (học bạ lớp 10, 11, HK1 lớp 12)
# ---------------------------------------------------------------------------

TRANSCRIPT_VERSION = "2020.1"
TRANSCRIPT_SOURCE = (
    "Bản mô tả chương trình đào tạo đại học chính quy ngành Kiểm toán năm 2020, mục 4.1"
    " (xét tuyển sử dụng kết quả học tập THPT lớp 10, 11 và học kỳ 1 lớp 12)"
)
TRANSCRIPT_TITLE = "Điểm xét tuyển học bạ THPT (lớp 10, lớp 11, học kỳ 1 lớp 12)"
TRANSCRIPT_THRESHOLD = Decimal("19.5")
TRANSCRIPT_SUBJECTS = (1, 2, 3)
TRANSCRIPT_FORMULA_LATEX = (
    r"\text{Điểm môn} = \frac{\text{TB cả năm lớp 10} + \text{TB cả năm lớp 11}"
    r" + \text{TB HK1 lớp 12}}{3}",
    r"\text{Tổng} = \text{Môn}_1 + \text{Môn}_2 + \text{Môn}_3 \ \ge\ \text{ngưỡng nhận hồ sơ}",
)
TRANSCRIPT_FORMULA_TEXT = (
    "Điểm môn = (TB cả năm lớp 10 + TB cả năm lớp 11 + TB học kỳ 1 lớp 12) / 3",
    "Tổng điểm xét tuyển = điểm môn 1 + điểm môn 2 + điểm môn 3 (tổ hợp 3 môn)",
    "Ngưỡng nhận hồ sơ: tổng 3 môn tối thiểu 19.50 (theo bản mô tả CTĐT 2020)",
)
TRANSCRIPT_OUTPUTS = (
    OutputSpec("mon1", "điểm môn 1 (trung bình 3 học kỳ)"),
    OutputSpec("mon2", "điểm môn 2"),
    OutputSpec("mon3", "điểm môn 3"),
    OutputSpec("tong", "tổng điểm 3 môn"),
)
_TRANSCRIPT_TERMS = (
    ("10", "TB cả năm lớp 10"),
    ("11", "TB cả năm lớp 11"),
    ("12", "TB học kỳ 1 lớp 12"),
)


def _subject_name(params: Mapping[str, object], index: int) -> str:
    raw = params.get(f"mon{index}")
    name = " ".join(str(raw).replace("*", "").split())[:40] if isinstance(raw, str) else ""
    return f"Môn {index}" + (f" ({name})" if name else "")


def _transcript_params() -> tuple[ParamSpec, ...]:
    specs: list[ParamSpec] = []
    for index in TRANSCRIPT_SUBJECTS:
        specs.append(
            ParamSpec(
                name=f"mon{index}",
                label=f"Tên môn {index} trong tổ hợp (VD Toán)",
                tab_label=f"Môn {index}",
                kind="text",
                required=_never,
            )
        )
        for suffix, term in _TRANSCRIPT_TERMS:

            def label(known: Mapping[str, object], index: int = index, term: str = term) -> str:
                return f"{_subject_name(known, index)}: {term}"

            specs.append(
                ParamSpec(
                    name=f"m{index}_{suffix}",
                    label=f"Môn {index}: {term}",
                    label_from=label,
                    tab_label=f"M{index} {'HK1 12' if suffix == '12' else 'lớp ' + suffix}",
                    kind="number",
                    min=SCORE_MIN,
                    max=SCORE_MAX,
                    step=Decimal("0.01"),
                )
            )
    specs.append(
        ParamSpec(
            name="nguong",
            label="Ngưỡng nhận hồ sơ (tổng 3 môn), mặc định 19.5",
            tab_label="Ngưỡng",
            kind="number",
            required=_never,
            min=SCORE_MIN,
            max=ADMISSION_SCALE,
            step=Decimal("0.01"),
        )
    )
    return tuple(specs)


TRANSCRIPT_PARAMS = _transcript_params()


def transcript_score_3hk(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    terms: dict[int, list[Decimal]] = {}
    for index in TRANSCRIPT_SUBJECTS:
        scores: list[Decimal] = []
        for suffix, _ in _TRANSCRIPT_TERMS:
            name = f"m{index}_{suffix}"
            if params.get(name) is None:
                _missing(name, errors)
                continue
            score = parse_score(params[name], name, errors)
            if score is not None:
                scores.append(score)
        terms[index] = scores
    threshold = _optional_number(
        params,
        "nguong",
        errors,
        minimum=SCORE_MIN,
        maximum=ADMISSION_SCALE,
        default=TRANSCRIPT_THRESHOLD,
    )
    if errors or threshold is None or any(len(scores) != 3 for scores in terms.values()):
        raise CalculationInputError(errors)

    steps: list[Step] = []
    inputs: list[tuple[str, str]] = []
    values: dict[str, Decimal] = {}
    total = Decimal(0)
    for index in TRANSCRIPT_SUBJECTS:
        scores = terms[index]
        mean = sum(scores, Decimal(0)) / 3
        name = _subject_name(params, index)
        inputs.append((name, " - ".join(plain(score) for score in scores)))
        steps.append(
            Step(
                label=f"Điểm {name.lower() if name.startswith('Môn') else name}",
                symbolic=TRANSCRIPT_FORMULA_TEXT[0],
                substituted=(
                    f"{name} = ({' + '.join(plain(s) for s in scores)}) / 3"
                    f" = {plain(sum(scores, Decimal(0)))} / 3 {equals(mean)}"
                ),
                value=mean,
                display=fmt(mean),
            )
        )
        values[f"mon{index}"] = mean
        total += mean
    inputs.append(("Ngưỡng nhận hồ sơ", plain(threshold)))
    steps.append(
        Step(
            label="Tổng 3 môn",
            symbolic=TRANSCRIPT_FORMULA_TEXT[1],
            substituted=(
                "Tổng = "
                + " + ".join(short(values[f"mon{i}"]) for i in TRANSCRIPT_SUBJECTS)
                + f" {equals(total)}"
            ),
            value=total,
            display=fmt(total),
        )
    )
    values["tong"] = total
    reached = total >= threshold
    verdict = "đủ điều kiện nộp hồ sơ" if reached else "chưa đạt ngưỡng nhận hồ sơ"
    sign = "≥" if reached else "<"
    return CalculationResult(
        formula_id="transcript_score_3hk",
        title=TRANSCRIPT_TITLE,
        formula_text=TRANSCRIPT_FORMULA_TEXT,
        formula_latex=TRANSCRIPT_FORMULA_LATEX,
        inputs=tuple(inputs),
        steps=tuple(steps),
        outputs=(
            *((f"Môn {i}", fmt(values[f"mon{i}"])) for i in TRANSCRIPT_SUBJECTS),
            ("Tổng 3 môn", fmt(total)),
        ),
        summary=f"Tổng 3 môn **{fmt(total)}** {sign} ngưỡng {plain(threshold)} → {verdict}",
        values=values,
    )


# ---------------------------------------------------------------------------
# Weighted course score (trọng số riêng của đề cương học phần)
# ---------------------------------------------------------------------------

WEIGHTED_SOURCE = (
    "Trọng số đánh giá ghi trong đề cương học phần (do người hỏi hoặc tài liệu nêu);"
    " làm tròn 0.1 và điểm chữ theo quy chế của Trường"
)
WEIGHTED_TITLE = "Điểm học phần theo trọng số đề cương"
WEIGHTED_COMPONENTS_MAX = 10
WEIGHT_TOTAL = Decimal(100)
WEIGHT_TOLERANCE = Decimal("0.5")
WEIGHTED_FORMULA_LATEX = (r"\text{Điểm học phần} = \sum_{i} \frac{w_i}{100} \times \text{điểm}_i",)
WEIGHTED_FORMULA_TEXT = (
    f"Điểm học phần = Σ (trọng số i / 100 {TIMES} điểm thành phần i), làm tròn đến 0.1",
    "Điểm thành phần làm tròn theo bước đã nêu (VD 0.5) trước khi nhân trọng số, nếu có",
    "Điểm chữ và thang 4 theo bảng quy đổi của Trường",
)
WEIGHTED_OUTPUTS = (
    OutputSpec("diem_hp", "điểm học phần (thang 10)"),
    OutputSpec("thang4", "điểm thang 4"),
)
WEIGHTED_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        name="trong_so",
        label="Trọng số từng thành phần theo đề cương, đơn vị % (VD 10, 10, 30, 50)",
        tab_label="Trọng số",
        kind="number_list",
        min=SCORE_MIN,
        max=WEIGHT_TOTAL,
        step=Decimal("0.01"),
        unit="%",
        max_items=WEIGHTED_COMPONENTS_MAX,
    ),
    ParamSpec(
        name="diem",
        label="Điểm từng thành phần, cùng thứ tự với trọng số, thang 10",
        tab_label="Điểm",
        kind="number_list",
        min=SCORE_MIN,
        max=SCORE_MAX,
        step=Decimal("0.01"),
        max_items=WEIGHTED_COMPONENTS_MAX,
    ),
    ParamSpec(
        name="lam_tron_thanh_phan",
        label="Bước làm tròn điểm thành phần trước khi tính (VD 0.5); bỏ trống nếu không",
        tab_label="Làm tròn TP",
        kind="number",
        required=_never,
        min=SCORE_MIN,
        max=Decimal(1),
        step=Decimal("0.1"),
    ),
)


def _round_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step == Decimal("0.5"):
        return round_to_half_point(value)
    return (value / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step


def weighted_score(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    raw_weights = params.get("trong_so")
    raw_scores = params.get("diem")
    weights: list[Decimal] = []
    if raw_weights is None:
        _missing("trong_so", errors)
    elif not isinstance(raw_weights, list | tuple) or not raw_weights:
        errors.append(FieldError("trong_so", "cần ít nhất một trọng số"))
    elif len(raw_weights) > WEIGHTED_COMPONENTS_MAX:
        errors.append(FieldError("trong_so", f"tối đa {WEIGHTED_COMPONENTS_MAX} thành phần"))
    else:
        for item in raw_weights:
            number = to_decimal(item)
            if number is None or number < 0:
                errors.append(FieldError("trong_so", "mỗi trọng số phải là một số không âm"))
                break
            weights.append(number)
        else:
            total_weight = sum(weights, Decimal(0))
            if abs(total_weight - 1) <= Decimal("0.001"):  # given as fractions (0.1, 0.3, 0.5)
                weights = [weight * 100 for weight in weights]
                total_weight = sum(weights, Decimal(0))
            if abs(total_weight - WEIGHT_TOTAL) > WEIGHT_TOLERANCE:
                errors.append(
                    FieldError(
                        "trong_so", f"tổng trọng số là {plain(total_weight)} %, phải bằng 100 %"
                    )
                )
    scores: list[Decimal | None] = []
    if raw_scores is None:
        _missing("diem", errors)
    elif not isinstance(raw_scores, list | tuple):
        errors.append(FieldError("diem", "phải là danh sách điểm"))
    else:
        for position, item in enumerate(raw_scores, start=1):
            if item is None:
                errors.append(FieldError("diem", f"còn thiếu điểm thành phần thứ {position}"))
                scores.append(None)
            else:
                scores.append(parse_score(item, "diem", errors))
        if weights and len(scores) != len(weights):
            errors.append(
                FieldError(
                    "diem",
                    f"có {len(weights)} trọng số nhưng {len(scores)} điểm"
                    " - mỗi thành phần đúng một điểm",
                )
            )
    step = _optional_number(
        params,
        "lam_tron_thanh_phan",
        errors,
        minimum=SCORE_MIN,
        maximum=Decimal(1),
        default=Decimal(0),
    )
    if errors or step is None or any(score is None for score in scores):
        raise CalculationInputError(errors)
    given = [score for score in scores if score is not None]

    steps: list[Step] = []
    rounded = given
    if step > 0:
        rounded = [_round_to_step(score, step) for score in given]
        changes = [
            f"TP{i} {plain(before)} → {plain(after)}"
            for i, (before, after) in enumerate(zip(given, rounded, strict=True), start=1)
            if before != after
        ]
        if changes:
            steps.append(
                Step(
                    label="Làm tròn điểm thành phần",
                    symbolic=WEIGHTED_FORMULA_TEXT[1],
                    substituted="; ".join(changes),
                    value=Decimal(0),
                    display="",
                    note=f"làm tròn đến {plain(step)}",
                )
            )
    parts = [weight / 100 * score for weight, score in zip(weights, rounded, strict=True)]
    raw_total = sum(parts, Decimal(0))
    final = round_half_up(raw_total, COURSE_SCORE_PLACES)
    tail = (
        f"= {fixed(final, 1)}"
        if raw_total == final
        else f"= {before_rounding(raw_total, final, COURSE_SCORE_PLACES)} → {fixed(final, 1)}"
    )
    steps.append(
        Step(
            label="Điểm học phần",
            symbolic=WEIGHTED_FORMULA_TEXT[0],
            substituted=(
                "Điểm = "
                + " + ".join(
                    f"{plain(weight)}% {TIMES} {plain(score)}"
                    for weight, score in zip(weights, rounded, strict=True)
                )
                + " = "
                + " + ".join(short(part) for part in parts)
                + f" {tail}"
            ),
            value=final,
            display=fixed(final, 1),
            note="làm tròn đến 0.1",
        )
    )
    band = grade_band(final)
    return CalculationResult(
        formula_id="weighted_score",
        title=WEIGHTED_TITLE,
        formula_text=WEIGHTED_FORMULA_TEXT,
        formula_latex=WEIGHTED_FORMULA_LATEX,
        inputs=tuple(
            (f"TP{i} ({plain(weight)} %)", plain(score))
            for i, (weight, score) in enumerate(zip(weights, given, strict=True), start=1)
        ),
        steps=tuple(steps),
        outputs=(
            ("Điểm học phần", fixed(final, 1)),
            ("Điểm chữ", band.letter),
            ("Thang 4", fixed(band.gp4, 1)),
        ),
        summary=_conversion_summary("Điểm học phần", final, band),
        values={"diem_hp": final, "thang4": band.gp4},
    )


HALF_POINT = Decimal("0.5")


def _course_score_rounding(params: Mapping[str, object], name: str) -> Decimal | None:
    """Recorded component scores (TX columns, GK, CK, TH) are rounded to 0.5; a TBtx
    given as the average is not."""

    if name in ("gk", "ck", "th"):
        return HALF_POINT
    if name == "tbtx" and isinstance(params.get("tbtx"), list | tuple):
        return HALF_POINT
    return None


def _weighted_rounding(params: Mapping[str, object], name: str) -> Decimal | None:
    step = to_decimal(params.get("lam_tron_thanh_phan"))
    return step if name == "diem" and step is not None and step > 0 else None


# ---------------------------------------------------------------------------
# Date span (khoảng ngày giữa hai mốc) and time span (tổng giờ của các khoảng giờ)
# ---------------------------------------------------------------------------

DATE_SPAN_VERSION = "1"
DATE_SPAN_SOURCE = (
    "cách đếm ngày theo lịch (hiệu hai ngày; tính cả ngày đầu và ngày cuối = hiệu + 1)"
)
DATE_SPAN_TITLE = "Khoảng ngày giữa hai mốc"
DATE_SPAN_FORMULA_TEXT = (
    "Số ngày cách nhau = ngày kết thúc \u2212 ngày bắt đầu (theo lịch)",
    "Số ngày tính cả ngày đầu và ngày cuối = số ngày cách nhau + 1",
    "Có thời hạn N năm: mốc bắt đầu được cộng thêm N năm trước khi so với ngày kết thúc",
)
DATE_SPAN_OUTPUTS = (
    OutputSpec("so_ngay", "số ngày cách nhau (không tính ngày đầu)"),
    OutputSpec("ca_hai_dau", "số ngày tính cả ngày đầu và ngày cuối"),
)
DATE_SPAN_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(name="tu_ngay", label="Ngày bắt đầu (dd/mm/yyyy)", tab_label="Từ ngày", kind="text"),
    ParamSpec(
        name="den_ngay", label="Ngày kết thúc (dd/mm/yyyy)", tab_label="Đến ngày", kind="text"
    ),
    ParamSpec(
        name="cong_them_nam",
        label="Thời hạn cộng thêm vào ngày bắt đầu, tính bằng năm (bỏ trống nếu không)",
        tab_label="Hạn (năm)",
        kind="number",
        required=_never,
        min=Decimal(0),
        max=Decimal(20),
        step=Decimal(1),
    ),
)
_DATE_TEXT = re.compile(r"^\s*(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})\s*$")


def parse_date_text(value: object, field: str, errors: list[FieldError]) -> _dt.date | None:
    match = _DATE_TEXT.match(str(value)) if isinstance(value, str) else None
    if match is None:
        errors.append(FieldError(field, "ngày phải viết dạng dd/mm/yyyy"))
        return None
    day, month, year = (int(part) for part in match.groups())
    try:
        return _dt.date(year, month, day)
    except ValueError:
        errors.append(FieldError(field, f"không có ngày {day:02d}/{month:02d}/{year}"))
        return None


def _shift_years(date: _dt.date, years: int) -> _dt.date:
    try:
        return date.replace(year=date.year + years)
    except ValueError:  # 29/02 → 28/02
        return date.replace(year=date.year + years, day=28)


def _date_text(date: _dt.date) -> str:
    return date.strftime("%d/%m/%Y")


def date_span(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    start = end = None
    if params.get("tu_ngay") is None:
        _missing("tu_ngay", errors)
    else:
        start = parse_date_text(params["tu_ngay"], "tu_ngay", errors)
    if params.get("den_ngay") is None:
        _missing("den_ngay", errors)
    else:
        end = parse_date_text(params["den_ngay"], "den_ngay", errors)
    years = _optional_number(
        params, "cong_them_nam", errors, minimum=Decimal(0), maximum=Decimal(20), default=Decimal(0)
    )
    if errors or start is None or end is None or years is None:
        raise CalculationInputError(errors)
    steps: list[Step] = []
    inputs: list[tuple[str, str]] = [
        ("Ngày bắt đầu", _date_text(start)),
        ("Ngày kết thúc", _date_text(end)),
    ]
    anchor = start
    if years > 0:
        anchor = _shift_years(start, int(years))
        inputs.append(("Thời hạn", f"{plain(years)} năm"))
        steps.append(
            Step(
                label="Mốc sau thời hạn",
                symbolic=DATE_SPAN_FORMULA_TEXT[2],
                substituted=f"{_date_text(start)} + {plain(years)} năm = {_date_text(anchor)}",
                value=Decimal(0),
                display=_date_text(anchor),
            )
        )
    days = (end - anchor).days
    if days < 0:
        raise CalculationInputError(
            [
                FieldError(
                    "den_ngay", f"ngày kết thúc {_date_text(end)} trước mốc {_date_text(anchor)}"
                )
            ]
        )
    span = Decimal(days)
    inclusive = span + 1
    steps.append(
        Step(
            label="Số ngày cách nhau",
            symbolic=DATE_SPAN_FORMULA_TEXT[0],
            substituted=f"{_date_text(end)} \u2212 {_date_text(anchor)} = {days} ngày",
            value=span,
            display=str(days),
        )
    )
    steps.append(
        Step(
            label="Tính cả ngày đầu và ngày cuối",
            symbolic=DATE_SPAN_FORMULA_TEXT[1],
            substituted=f"{days} + 1 = {days + 1} ngày",
            value=inclusive,
            display=str(days + 1),
        )
    )
    return CalculationResult(
        formula_id="date_span",
        title=DATE_SPAN_TITLE,
        formula_text=DATE_SPAN_FORMULA_TEXT,
        inputs=tuple(inputs),
        steps=tuple(steps),
        outputs=(("Cách nhau", f"{days} ngày"), ("Tính cả hai đầu", f"{days + 1} ngày")),
        summary=(
            f"Từ {_date_text(anchor)} đến {_date_text(end)}: cách nhau **{days} ngày**;"
            f" tính cả ngày đầu và ngày cuối **{days + 1} ngày**"
        ),
        values={"so_ngay": span, "ca_hai_dau": inclusive},
    )


TIME_SPAN_VERSION = "1"
TIME_SPAN_SOURCE = (
    "cách cộng khoảng giờ theo đồng hồ (giờ kết thúc \u2212 giờ bắt đầu, cộng các khoảng)"
)
TIME_SPAN_TITLE = "Tổng thời gian của các khoảng giờ"
TIME_SPAN_FORMULA_TEXT = (
    "Mỗi khoảng = giờ kết thúc \u2212 giờ bắt đầu (qua nửa đêm thì cộng 24 giờ)",
    "Tổng = cộng các khoảng, đổi ra giờ và phút",
)
TIME_SPAN_OUTPUTS = (
    OutputSpec("tong_gio", "tổng số giờ (thập phân)"),
    OutputSpec("tong_phut", "tổng số phút"),
)
TIME_SPAN_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        name="khoang_gio",
        label="Các khoảng giờ, mỗi khoảng dạng hh:mm-hh:mm, cách nhau bằng dấu chấm phẩy"
        " (VD 06:30-19:40; 06:30-19:40; 06:30-17:40)",
        tab_label="Khoảng giờ",
        kind="text",
    ),
)
_TIME_RANGE = re.compile(
    r"(\d{1,2})\s*[:h]\s*(\d{2})\s*(?:-|\u2013|\u2014|đến|to)\s*(\d{1,2})\s*[:h]\s*(\d{2})",
    re.IGNORECASE,
)
TIME_RANGES_MAX = 20


def time_span(params: Mapping[str, object]) -> CalculationResult:
    errors: list[FieldError] = []
    raw = params.get("khoang_gio")
    if raw is None:
        _missing("khoang_gio", errors)
        raise CalculationInputError(errors)
    ranges = _TIME_RANGE.findall(str(raw))
    if not ranges:
        errors.append(FieldError("khoang_gio", "chưa đọc được khoảng giờ nào dạng hh:mm-hh:mm"))
    if len(ranges) > TIME_RANGES_MAX:
        errors.append(FieldError("khoang_gio", f"tối đa {TIME_RANGES_MAX} khoảng giờ"))
    spans: list[tuple[str, int]] = []
    for h1, m1, h2, m2 in ranges:
        start, end = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
        if int(h1) > 23 or int(h2) > 23 or int(m1) > 59 or int(m2) > 59:
            errors.append(FieldError("khoang_gio", f"giờ {h1}:{m1}-{h2}:{m2} không hợp lệ"))
            continue
        if end < start:
            end += 24 * 60
        spans.append((f"{int(h1):02d}:{m1}-{int(h2):02d}:{m2}", end - start))
    if errors:
        raise CalculationInputError(errors)

    def hm(minutes: int) -> str:
        return f"{minutes // 60} giờ {minutes % 60:02d} phút"

    steps = [
        Step(
            label=f"Khoảng {index}",
            symbolic=TIME_SPAN_FORMULA_TEXT[0],
            substituted=f"{text} = {hm(minutes)}",
            value=Decimal(minutes),
            display=hm(minutes),
        )
        for index, (text, minutes) in enumerate(spans, start=1)
    ]
    total = sum(minutes for _, minutes in spans)
    hours = Decimal(total) / 60
    steps.append(
        Step(
            label="Tổng",
            symbolic=TIME_SPAN_FORMULA_TEXT[1],
            substituted=" + ".join(hm(m) for _, m in spans) + f" = {hm(total)} ({fmt(hours)} giờ)",
            value=hours,
            display=hm(total),
        )
    )
    return CalculationResult(
        formula_id="time_span",
        title=TIME_SPAN_TITLE,
        formula_text=TIME_SPAN_FORMULA_TEXT,
        inputs=tuple((f"Khoảng {i}", text) for i, (text, _) in enumerate(spans, start=1)),
        steps=tuple(steps),
        outputs=(("Tổng", f"{hm(total)} ({fmt(hours)} giờ)"),),
        summary=f"Tổng thời gian **{hm(total)}** (= **{fmt(hours)} giờ**)",
        values={"tong_gio": hours, "tong_phut": Decimal(total)},
    )


FORMULAS: dict[FormulaId, Formula] = {
    "course_score": Formula(
        formula_id="course_score",
        rounding_of=_course_score_rounding,
        version="1",
        source=COURSE_SCORE_SOURCE,
        outputs=(OutputSpec("dtkhp", "ĐTKHP (thang 10)"), OutputSpec("thang4", "điểm thang 4")),
        title="Điểm tổng kết học phần (lý thuyết + thực hành)",
        description="Điểm tổng kết một học phần từ điểm TX/GK/CK và các cột thực hành",
        params=(
            _credit_spec("tclt", "Số tín chỉ lý thuyết của học phần", "TC lý thuyết"),
            _credit_spec("tcth", "Số tín chỉ thực hành của học phần", "TC thực hành"),
            ParamSpec(
                name="tbtx",
                label="Điểm thường xuyên (các cột TX), thang 10",
                tab_label="Điểm TX",
                kind="number_or_list",
                required=_unless_zero("tclt"),
                min=SCORE_MIN,
                max=SCORE_MAX,
                step=Decimal("0.01"),
                max_items=PRACTICE_SCORES_MAX,
            ),
            _score_spec("gk", "Điểm giữa kỳ, thang 10", "Điểm GK", "tclt"),
            _score_spec("ck", "Điểm cuối kỳ, thang 10", "Điểm CK", "tclt"),
            ParamSpec(
                name="th",
                label="Các cột điểm thực hành, thang 10",
                tab_label="Điểm TH",
                kind="number_list",
                required=_unless_zero("tcth"),
                min=SCORE_MIN,
                max=SCORE_MAX,
                step=Decimal("0.01"),
                max_items=PRACTICE_SCORES_MAX,
            ),
        ),
        compute=course_score,
    ),
    "gpa": Formula(
        formula_id="gpa",
        version="1",
        source=COURSE_SCORE_SOURCE,
        outputs=(OutputSpec("gpa", "GPA (thang 4)"),),
        title="Điểm trung bình (GPA) thang 4",
        description="GPA học kỳ hoặc tích lũy từ danh sách môn, số tín chỉ và điểm",
        params=(
            ParamSpec(
                name="courses",
                label="Các môn: số tín chỉ và điểm (thang 10 hoặc điểm chữ)",
                tab_label="Các môn",
                kind="course_table",
                max_items=GPA_COURSES_MAX,
            ),
        ),
        compute=gpa,
    ),
    "grade_conversion": Formula(
        formula_id="grade_conversion",
        version="1",
        source=COURSE_SCORE_SOURCE,
        outputs=(OutputSpec("diem10", "điểm thang 10"), OutputSpec("thang4", "điểm thang 4")),
        title="Quy đổi điểm thang 10 sang điểm chữ và thang 4",
        description="Quy đổi một điểm thang 10 sang điểm chữ và điểm thang 4",
        params=(
            ParamSpec(
                name="score10",
                label="Điểm thang 10 cần quy đổi",
                tab_label="Điểm",
                kind="number",
                min=SCORE_MIN,
                max=SCORE_MAX,
                step=Decimal("0.01"),
            ),
        ),
        compute=grade_conversion,
    ),
    **{
        scope: Formula(
            formula_id=scope,
            version="1",
            source=TUITION_SOURCE,
            outputs=(OutputSpec("hoc_phi", f"học phí {period}", money=True),),
            title=f"Học phí {period}",
            description=(
                f"Học phí {period} từ số tín chỉ lý thuyết, thực hành và đơn giá 1 tín chỉ"
            ),
            params=_tuition_params(scope),
            compute=_tuition(scope),
        )
        for scope, period in TUITION_SCOPES.items()
    },
    "admission_score_2026": Formula(
        formula_id="admission_score_2026",
        version=ADMISSION_2026_VERSION,
        source=ADMISSION_2026_SOURCE,
        source_url=ADMISSION_2026_SOURCE_URL,
        outputs=ADMISSION_2026_OUTPUTS,
        title=ADMISSION_2026_TITLE,
        description=(
            "Điểm xét tuyển đại học chính quy 2026 (thang 30) từ tổng điểm 3 môn thi tốt nghiệp,"
            " điểm ĐGNL (quy đổi thang 30), tổng điểm học bạ 3 môn, điểm ưu tiên và điểm cộng"
        ),
        params=ADMISSION_2026_PARAMS,
        compute=admission_score_2026,
    ),
    "transcript_score_3hk": Formula(
        formula_id="transcript_score_3hk",
        version=TRANSCRIPT_VERSION,
        source=TRANSCRIPT_SOURCE,
        outputs=TRANSCRIPT_OUTPUTS,
        title=TRANSCRIPT_TITLE,
        description=(
            "Điểm xét tuyển học bạ: điểm mỗi môn = (TB cả năm lớp 10 + TB cả năm lớp 11"
            " + TB học kỳ 1 lớp 12) / 3, tổng 3 môn so với ngưỡng nhận hồ sơ"
        ),
        params=TRANSCRIPT_PARAMS,
        compute=transcript_score_3hk,
    ),
    "date_span": Formula(
        formula_id="date_span",
        version=DATE_SPAN_VERSION,
        source=DATE_SPAN_SOURCE,
        outputs=DATE_SPAN_OUTPUTS,
        title=DATE_SPAN_TITLE,
        description=(
            "Số ngày giữa hai mốc ngày (kéo dài bao nhiêu ngày, còn bao nhiêu ngày, quá hạn bao"
            " nhiêu ngày), có hoặc không tính cả ngày đầu và ngày cuối; thời hạn N năm cộng vào"
            " mốc đầu"
        ),
        params=DATE_SPAN_PARAMS,
        compute=date_span,
    ),
    "time_span": Formula(
        formula_id="time_span",
        version=TIME_SPAN_VERSION,
        source=TIME_SPAN_SOURCE,
        outputs=TIME_SPAN_OUTPUTS,
        title=TIME_SPAN_TITLE,
        description=(
            "Tổng giờ và phút của một hay nhiều khoảng giờ hh:mm-hh:mm (lịch làm việc, ca thi)"
        ),
        params=TIME_SPAN_PARAMS,
        compute=time_span,
    ),
    "weighted_score": Formula(
        formula_id="weighted_score",
        rounding_of=_weighted_rounding,
        version="1",
        source=WEIGHTED_SOURCE,
        outputs=WEIGHTED_OUTPUTS,
        title=WEIGHTED_TITLE,
        description=(
            "Điểm học phần theo trọng số riêng của đề cương (VD trắc nghiệm 10 %, giữa kỳ 30 %,"
            " cuối kỳ 50 %) khi khác công thức 20/30/50 của Trường"
        ),
        params=WEIGHTED_PARAMS,
        compute=weighted_score,
    ),
}


def formula_of(formula_id: FormulaId) -> Formula:
    return FORMULAS[formula_id]


def param_spec(formula_id: FormulaId, name: str) -> ParamSpec | None:
    return next((spec for spec in FORMULAS[formula_id].params if spec.name == name), None)


def label_for(spec: ParamSpec, known: Mapping[str, object]) -> str:
    """What the student is asked, worded for the params already known."""

    return spec.label_from(known) if spec.label_from is not None else spec.label


def missing_params(formula_id: FormulaId, params: Mapping[str, object]) -> list[ParamSpec]:
    """Required params not given yet, in display order."""

    return [
        spec
        for spec in FORMULAS[formula_id].params
        if params.get(spec.name) is None and spec.required(params)
    ]


def calculate(formula_id: FormulaId, params: Mapping[str, object]) -> CalculationResult:
    """Raises `CalculationInputError` (every bad or missing field at once). The result
    carries the registry's version and source of the formula."""

    formula = FORMULAS[formula_id]
    return replace(
        formula.compute(params),
        version=formula.version,
        source=formula.source,
        source_url=formula.source_url,
    )


def output_spec(formula_id: FormulaId, key: str) -> OutputSpec | None:
    return next((spec for spec in FORMULAS[formula_id].outputs if spec.key == key), None)


# ---------------------------------------------------------------------------
# Rule-based routing to a built-in formula (runs before the LLM extractor)
# ---------------------------------------------------------------------------

BUILTIN_TRIGGERS: dict[FormulaId, tuple[re.Pattern[str], ...]] = {
    "gpa": (
        re.compile(r"\bgpa\b"),
        re.compile(r"điểm trung bình (chung |tích lũy |tích luỹ |học kỳ |học kì )"),
        re.compile(r"\bđtb(c|tl)?\b"),
        re.compile(r"trung bình (tích lũy|tích luỹ|học kỳ|học kì)"),
    ),
    "course_score": (
        re.compile(r"(điểm )?tổng kết (học phần|môn)"),
        re.compile(r"điểm học phần"),
        re.compile(r"lý thuyết.{0,40}thực hành|thực hành.{0,40}lý thuyết"),
        re.compile(r"\b(tx|tbtx|gk|ck)\b.{0,30}\b(tx|tbtx|gk|ck)\b"),
        re.compile(r"giữa kỳ.{0,40}cuối kỳ|giữa kì.{0,40}cuối kì"),
        re.compile(r"(điểm )?(cuối|giữa) k[ỳì] (cần|phải|tối thiểu|bao nhiêu)"),
    ),
    "grade_conversion": (
        re.compile(r"quy đổi|qui đổi|đổi (sang|ra) (điểm chữ|thang 4|hệ 4)"),
        re.compile(r"(là|được|ra) (điểm )?(chữ )?[abcdf]\+?(\s|$|\?|,|\.)"),
        re.compile(r"điểm chữ"),
    ),
    # Tuition triggers name only the period; route_builtin requires "học phí" too.
    "tuition_semester": (re.compile(r"học kỳ|học kì|\bkỳ\b|\bkì\b"),),
    "tuition_year": (re.compile(r"năm học|một năm|\b1 năm|mỗi năm|cả năm"),),
    "tuition_program": (
        re.compile(
            r"(cả|toàn|trọn) (khoá|khóa|ngành|chương trình)"
            r"|(đến khi|tới khi|cho tới lúc) (ra trường|tốt nghiệp)"
        ),
    ),
    "admission_score_2026": (
        re.compile(r"điểm xét tuyển|\bđxt\b|\bxt[123]\b"),
        re.compile(r"đgnl|đánh giá năng lực|điểm thủ khoa"),
        re.compile(r"(điểm|kết quả) (thi )?(tốt nghiệp|tn) .{0,40}(học bạ|ưu tiên|điểm cộng)"),
    ),
    "transcript_score_3hk": (
        re.compile(r"học bạ|kết quả học tập thpt"),
        re.compile(r"lớp 10.{0,60}lớp 11|học kỳ 1 lớp 12|hk1 lớp 12|hk 1 lớp 12"),
        re.compile(r"ngưỡng nhận hồ sơ"),
    ),
    "date_span": (
        re.compile(r"(bao nhiêu|mấy) ngày|kéo dài|cách nhau|quá hạn|còn .{0,20}ngày"),
        re.compile(r"\d{1,2}/\d{1,2}/\d{4}.{0,40}\d{1,2}/\d{1,2}/\d{4}"),
    ),
    "time_span": (
        re.compile(
            r"(bao nhiêu|mấy) (giờ|tiếng)|tổng thời gian|\d{1,2}h\d{2}\s*[-\u2013]\s*\d{1,2}h\d{2}"
        ),
    ),
    "weighted_score": (
        re.compile(r"trọng số"),
        re.compile(r"\d+\s*%.{0,80}\d+\s*%"),
        re.compile(r"đề cương|flipped|multiple.choice|mid.term|final test"),
    ),
}
_TUITION = re.compile(r"học phí")


def route_builtin(question: str) -> list[FormulaId]:
    """Built-in formulas whose trigger matches the (lower-cased, NFC) question. A
    question about tuition is only ever routed to a tuition formula: "15 tín lý thuyết
    4 tín thực hành, học phí bao nhiêu" names credits, not a course score."""

    text = unicodedata.normalize("NFC", question).lower()
    tuition = _TUITION.search(text) is not None
    return [
        formula_id
        for formula_id, patterns in BUILTIN_TRIGGERS.items()
        if (formula_id in TUITION_SCOPES) == tuition
        and any(pattern.search(text) for pattern in patterns)
    ]


def describe_builtin_formulas() -> str:
    """The formula list rendered into the extractor prompt (one source of truth): id,
    version, what it computes, its params and the outputs a target may name."""

    lines: list[str] = []
    for formula in FORMULAS.values():
        params = "; ".join(f"`{spec.name}`: {spec.label}" for spec in formula.params)
        outputs = ", ".join(f"`{spec.key}` ({spec.label})" for spec in formula.outputs)
        lines.append(
            f"- `{formula.formula_id}` (v{formula.version}) - {formula.description}."
            f" Tham số: {params}. Đầu ra: {outputs}"
        )
    return "\n".join(lines)


def describe_registry() -> str:
    """One line per formula for the classifier: what the system can compute itself."""

    return "\n".join(f"- {formula.title}: {formula.description}" for formula in FORMULAS.values())


def describe_builtin_rules() -> str:
    """The built-in formulas with every rule (weights, rounding, grade scale), for the
    LLM that answers what Python does not compute (e.g. "cần bao nhiêu để được A+").
    Rendered from the constants above - one source of truth."""

    scale = "\n".join(
        f"- từ {fixed(band.min_score, 1)}: {band.letter}, thang 4 = {fixed(band.gp4, 1)}"
        for band in GRADE_SCALE
    )
    return "\n".join(
        [
            f"### {FORMULAS['course_score'].title}",
            *(f"- {line}" for line in COURSE_SCORE_FORMULA_TEXT),
            "",
            f"### {FORMULAS['gpa'].title}",
            *(f"- {line}" for line in GPA_FORMULA_TEXT),
            "",
            "### Bảng quy đổi (điểm thang 10 đã làm tròn đến 0.1)",
            scale,
            "",
            "### Học phí (một học kỳ, một năm học hoặc cả khoá)",
            *(f"- {line}" for line in TUITION_FORMULA_TEXT),
            "- Đơn giá lấy theo đúng khoá tuyển sinh, khối/ngành, hệ và cơ sở trong biểu phí",
        ]
    )
