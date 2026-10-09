"""Every academic business rule UniSage calculates lives in this file.

The grade scale, the theory weights, the rounding places and the input limits
are declared here once; nothing else in the codebase may hard-code them. All
arithmetic is `Decimal` (inputs go through `Decimal(str(value))`), rounding is
half-up via `round_half_up` - never Python's banker's `round()`.

Spec: docs/specs/SPEC-calc-engine.md.
"""

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    )


# ---------------------------------------------------------------------------
# Course score (học phần tích hợp lý thuyết + thực hành)
# ---------------------------------------------------------------------------

COURSE_SCORE_FORMULA_TEXT = (
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
    """ĐTKHP of a theory + practice course. TBtx, ĐLT and ĐTH are each rounded to 0.1
    before the next step uses them; ĐTKHP is rounded to 0.1 at the end. `tbtx` is
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
                symbolic=COURSE_SCORE_FORMULA_TEXT[0],
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
                symbolic=COURSE_SCORE_FORMULA_TEXT[1],
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
                symbolic=COURSE_SCORE_FORMULA_TEXT[2],
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
            symbolic=COURSE_SCORE_FORMULA_TEXT[3],
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
        inputs=tuple(inputs),
        steps=tuple(steps),
        outputs=(
            ("ĐTKHP", fixed(final, 1)),
            ("Điểm chữ", band.letter),
            ("Thang 4", fixed(band.gp4, 1)),
        ),
        summary=_conversion_summary("ĐTKHP", final, band),
    )


# ---------------------------------------------------------------------------
# GPA (thang 4)
# ---------------------------------------------------------------------------

GPA_COURSES_MAX = 30
COURSE_CREDIT_MIN = 1
COURSE_CREDIT_MAX = 10
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
        inputs=tuple(
            (name, f"{credits} TC, điểm {score_text.split(' → ')[0]}")
            for name, credits, score_text, _ in rows
        ),
        steps=tuple(steps),
        outputs=(("GPA", fixed(final, 2)), ("Tổng tín chỉ", str(total_credits))),
    )


# ---------------------------------------------------------------------------
# Parameters per formula, and dispatch
# ---------------------------------------------------------------------------

ParamKind = Literal["number", "number_list", "number_or_list", "course_table"]
FormulaId = Literal["gpa", "course_score", "grade_conversion"]


def _always(_: Mapping[str, object]) -> bool:
    return True


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


@dataclass(frozen=True)
class Formula:
    formula_id: FormulaId
    title: str
    description: str
    params: tuple[ParamSpec, ...]
    compute: Callable[[Mapping[str, object]], CalculationResult]


FORMULAS: dict[FormulaId, Formula] = {
    "course_score": Formula(
        formula_id="course_score",
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
}


def param_spec(formula_id: FormulaId, name: str) -> ParamSpec | None:
    return next((spec for spec in FORMULAS[formula_id].params if spec.name == name), None)


def missing_params(formula_id: FormulaId, params: Mapping[str, object]) -> list[ParamSpec]:
    """Required params not given yet, in display order."""

    return [
        spec
        for spec in FORMULAS[formula_id].params
        if params.get(spec.name) is None and spec.required(params)
    ]


def calculate(formula_id: FormulaId, params: Mapping[str, object]) -> CalculationResult:
    """Raises `CalculationInputError` (every bad or missing field at once)."""

    return FORMULAS[formula_id].compute(params)


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
    ),
    "grade_conversion": (
        re.compile(r"quy đổi|qui đổi|đổi (sang|ra) (điểm chữ|thang 4|hệ 4)"),
        re.compile(r"(là|được|ra) (điểm )?(chữ )?[abcdf]\+?(\s|$|\?|,|\.)"),
        re.compile(r"điểm chữ"),
    ),
}


def route_builtin(question: str) -> list[FormulaId]:
    """Built-in formulas whose trigger matches the (lower-cased, NFC) question."""

    text = unicodedata.normalize("NFC", question).lower()
    return [
        formula_id
        for formula_id, patterns in BUILTIN_TRIGGERS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]


def describe_builtin_formulas() -> str:
    """The formula list rendered into the extractor prompt (one source of truth)."""

    lines: list[str] = []
    for formula in FORMULAS.values():
        params = "; ".join(f"`{spec.name}`: {spec.label}" for spec in formula.params)
        lines.append(f"- `{formula.formula_id}` - {formula.description}. Tham số: {params}")
    return "\n".join(lines)
