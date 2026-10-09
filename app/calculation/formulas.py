"""Every academic business rule UniSage calculates lives in this file.

The grade scale, the theory weights, the rounding places and the input limits
are declared here once; nothing else in the codebase may hard-code them. All
arithmetic is `Decimal` (inputs go through `Decimal(str(value))`), rounding is
half-up via `round_half_up` - never Python's banker's `round()`.

Spec: docs/specs/SPEC-calc-engine.md.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from app.calculation.result import CalculationInputError, CalculationResult, FieldError, Step

SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("10")
SCORE_MAX_PLACES = 2
COMPONENT_CREDIT_MAX = 10
PRACTICE_SCORES_MAX = 20

COURSE_SCORE_PLACES = 1
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

    for digits in (DISPLAY_PLACES, 4):
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


def _conversion_step(rounded: Decimal, band: GradeBand) -> Step:
    return Step(
        label="Quy đổi",
        symbolic="Điểm thang 10 → điểm chữ → thang 4",
        substituted=(
            f"{fixed(rounded, 1)} thuộc khoảng {_band_range(band)} → {band.letter} → "
            f"{fixed(band.gp4, 1)}"
        ),
        value=band.gp4,
        display=fixed(band.gp4, 1),
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
    steps = (
        Step(
            label="Làm tròn",
            symbolic="Làm tròn điểm thang 10 đến 0.1",
            substituted=f"{plain(score)} → {fixed(rounded, 1)}",
            value=rounded,
            display=fixed(rounded, 1),
            note="làm tròn đến 0.1",
        ),
        _conversion_step(rounded, band),
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
    )


# ---------------------------------------------------------------------------
# Course score (học phần tích hợp lý thuyết + thực hành)
# ---------------------------------------------------------------------------

COURSE_SCORE_FORMULA_TEXT = (
    f"ĐLT = 20% {TIMES} TBtx + 30% {TIMES} GK + 50% {TIMES} CK",
    "ĐTH = (TH1 + TH2 + … + THn) / n",
    f"ĐTKHP = (ĐLT {TIMES} TCLT + ĐTH {TIMES} TCTH) / (TCLT + TCTH), làm tròn đến 0.1",
)


def course_score(params: Mapping[str, object]) -> CalculationResult:
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
    theory: dict[str, Decimal] = {}
    if tclt is None or tclt > 0:
        for field in ("tbtx", "gk", "ck"):
            if params.get(field) is None:
                if tclt:
                    _missing(field, errors)
                continue
            score = parse_score(params[field], field, errors)
            if score is not None:
                theory[field] = score

    practice: list[Decimal] = []
    if tcth is None or tcth > 0:
        raw_practice = params.get("th")
        if raw_practice is None:
            if tcth:
                _missing("th", errors)
        elif not isinstance(raw_practice, list | tuple) or not raw_practice:
            errors.append(FieldError("th", "cần ít nhất một cột điểm thực hành"))
        elif len(raw_practice) > PRACTICE_SCORES_MAX:
            errors.append(FieldError("th", f"tối đa {PRACTICE_SCORES_MAX} cột điểm thực hành"))
        else:
            for item in raw_practice:
                score = parse_score(item, "th", errors)
                if score is not None:
                    practice.append(score)

    if errors or tclt is None or tcth is None:
        raise CalculationInputError(errors)

    steps: list[Step] = []
    inputs: list[tuple[str, str]] = [
        ("Tín chỉ lý thuyết", str(tclt)),
        ("Tín chỉ thực hành", str(tcth)),
    ]

    dlt = Decimal(0)
    if tclt:
        weights = THEORY_WEIGHTS
        tbtx, gk, ck = theory["tbtx"], theory["gk"], theory["ck"]
        parts = (weights.tx * tbtx, weights.gk * gk, weights.ck * ck)
        dlt = sum(parts, Decimal(0))
        inputs += [
            ("Điểm thường xuyên", plain(tbtx)),
            ("Điểm giữa kỳ", plain(gk)),
            ("Điểm cuối kỳ", plain(ck)),
        ]
        steps.append(
            Step(
                label="Điểm lý thuyết",
                symbolic=COURSE_SCORE_FORMULA_TEXT[0],
                substituted=(
                    f"ĐLT = {plain(weights.tx)} {TIMES} {plain(tbtx)}"
                    f" + {plain(weights.gk)} {TIMES} {plain(gk)}"
                    f" + {plain(weights.ck)} {TIMES} {plain(ck)} = "
                    + " + ".join(plain(part) for part in parts)
                    + f" {equals(dlt)}"
                ),
                value=dlt,
                display=fmt(dlt),
            )
        )

    dth = Decimal(0)
    if tcth:
        total = sum(practice, Decimal(0))
        dth = total / len(practice)
        inputs.append(("Điểm thực hành", ", ".join(plain(score) for score in practice)))
        steps.append(
            Step(
                label="Điểm thực hành",
                symbolic=COURSE_SCORE_FORMULA_TEXT[1],
                substituted=(
                    f"ĐTH = ({' + '.join(plain(score) for score in practice)}) / {len(practice)}"
                    f" = {plain(total)} / {len(practice)} {equals(dth)}"
                ),
                value=dth,
                display=fmt(dth),
            )
        )

    if tclt and tcth:
        numerator = dlt * tclt + dth * tcth
        raw = numerator / (tclt + tcth)
        substituted = (
            f"ĐTKHP = ({short(dlt)} {TIMES} {tclt} + {short(dth)} {TIMES} {tcth})"
            f" / ({tclt} + {tcth})"
            f" = {short(numerator)} / {tclt + tcth} {equals(raw)}"
        )
    elif tclt:
        raw = dlt
        substituted = f"ĐTKHP = ĐLT = {short(dlt)} (học phần chỉ có lý thuyết)"
    else:
        raw = dth
        substituted = f"ĐTKHP = ĐTH = {short(dth)} (học phần chỉ có thực hành)"
    steps.append(
        Step(
            label="Điểm tổng kết học phần",
            symbolic=COURSE_SCORE_FORMULA_TEXT[2].removesuffix(", làm tròn đến 0.1"),
            substituted=substituted,
            value=raw,
            display=fmt(raw),
        )
    )

    final = round_half_up(raw, COURSE_SCORE_PLACES)
    steps.append(
        Step(
            label="Làm tròn",
            symbolic="Làm tròn ĐTKHP đến 0.1",
            substituted=f"{before_rounding(raw, final, COURSE_SCORE_PLACES)} → {fixed(final, 1)}",
            value=final,
            display=fixed(final, 1),
            note="làm tròn đến 0.1",
        )
    )
    band = grade_band(final)
    steps.append(_conversion_step(final, band))

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
    )
