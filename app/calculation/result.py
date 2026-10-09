"""What a calculation returns: the result plus every step, already substituted.

Plain data, no logic - `formulas.py` builds these and `render.py` turns them
into markdown. Every number is a `Decimal`; `display` is the string shown to the
student (up to 2 decimals, `≈` when cut).
"""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class Step:
    label: str
    symbolic: str
    substituted: str
    value: Decimal
    display: str
    note: str | None = None


@dataclass(frozen=True)
class CalculationResult:
    formula_id: str
    title: str
    formula_text: tuple[str, ...]
    inputs: tuple[tuple[str, str], ...]
    steps: tuple[Step, ...]
    outputs: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # The one result line shown to the student (markdown); None = built from `outputs`.
    # Grade conversions fold the band lookup in here instead of repeating it as a step.
    summary: str | None = None
    # The headline number (ĐTKHP, GPA, the regulation formula's result) at full
    # precision - what the target solver compares against a goal.
    primary_value: Decimal | None = None
    # Optional extra table (header row first), e.g. the solver's trade-off table.
    table: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FieldError:
    field: str
    reason: str


class CalculationInputError(Exception):
    """One or more inputs are invalid - several at once, so the panel can ask
    for all of them in one round instead of one per turn."""

    def __init__(self, errors: list[FieldError]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{error.field}: {error.reason}" for error in errors))
