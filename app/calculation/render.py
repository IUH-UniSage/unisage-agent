"""CalculationResult -> markdown shown to the student. Deterministic: the same
result always renders the same text, and no LLM ever touches these numbers.

Each value appears once: the inputs are already on the answered-panel card and
in the substituted steps (no separate "Số liệu" list), and the result is one
line (a grade conversion is folded into it, not repeated as a step).
"""

from app.calculation.result import CalculationResult


def render_markdown(result: CalculationResult, *, notice: str | None = None) -> str:
    lines: list[str] = []
    if notice:
        lines += [f"**{notice}**", ""]
    lines += [f"**{result.title}**", "", "Công thức:"]
    lines += [f"- {line}" for line in result.formula_text]
    if result.steps:
        lines += ["", "Thay số:"]
        lines += [
            f"{index}. {step.label}: {step.substituted}"
            for index, step in enumerate(result.steps, start=1)
        ]
    summary = result.summary or " · ".join(
        f"{label} **{value}**" for label, value in result.outputs
    )
    lines += ["", f"Kết quả: {summary}"]
    lines += [f"> {warning}" for warning in result.warnings]
    return "\n".join(lines)
