"""CalculationResult -> markdown shown to the student. Deterministic: the same
result always renders the same text, and no LLM ever touches these numbers."""

from app.calculation.result import CalculationResult


def render_markdown(result: CalculationResult, *, notice: str | None = None) -> str:
    lines: list[str] = []
    if notice:
        lines += [f"**{notice}**", ""]
    lines += [f"**{result.title}**", "", "Công thức:"]
    lines += [f"- {line}" for line in result.formula_text]
    if result.inputs:
        lines += ["", "Số liệu:"]
        lines += [f"- {label}: {value}" for label, value in result.inputs]
    lines += ["", "Thay số:"]
    lines += [
        f"{index}. {step.label}: {step.substituted}"
        for index, step in enumerate(result.steps, start=1)
    ]
    outputs = " · ".join(f"{label} **{value}**" for label, value in result.outputs)
    lines += ["", f"Kết quả: {outputs}"]
    lines += [f"> {warning}" for warning in result.warnings]
    return "\n".join(lines)
