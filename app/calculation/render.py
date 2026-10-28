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
    if result.formula_latex:
        # Each expression is its own display-math block; one blank line before the first
        # and after the last keeps them out of the surrounding list, none between
        # consecutive blocks (the client spaces them itself).
        lines.append("")
        for tex in result.formula_latex:
            lines += ["$$", tex, "$$"]
        lines.append("")
    lines += [f"- {line}" for line in result.formula_text]
    if result.version:
        source = (
            f"[{result.source}]({result.source_url})" if result.source_url else result.source
        )
        lines.append(f"- _Phiên bản công thức {result.version} · nguồn: {source}_")
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
