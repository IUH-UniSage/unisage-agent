"""CalculationResult -> markdown shown to the student. Deterministic: the same
result always renders the same text, and no LLM ever touches these numbers.

Each value appears once: the inputs are already on the answered-panel card and
in the substituted steps (no separate "Số liệu" list), and the result is one
line (a grade conversion is folded into it, not repeated as a step).
"""

from collections.abc import Sequence

from app.calculation.result import CalculationResult


def formula_lines(
    title: str,
    latex: Sequence[str],
    text: Sequence[str],
    *,
    version: str = "",
    source: str = "",
    source_url: str = "",
) -> list[str]:
    """The title and the formula block, the same before the inputs are asked (the
    preview above the panel) and in the result."""

    lines = [f"**{title}**", "", "Công thức:"]
    if latex:
        # Each expression is its own display-math block; one blank line before the first
        # and after the last keeps them out of the surrounding list, none between
        # consecutive blocks (the client spaces them itself).
        lines.append("")
        for tex in latex:
            lines += ["$$", tex, "$$"]
        lines.append("")
    lines += [f"- {line}" for line in text]
    if version:
        shown = f"[{source}]({source_url})" if source_url else source
        lines.append(f"- _Phiên bản công thức {version} · nguồn: {shown}_")
    return lines


def render_markdown(result: CalculationResult, *, notice: str | None = None) -> str:
    lines: list[str] = []
    if notice:
        lines += [f"**{notice}**", ""]
    lines += formula_lines(
        result.title,
        result.formula_latex,
        result.formula_text,
        version=result.version,
        source=result.source,
        source_url=result.source_url,
    )
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
