from collections.abc import Mapping
from pathlib import Path

import pytest

from app.calculation.formulas import FormulaId, calculate
from app.calculation.render import render_markdown

SNAPSHOTS = Path(__file__).parent / "snapshots"

CASES: dict[FormulaId, Mapping[str, object]] = {
    "course_score": {"tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8], "tclt": 2, "tcth": 1},
    "gpa": {
        "courses": [
            {"name": "Toán", "credits": 3, "score": 8.5},
            {"name": "Lý", "credits": 2, "score": "B+"},
            {"credits": 4, "score": 3.9},
        ]
    },
    "grade_conversion": {"score10": "8.45"},
}


@pytest.mark.parametrize("formula_id", sorted(CASES))
def test_render_matches_snapshot(formula_id: FormulaId) -> None:
    rendered = render_markdown(calculate(formula_id, CASES[formula_id]))
    expected = (SNAPSHOTS / f"{formula_id}.md").read_text(encoding="utf-8")
    assert rendered == expected.rstrip("\n")


def test_render_is_deterministic() -> None:
    first = render_markdown(calculate("gpa", CASES["gpa"]))
    assert render_markdown(calculate("gpa", CASES["gpa"])) == first


def test_notice_comes_first() -> None:
    rendered = render_markdown(
        calculate("grade_conversion", {"score10": 7}), notice="Kết quả tham khảo theo quy chế"
    )
    assert rendered.startswith("**Kết quả tham khảo theo quy chế**\n\n**Quy đổi")
