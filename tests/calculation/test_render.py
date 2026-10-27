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
    "tuition_program": {"tc_lt": 100, "tc_th": 42, "don_gia_lt": 980000, "don_gia_th": 1600000},
    "admission_score_2026": {"dtn": 22, "dgnl": 900, "dtk": 1100, "dhb": 25, "uu_tien": 0.25},
    "transcript_score_3hk": {
        "mon1": "Toán",
        "m1_10": 7.5,
        "m1_11": 8,
        "m1_12": 8.5,
        "m2_10": 6,
        "m2_11": 6.5,
        "m2_12": 7,
        "mon3": "Lý",
        "m3_10": 4.5,
        "m3_11": 5,
        "m3_12": 5.5,
    },
    "weighted_score": {
        "trong_so": [20, 30, 50],
        "diem": [9, 7, 9.2],
        "lam_tron_thanh_phan": 0.5,
    },
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


def test_formula_latex_is_rendered_as_display_math_blocks() -> None:
    rendered = render_markdown(calculate("admission_score_2026", CASES["admission_score_2026"]))
    assert "\n$$\n\\text{ĐXT} = \\max(\\text{XT1},\\ \\text{XT2},\\ \\text{XT3})\n$$\n" in rendered
    assert rendered.index("$$") < rendered.index("- ĐTN: tổng điểm 3 môn")
    # A formula without LaTeX keeps its text lines only.
    assert "$$" not in render_markdown(calculate("grade_conversion", {"score10": 8}))
