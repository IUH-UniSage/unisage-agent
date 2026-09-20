from dataclasses import replace

import pymupdf
import pytest

from app.rag.ingestion.canonical_table import (
    RowSignals,
    TableBlock,
    build_table_from_plain_rows,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_hierarchy import (
    HIERARCHY_SCORING,
    _parse_ordinal,
    infer_hierarchy,
)
from app.schemas.ingestion import HeaderSource, RegionType
from tests.test_table_evidence import SAMPLE


def _table(header: list[str], rows: list[list[str]]) -> TableBlock:
    return build_table_from_plain_rows(header, rows, HeaderSource.INFERRED, 0.6)


def _ancestors(table: TableBlock) -> list[list[str]]:
    return [row.ancestors for row in infer_hierarchy(table).rows]


def _with_signals(
    table: TableBlock, x0s: list[float | None], *, label_col: int = 1, **extra
) -> TableBlock:
    """Attach per-row geometry (x0 of the label cell at `label_col`)."""

    rows = []
    for row, x0 in zip(table.rows, x0s, strict=True):
        count = len(row.cells)
        rows.append(
            replace(
                row,
                signals=RowSignals(
                    cell_x0=[x0 if i == label_col else None for i in range(count)],
                    cell_bold=extra.get("bold", [False] * count),
                    cell_size=[9.0 if i == label_col else None for i in range(count)],
                    cell_merged=[False] * count,
                    height=17.0,
                ),
            )
        )
    return replace(table, rows=rows)


def test_scoring_constants_are_pinned() -> None:
    s = HIERARCHY_SCORING

    weights = (
        s.weight_numbering,
        s.weight_indent,
        s.weight_span,
        s.weight_font,
        s.weight_mask,
        s.weight_height,
    )
    assert weights == (0.35, 0.25, 0.15, 0.10, 0.10, 0.05)
    assert sum(weights) == pytest.approx(1.0)
    assert (s.min_present_weight, s.assign_threshold, s.tie_gap) == (0.25, 0.60, 0.10)


@pytest.mark.parametrize(
    ("text", "style"),
    [
        ("1", "number"),
        ("12.", "number"),
        ("1.1", "dotted2"),
        ("1.1.1", "dotted3"),
        ("A.", "upper"),
        ("a)", "lower"),
        ("II", "roman_upper"),
        ("iv.", "roman_lower"),
        ("(3)", "number"),
    ],
)
def test_ordinal_styles(text: str, style: str) -> None:
    parsed = _parse_ordinal(text)

    assert parsed is not None and parsed[0] == style


@pytest.mark.parametrize("text", ["", "1.300.000", "980.000", "Khoa hoc", "2025", "12.500"])
def test_money_and_words_are_not_ordinals(text: str) -> None:
    assert _parse_ordinal(text) is None


def test_numbered_rows_nest_by_first_appearance_of_each_style() -> None:
    table = _table(
        ["TT", "Name", "Fee"],
        [
            ["A.", "Section A", ""],
            ["1", "Group one", ""],
            ["1.1", "Item one", "10"],
            ["1.2", "Item two", "20"],
            ["2", "Group two", ""],
            ["2.1", "Item three", "30"],
            ["B.", "Section B", ""],
            ["1", "Group again", ""],
        ],
    )

    assert _ancestors(table) == [
        [],
        ["A. Section A"],
        ["A. Section A", "1 Group one"],
        ["A. Section A", "1 Group one"],
        ["A. Section A"],
        ["A. Section A", "2 Group two"],
        [],
        ["B. Section B"],
    ]


def test_roman_letter_and_multi_level_numbering() -> None:
    table = _table(
        ["No", "Title", "Amount"],
        [
            ["I", "Revenue", ""],
            ["1", "Tuition", ""],
            ["a)", "Regular", "5"],
            ["b)", "Part-time", "6"],
            ["2", "Fees", "7"],
            ["II", "Costs", ""],
            ["1", "Staff", "8"],
        ],
    )

    result = _ancestors(table)

    assert result[2] == ["I Revenue", "1 Tuition"]
    assert result[3] == ["I Revenue", "1 Tuition"]
    assert result[4] == ["I Revenue"]
    assert result[5] == []
    assert result[6] == ["II Costs"]


def test_parent_row_with_a_total_value_is_still_a_parent() -> None:
    table = _table(
        ["TT", "Label", "Unit price", "Total"],
        [
            ["1", "Section", "", ""],
            ["", "Group X", "", "900"],
            ["", "Leaf one", "10", ""],
            ["", "Leaf two", "20", ""],
            ["", "Group Y", "", "700"],
            ["", "Leaf three", "30", ""],
            ["2", "Section two", "", ""],
        ],
    )
    # numeric mask alone is only 0.10 of evidence + the 0.35 numbering prior: assigned
    result = _ancestors(table)

    assert result[1] == ["1 Section"]
    assert result[2] == ["1 Section", "Group X"]
    assert result[3] == ["1 Section", "Group X"]
    assert result[4] == ["1 Section"]  # a sibling of Group X, not its child
    assert result[5] == ["1 Section", "Group Y"]
    assert result[6] == []


def test_unnumbered_rows_with_the_same_fill_pattern_are_siblings_not_parents() -> None:
    table = _table(
        ["TT", "Label", "A", "B"],
        [
            ["1", "Section", "", ""],
            ["", "Item one", "10", "20"],
            ["", "Item two", "30", "40"],
            ["", "Note", "50", ""],
            ["2", "Section two", "", ""],
        ],
    )

    assert _ancestors(table) == [[], ["1 Section"], ["1 Section"], ["1 Section"], []]


def test_flat_table_gets_no_ancestors() -> None:
    table = _table(
        ["Name", "Qty", "Price"],
        [["Alpha", "1", "10"], ["Beta", "", "20"], ["Gamma", "3", "30"], ["Delta", "4", ""]],
    )

    assert _ancestors(table) == [[], [], [], []]


def test_numbered_flat_list_has_no_ancestors() -> None:
    table = _table(
        ["STT", "Name", "Price"],
        [["1", "Alpha", "10"], ["2", "Beta", "20"], ["3", "Gamma", "30"], ["4", "Delta", "40"]],
    )

    assert _ancestors(table) == [[], [], [], []]


def test_quantity_column_right_of_the_label_is_not_an_ordinal_column() -> None:
    table = _table(
        ["Name", "Qty", "Price"],
        [["Alpha", "1", "10"], ["Beta", "2", "20"], ["Gamma", "", "30"], ["Delta", "4", "40"]],
    )

    assert _ancestors(table) == [[], [], [], []]


def test_ordinal_glued_to_the_label_is_recognised_when_there_is_no_ordinal_column() -> None:
    table = _table(
        ["Content", "Amount"],
        [
            ["1. Revenue", ""],
            ["1.1. Tuition", "5"],
            ["1.2. Fees", "6"],
            ["2. Costs", ""],
            ["2.1. Staff", "7"],
        ],
    )

    result = _ancestors(table)

    assert result[1] == ["1. Revenue"]
    assert result[3] == []
    assert result[4] == ["2. Costs"]


def test_table_without_enough_rows_is_returned_unchanged() -> None:
    table = _table(["TT", "Name"], [["1", "Only"]])

    assert infer_hierarchy(table) is table


def test_a_skipped_numbering_level_still_nests_directly_under_its_parent() -> None:
    table = _table(
        ["TT", "Name", "Val"],
        [
            ["1", "Top", ""],
            ["1.1.1", "Deep item", "5"],
            ["1.1.2", "Sibling", "6"],
            ["2", "Next", ""],
        ],
    )

    result = infer_hierarchy(table)

    assert [row.ancestors for row in result.rows] == [[], ["1 Top"], ["1 Top"], []]
    assert all("depth_clamped" not in row.warnings for row in result.rows)


def test_a_single_ordinal_cell_is_not_enough_evidence_for_numbering() -> None:
    table = _table(
        ["TT", "Name", "Val"], [["1", "Only numbered", ""], ["", "Row", "5"], ["", "Row2", "6"]]
    )

    assert _ancestors(table) == [[], [], []]


def test_indentation_alone_gives_the_hierarchy_without_any_numbering() -> None:
    table = _table(
        ["Item", "Amount"],
        [
            ["Fruit", ""],
            ["Apple", "5"],
            ["Pear", "6"],
            ["Vegetable", ""],
            ["Carrot", "7"],
        ],
    )
    table = _with_signals(table, [50.0, 60.0, 60.0, 50.0, 60.0], label_col=0)

    result = _ancestors(table)

    assert result == [[], ["Fruit"], ["Fruit"], [], ["Vegetable"]]


def test_three_indent_levels_give_the_full_ancestor_stack() -> None:
    table = _table(
        ["Item", "Amount"],
        [
            ["Region", ""],
            ["City", ""],
            ["Shop", "1"],
            ["Kiosk", "2"],
            ["Other city", ""],
            ["Stall", "3"],
        ],
    )
    table = _with_signals(table, [50.0, 60.0, 70.0, 70.0, 60.0, 70.0], label_col=0)

    assert _ancestors(table) == [
        [],
        ["Region"],
        ["Region", "City"],
        ["Region", "City"],
        ["Region"],
        ["Region", "Other city"],
    ]


def test_missing_geometry_falls_back_to_no_assignment() -> None:
    table = _table(["Item", "Amount"], [["Fruit", ""], ["Apple", "5"], ["Pear", "6"]])

    assert _ancestors(table) == [[], [], []]  # indentation unknown: no guess


def test_conflicting_signals_leave_the_row_uncertain_and_the_rest_unharmed() -> None:
    # Numbering says the unnumbered row is a child; indentation says it is a sibling
    table = _table(
        ["TT", "Name", "Val"],
        [["1", "Top", ""], ["", "Odd row", "5"], ["", "Next", "6"], ["2", "Second", ""]],
    )
    table = _with_signals(table, [50.0, 50.0, 60.0, 50.0])

    result = infer_hierarchy(table)

    assert "hierarchy_uncertain" in result.rows[1].warnings
    assert result.rows[1].ancestors == []
    assert result.rows[1].confidence < HIERARCHY_SCORING.assign_threshold
    assert result.rows[3].ancestors == []  # later numbered rows are unaffected


def test_inference_is_deterministic() -> None:
    table = _table(
        ["TT", "Label", "V"],
        [["1", "A", ""], ["", "B", "1"], ["", "C", ""], ["", "D", "2"], ["2", "E", ""]],
    )

    first = [(r.ancestors, r.confidence, r.warnings) for r in infer_hierarchy(table).rows]
    second = [(r.ancestors, r.confidence, r.warnings) for r in infer_hierarchy(table).rows]

    assert first == second


def _pdf_from_indented_rows(rows: list[tuple[str, str, int]]) -> bytes:
    """A ruled 2-column PDF table where hierarchy is shown ONLY by indenting
    the label text (no ordinal column, no bold)."""

    doc = pymupdf.open()
    page = doc.new_page()
    top, height = 100, 20
    for index, (label, amount, level) in enumerate([("Item", "Amount", 0), *rows]):
        y = top + index * height
        for x0, x1 in ((50, 350), (350, 500)):
            page.draw_rect(pymupdf.Rect(x0, y, x1, y + height), color=(0, 0, 0), width=0.8)
        page.insert_text((53 + level * 12, y + 14), label, fontsize=9)
        if amount:
            page.insert_text((353, y + 14), amount, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def test_indent_only_pdf_table_recovers_ancestors_end_to_end() -> None:
    content = _pdf_from_indented_rows(
        [
            ("Faculty of Science", "", 0),
            ("Department of Physics", "", 1),
            ("Course PHY101", "3", 2),
            ("Course PHY102", "4", 2),
            ("Department of Chemistry", "", 1),
            ("Course CHE101", "3", 2),
            ("Faculty of Arts", "", 0),
            ("Course ART100", "2", 1),
        ]
    )

    regions = split_regions(content, "t.pdf", "pdf")
    table = next(r.table for r in regions if r.region_type == RegionType.TABLE and r.table)

    by_label = {row.cells[0]: row.ancestors for row in table.rows}
    assert by_label["Faculty of Science"] == []
    assert by_label["Department of Physics"] == ["Faculty of Science"]
    assert by_label["Course PHY102"] == ["Faculty of Science", "Department of Physics"]
    assert by_label["Course CHE101"] == ["Faculty of Science", "Department of Chemistry"]
    assert by_label["Course ART100"] == ["Faculty of Arts"]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_sample_pdf_rows_carry_the_full_ancestor_stack_across_pages() -> None:
    regions = split_regions(SAMPLE.read_bytes(), SAMPLE.name, "pdf")
    table = next(r.table for r in regions if r.region_type == RegionType.TABLE and r.table)

    page3_first = next(row for row in table.rows if row.page_start == 3)
    assert page3_first.cells[1] == "Môn lý thuyết"
    assert len(page3_first.ancestors) == 4
    assert page3_first.ancestors[0].startswith("A. Đối với Trụ sở chính")
    assert page3_first.ancestors[1].startswith("3 Đại học chính quy")
    assert page3_first.ancestors[2].startswith("3.1 Khóa tuyển sinh năm học 2025-2026")
    assert page3_first.ancestors[3].startswith("Khối kinh tế")

    congnghe = [
        row
        for row in table.rows
        if row.cells[1] == "Môn lý thuyết" and "Khối Công nghệ" in row.ancestors
    ]
    assert congnghe
    assert all("3.1 Khóa tuyển sinh năm học 2025-2026" in row.ancestors[2] for row in congnghe[:1])
    assert not any("hierarchy_uncertain" in row.warnings for row in table.rows)
    # section B restarts the stack
    b_row = next(row for row in table.rows if row.cells[0] == "B")
    assert b_row.ancestors == []
