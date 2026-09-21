"""Regressions found by running the pipeline on real curriculum PDFs: group rows
in the first column, markup in cells, wrapped cells split over markdown rows,
tables continuing across pages with no header and no usable geometry."""

from dataclasses import replace
from pathlib import Path

import pytest

from app.rag.ingestion.canonical_table import (
    EventKind,
    RowDisposition,
    RowSignals,
    build_table_from_plain_rows,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import EvidenceCell, EvidenceRow, TableEvidence
from app.rag.ingestion.table_hierarchy import infer_hierarchy
from app.rag.ingestion.table_merger import (
    MERGE_SCORING,
    MergeCandidate,
    decide_merge,
    merge_page_tables,
)
from app.rag.ingestion.table_normalizer import build_page_table, strip_markup
from app.schemas.ingestion import HeaderSource, RegionType

SAMPLES = Path(__file__).parent.parent / "_to_delete"


def _sample(pattern: str) -> Path | None:
    found = sorted(SAMPLES.glob(pattern)) if SAMPLES.exists() else []
    return found[0] if found else None


def _tables(path: Path):
    regions = split_regions(path.read_bytes(), path.name, "pdf")
    return [r for r in regions if r.region_type == RegionType.TABLE and r.table]


# --- C: markup is formatting, not content ------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("**STT**", "STT"),
        ("_3_", "3"),
        ("**Nhóm** 2", "Nhóm 2"),
        ("~~1 130 000~~", "1 130 000"),
        ("<sup>Nguyên lý</sup><br>ứng", "Nguyên lý<br>ứng"),
        ("snake_case_name", "snake_case_name"),
        ("plain", "plain"),
    ],
)
def test_strip_markup(raw: str, clean: str) -> None:
    assert strip_markup(raw) == clean


def test_markup_is_kept_in_the_ledger_but_not_in_names_or_cells() -> None:
    table = build_page_table(
        [
            "|**STT**|**Mã môn**|**Tên**|**Tín chỉ**|",
            "|---|---|---|---|",
            "|1|2112012 <br>|<sup>Triết học</sup><br>Mác|_3_|",
            "|2|2113430|Toán|_3_|",
        ],
        1,
        None,
    )

    assert table.header_row == ["STT", "Mã môn", "Tên", "Tín chỉ"]
    assert table.data_rows[0] == ["1", "2112012 <br>", "Triết học<br>Mác", "3"]
    assert "**STT**" in table.source_rows[0].raw_text
    assert check_table_invariants(table) == []


# --- B: group rows in the first column ---------------------------------------


def _curriculum():
    header = ["STT", "Mã", "Tên môn học", "Tín chỉ"]
    rows = [
        ["", "", "", "40"],
        ["HỌC KỲ 1", "", "", "20"],
        ["Bắt buộc", "", "", "20"],
        ["1", "211", "Triết học", "3"],
        ["2", "212", "Toán cao cấp", "3"],
        ["HỌC KỲ 2", "", "", "20"],
        ["Bắt buộc", "", "", "17"],
        ["1", "213", "Kinh tế vi mô", "3"],
        ["Tự chọn", "", "", "3"],
        ["Nhóm 1", "", "", "3"],
        ["1", "214", "Logic học", "3"],
        ["2", "215", "Toán ứng dụng", "3"],
        ["HỌC KỲ 3", "", "", "20"],
        ["Bắt buộc", "", "", "20"],
        ["1", "216", "Kinh tế lượng", "3"],
        ["Tự chọn", "", "", "3"],
        ["Nhóm", "2", "", "3"],
        ["1", "217", "Xã hội học", "3"],
    ]
    return build_table_from_plain_rows(header, rows, HeaderSource.INFERRED, 0.75)


def test_group_rows_in_the_first_column_head_the_hierarchy() -> None:
    result = infer_hierarchy(_curriculum())
    by_name = {row.cells[2]: row.ancestors for row in result.rows if row.cells[2]}

    assert by_name["Triết học"] == ["HỌC KỲ 1", "Bắt buộc"]
    assert by_name["Kinh tế vi mô"] == ["HỌC KỲ 2", "Bắt buộc"]
    assert by_name["Logic học"] == ["HỌC KỲ 2", "Tự chọn", "Nhóm 1"]
    assert by_name["Kinh tế lượng"] == ["HỌC KỲ 3", "Bắt buộc"]
    assert by_name["Xã hội học"] == ["HỌC KỲ 3", "Tự chọn", "Nhóm 2"]  # label spread over two cells
    groups = {row.cells[0]: row.ancestors for row in result.rows if not row.cells[2]}
    assert groups["HỌC KỲ 2"] == []
    assert groups["Bắt buộc"] in (["HỌC KỲ 1"], ["HỌC KỲ 2"], ["HỌC KỲ 3"])
    assert groups["Tự chọn"] == ["HỌC KỲ 2"] or groups["Tự chọn"] == ["HỌC KỲ 3"]


def test_an_item_with_an_empty_cell_is_never_taken_for_a_parent_in_group_mode() -> None:
    table = _curriculum()
    rows = list(table.rows)
    rows[3] = replace(rows[3], cells=["1", "", "Triết học", ""])  # code and credits missing
    rows[4] = replace(rows[4], cells=["2", "212", "Toán cao cấp", "3"])

    result = infer_hierarchy(replace(table, rows=rows))

    assert result.rows[3].ancestors == ["HỌC KỲ 1", "Bắt buộc"]
    assert result.rows[4].ancestors == ["HỌC KỲ 1", "Bắt buộc"]  # not under "Triết học"
    assert not any("hierarchy_uncertain" in row.warnings for row in result.rows)


def test_sparse_rows_do_not_make_a_group_layout() -> None:
    table = build_table_from_plain_rows(
        ["A", "B", "Name"],
        [["x", "", ""], ["", "", "one"], ["", "", "two"], ["", "", "three"], ["", "", "four"]],
        HeaderSource.INFERRED,
        0.6,
    )

    assert [row.ancestors for row in infer_hierarchy(table).rows] == [[]] * 5


# --- A: indentation is the offset from the cell edge, not an absolute x ------


def test_a_column_shifted_right_on_part_of_the_page_is_not_a_deeper_level() -> None:
    table = build_table_from_plain_rows(
        ["Item", "Amount"],
        [["Fruit", ""], ["Apple", "5"], ["Pear", "6"], ["Nut", "7"], ["Fig", "8"], ["Yam", "9"]],
        HeaderSource.INFERRED,
        0.6,
    )
    # the second half of the table is drawn 55pt further right; the text sits at
    # the same offset (3pt) from its own cell's left edge everywhere
    lefts = [50.0, 50.0, 50.0, 105.0, 105.0, 105.0]
    rows = [
        replace(
            row,
            signals=RowSignals(
                cell_x0=[left + 3.0, None],
                cell_bold=[False, False],
                cell_size=[9.0, None],
                cell_merged=[False, False],
                height=17.0,
                cell_left=[left, None],
            ),
        )
        for row, left in zip(table.rows, lefts, strict=True)
    ]

    result = infer_hierarchy(replace(table, rows=rows))

    assert all(row.ancestors == [] for row in result.rows)
    assert not any("hierarchy_uncertain" in row.warnings for row in result.rows)


def test_a_real_indent_is_still_read_from_the_offset() -> None:
    table = build_table_from_plain_rows(
        ["Item", "Amount"],
        [["Fruit", ""], ["Apple", "5"], ["Pear", "6"], ["Veg", ""], ["Kale", "7"]],
        HeaderSource.INFERRED,
        0.6,
    )
    offsets = [3.0, 12.0, 12.0, 3.0, 12.0]
    rows = [
        replace(
            row,
            signals=RowSignals(
                cell_x0=[200.0 + offset, None],  # a column far to the right, same for all
                cell_bold=[False, False],
                cell_size=[9.0, None],
                cell_merged=[False, False],
                height=17.0,
                cell_left=[200.0, None],
            ),
        )
        for row, offset in zip(table.rows, offsets, strict=True)
    ]

    result = infer_hierarchy(replace(table, rows=rows))

    assert [row.ancestors for row in result.rows] == [[], ["Fruit"], ["Fruit"], [], ["Veg"]]


# --- wrapped cells split over markdown rows ---------------------------------


def _evidence(rows: list[list[str]]) -> TableEvidence:
    built = []
    for index, texts in enumerate(rows):
        y0 = 100.0 + index * 20
        cells = [
            EvidenceCell((50.0 + i * 100, y0, 150.0 + i * 100, y0 + 20), text)
            for i, text in enumerate(texts)
        ]
        built.append(EvidenceRow(cells=cells, y0=y0, y1=y0 + 20))
    return TableEvidence(
        page_number=1,
        bbox=(50.0, 100.0, 50.0 + 100.0 * len(rows[0]), 100.0 + 20.0 * len(rows)),
        page_height=842.0,
        rows=built,
        column_bounds=[50.0 + 100.0 * i for i in range(len(rows[0]))],
        header_row_count=1,
    )


def test_markdown_rows_that_are_one_wrapped_geometry_row_are_merged() -> None:
    evidence = _evidence(
        [
            ["TT", "Name", "Description"],
            ["1", "Alpha", "first line second line third line"],
            ["2", "Beta", "short"],
        ]
    )
    lines = [
        "|TT|Name|Description|",
        "|---|---|---|",
        "|1|Alpha|first line|",
        "|||second line|",
        "|||third line|",
        "|2|Beta|short|",
    ]

    table = build_page_table(lines, 1, evidence)

    assert "evidence_unaligned" not in table.warnings
    assert table.data_rows == [
        ["1", "Alpha", "first line second line third line"],
        ["2", "Beta", "short"],
    ]
    merged = [s for s in table.source_rows if s.disposition == RowDisposition.MERGED]
    assert len(merged) == 3
    assert table.rows[0].source_row_ids == ["p1-r2", "p1-r3", "p1-r4"]
    assert any(e.kind == EventKind.MERGE_CELLS for e in table.normalization_events)
    assert check_table_invariants(table) == []


def test_text_the_geometry_row_does_not_have_stops_the_alignment() -> None:
    evidence = _evidence([["TT", "Name"], ["1", "Alpha"]])

    table = build_page_table(["|TT|Name|", "|---|---|", "|1|Alpha|", "|2|Extra|"], 1, evidence)

    assert "evidence_unaligned" in table.warnings
    assert table.data_rows == [["1", "Alpha"], ["2", "Extra"]]
    assert check_table_invariants(table) == []


# --- pages: cut text cell, and continuation without header or geometry -------


def _page(lines: list[str], page: int):
    return build_page_table(lines, page, None)


def _first_page():
    return _page(
        [
            "|STT|Code|Name|Description|",
            "|---|---|---|---|",
            "|1|211|Philosophy|A long description of the first course that goes on and on|",
            "|2|212|Calculus|Another long description of the second course, cut by the break|",
        ],
        1,
    )


def test_the_tail_of_a_cell_cut_by_the_page_break_rejoins_the_previous_row() -> None:
    second = _page(
        [
            "||||and this is the rest of that description|",
            "|---|---|---|---|",
            "|3|213|Physics|A long description of the third course that fills the cell|",
        ],
        2,
    )

    merged = merge_page_tables(_first_page(), second, header_present=False)

    assert [row.cells[1] for row in merged.rows] == ["211", "212", "213"]
    assert (
        merged.rows[1]
        .cells[3]
        .endswith("cut by the break and this is the rest of that description")
    )
    assert "continued_across_pages" in merged.rows[1].warnings
    assert merged.rows[1].page_end == 2
    assert [row.row_index for row in merged.rows] == [1, 2, 3]
    assert check_table_invariants(merged) == []


def test_a_table_continuing_without_header_or_geometry_is_merged_by_row_shape() -> None:
    first = _first_page()
    second = _page(
        [
            "|3|213|Physics|A long description of the third course that fills the cell|",
            "|4|214|Chemistry|A long description of the fourth course that fills the cell|",
            "|5|215|Biology|A long description of the fifth course that fills the cell|",
        ],
        2,
    )

    decision = decide_merge(
        MergeCandidate(first, 1, 1, [], None),
        MergeCandidate(second, 2, 2, [], None),
        only_furniture_between=True,
    )

    assert decision.merge and not decision.header_present
    assert decision.score is not None and decision.score >= MERGE_SCORING.threshold_without_header


def test_a_first_row_of_another_kind_is_not_taken_for_a_continuation() -> None:
    first = _first_page()
    second = _page(
        [
            "|Summary of the credits for the whole programme, split by faculty|a|b|c|",
            "|---|---|---|---|",
            "|Total|1|2|3|",
        ],
        2,
    )

    decision = decide_merge(
        MergeCandidate(first, 1, 1, [], None),
        MergeCandidate(second, 2, 2, [], None),
        only_furniture_between=True,
    )

    assert not decision.merge


def test_shape_weight_is_pinned_and_outside_the_four_weights() -> None:
    assert MERGE_SCORING.weight_shape == 0.45
    assert (
        MERGE_SCORING.weight_header
        + MERGE_SCORING.weight_columns
        + MERGE_SCORING.weight_bounds
        + MERGE_SCORING.weight_position
        == pytest.approx(1.0)
    )


# --- the real files ---------------------------------------------------------


LOGISTICS = _sample("Khung-chuong-trinh-log.pdf")
K20 = _sample("*K20).pdf")
AUTOMATION = _sample("*2022.pdf")


def _leaf_names(table) -> set[str]:
    """Names of the rows that carry a course code: real items, never parents."""

    return {row.cells[2] for row in table.rows if row.cells[1].strip() and row.cells[2].strip()}


@pytest.mark.skipif(LOGISTICS is None, reason="sample PDF not available")
def test_logistics_curriculum_gets_semester_and_group_context() -> None:
    tables = _tables(LOGISTICS)

    assert len(tables) == 1
    table = tables[0].table
    assert table.header_row == ["STT", "Mã môn", "Tên môn học", "Số tín chỉ", "Ghi chú"]
    assert check_table_invariants(table) == []
    by_name = {row.cells[2]: row.ancestors for row in table.rows}
    assert by_name["Triết học Mác – Lênin"] == ["HỌC KỲ 1", "Bắt buộc"]  # noqa: RUF001 - the source PDF uses an en dash
    assert by_name["Toán Ứng dụng"] == ["HỌC KỲ 2", "Tự chọn", "Nhóm 1"]
    leaves = _leaf_names(table)
    assert not any(ancestor in leaves for row in table.rows for ancestor in row.ancestors)
    assert not any("**" in cell or "_3_" in cell for row in table.rows for cell in row.cells)


@pytest.mark.skipif(K20 is None, reason="sample PDF not available")
def test_transport_engineering_curriculum_is_one_table_with_clean_context() -> None:
    tables = _tables(K20)

    assert len(tables) == 1
    region = tables[0]
    table = region.table
    assert (region.page_start, region.page_end) == (1, 8)
    assert "evidence_unaligned" not in table.warnings
    assert check_table_invariants(table) == []
    leaves = _leaf_names(table)
    assert not any(ancestor in leaves for row in table.rows for ancestor in row.ancestors)
    first_course = next(row for row in table.rows if row.cells[1].strip())
    assert first_course.ancestors[0] == "Học kỳ 1"
    assert len(first_course.ancestors) == 2


@pytest.mark.skipif(AUTOMATION is None, reason="sample PDF not available")
def test_automation_curriculum_merges_wrapped_rows_and_all_pages() -> None:
    tables = _tables(AUTOMATION)

    assert len(tables) == 1
    region = tables[0]
    table = region.table
    assert (region.page_start, region.page_end) == (1, 24)
    assert check_table_invariants(table) == []
    assert table.header_row is not None and table.header_row[0] == "STT"
    by_name = {row.cells[2].replace("<br>", " "): row.ancestors for row in table.rows}
    assert by_name["Triết học Mác - Lênin"] == [
        "1. Kiến thức giáo dục đại cương",
        "Bắt buộc",
    ]
    # a description wrapped over several markdown rows is ONE row, not stray rows
    assert not any(
        not row.cells[0].strip() and not row.cells[1].strip() and not row.cells[2].strip()
        for row in table.rows
    )
    leaves = _leaf_names(table)
    assert not any(ancestor in leaves for row in table.rows for ancestor in row.ancestors)
