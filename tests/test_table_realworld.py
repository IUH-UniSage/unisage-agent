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
    TableRow,
    build_table_from_plain_rows,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import (
    EvidenceCell,
    EvidenceRow,
    TableEvidence,
    _resolve_spans,
)
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


# --- merged cells resolved from geometry --------------------------------------


def _cell(x0: float, y0: float, x1: float, y1: float, text: str) -> EvidenceCell:
    return EvidenceCell((x0, y0, x1, y1), text)


def _spanned_evidence() -> TableEvidence:
    """TT | Label | Group header over (Per credit | Per year); `2.1` merged down
    over two data rows. Columns start at x = 0, 50, 250, 350; table ends at 450."""

    rows = [
        EvidenceRow(
            [
                _cell(0, 0, 50, 40, "TT"),
                _cell(50, 0, 250, 40, "Programme"),
                _cell(250, 0, 450, 20, "Fees 2025 - 2026"),
                None,
            ],
            0,
            20,
        ),
        EvidenceRow(
            [
                None,
                None,
                _cell(250, 20, 350, 40, "Per credit"),
                _cell(350, 20, 450, 40, "Per year"),
            ],
            20,
            40,
        ),
        EvidenceRow(
            [
                _cell(0, 40, 50, 80, "2.1"),
                _cell(50, 40, 250, 60, "Intake 2025 - Economics"),
                _cell(250, 40, 350, 60, "1.430.000"),
                _cell(350, 40, 450, 60, "43.000.000"),
            ],
            40,
            60,
        ),
        EvidenceRow(
            [
                None,
                _cell(50, 60, 250, 80, "Intake 2025 - Engineering | evening"),
                _cell(250, 60, 350, 80, "1.600.000"),
                _cell(350, 60, 450, 80, "48.000.000"),
            ],
            60,
            80,
        ),
    ]
    return TableEvidence(
        page_number=2,
        bbox=(0.0, 0.0, 450.0, 80.0),
        page_height=842.0,
        rows=_resolve_spans(rows, [0.0, 50.0, 250.0, 350.0], 450.0),
        column_bounds=[0.0, 50.0, 250.0, 350.0],
        header_row_count=2,
    )


def test_spans_are_resolved_to_the_cell_that_covers_them() -> None:
    evidence = _spanned_evidence()

    assert evidence.rows[0].covered_by[3] == (0, 2)  # header merged across columns
    assert evidence.rows[1].covered_by[:2] == [(0, 0), (0, 1)]  # header merged down
    assert evidence.rows[3].covered_by[0] == (2, 0)  # row label merged down


def test_a_garbled_markdown_header_and_a_misplaced_row_label_are_read_from_geometry() -> None:
    # The markdown extractor: loses the accents of a merged header cell into a
    # row of their own, pulls text from above the table into the header, cuts
    # the merged group header in two, and puts `2.1` on the row it is centred on.
    lines = [
        "||ÀÓỂ|Fees|Unit: VND<br>2025 - 2026|",
        "|---|---|---|---|",
        "|TT|Prgramme|Per credit|Per year|",
        "||Intake 2025 - Economics|1.430.000|43.000.000|",
        "|2.1|Intake 2025 - Engineering | evening|1.600.000|48.000.000|",
    ]

    table = build_page_table(lines, 2, _spanned_evidence())

    assert table.header_row == [
        "TT",
        "Programme",
        "Fees 2025 - 2026 > Per credit",
        "Fees 2025 - 2026 > Per year",
    ]
    assert [row.cells for row in table.rows] == [
        ["2.1", "Intake 2025 - Economics", "1.430.000", "43.000.000"],
        ["2.1", "Intake 2025 - Engineering | evening", "1.600.000", "48.000.000"],
    ]
    assert table.rows[1].inherited_cells == [0]
    assert not any("geometry_text_mismatch" in row.warnings for row in table.rows)
    assert any(e.kind == EventKind.FILL_MERGED_CELL for e in table.normalization_events)
    assert check_table_invariants(table) == []


def _row(cells: list[str], inherited: list[int] | None = None) -> TableRow:
    return TableRow(cells=cells, inherited_cells=inherited or [])


def _hierarchy(rows: list[TableRow]) -> list[list[str]]:
    table = build_table_from_plain_rows(
        ["TT", "Label", "Per credit", "Per year"],
        [row.cells for row in rows],
        HeaderSource.INFERRED,
        1.0,
    )
    numbered = [replace(row, row_index=index + 1) for index, row in enumerate(rows)]
    return [row.ancestors for row in infer_hierarchy(replace(table, rows=numbered)).rows]


def test_a_number_merged_down_from_a_row_with_values_makes_siblings() -> None:
    ancestors = _hierarchy(
        [
            _row(["2", "Postgraduate", "", ""]),
            _row(["2.1", "Economics", "1.430.000", "43.000.000"]),
            _row(["2.1", "Engineering", "1.600.000", "48.000.000"], [0]),
            _row(["2.2", "Economics", "1.400.000", "42.050.000"]),
        ]
    )

    assert ancestors[2] == ["2 Postgraduate"]
    assert ancestors[3] == ["2 Postgraduate"]


def test_a_number_merged_down_from_a_group_row_makes_children() -> None:
    ancestors = _hierarchy(
        [
            _row(["3", "Undergraduate", "", ""]),
            _row(["3.1", "Intake 2025", "", ""]),
            _row(["3.1", "Economics", "980.000", "36.260.000"], [0]),
            _row(["3.1", "Engineering", "980.000", "38.350.000"], [0]),
            _row(["3.2", "Intake 2024", "", ""]),
        ]
    )

    assert ancestors[2] == ["3 Undergraduate", "3.1 Intake 2025"]
    assert ancestors[3] == ["3 Undergraduate", "3.1 Intake 2025"]
    assert ancestors[4] == ["3 Undergraduate"]


TUITION = _sample("Quyet Dinh 1035*.pdf")


@pytest.mark.skipif(TUITION is None, reason="sample PDF not available")
def test_tuition_decision_header_spans_and_row_labels() -> None:
    assert TUITION is not None
    tables = _tables(TUITION)

    table = tables[0].table
    # page 3 ends mid-page; page 4 repeats the header and continues the table
    assert (tables[0].page_start, tables[0].page_end) == (2, 4)
    assert table.header_row == [
        "TT",
        "HỆ ĐÀO TẠO / KHÓA TUYỂN SINH",
        "Mức thu trong năm học 2025 - 2026 > Mức thu 01 tín chỉ",
        "Mức thu trong năm học 2025 - 2026 > Mức thu theo năm học",
    ]
    assert check_table_invariants(table) == []
    by_label: dict[str, TableRow] = {}
    for row in table.rows:  # first occurrence: section B repeats some labels
        by_label.setdefault(row.cells[1], row)
    # each row of a merged `2.x` cell carries its own number, never the next one
    assert by_label["Khóa tuyển sinh năm học 2024-2025 - Khối Công nghệ"].cells[0] == "2.2"
    economics = by_label["Khóa tuyển sinh năm học 2023-2024 trở về trước - Khối Kinh tế"]
    assert economics.cells[0] == "2.3"
    assert economics.ancestors == ["A ĐỐI VỚI TRỤ SỞ CHÍNH", "2 Cao học"]
    # a `|` inside a cell stays in that cell
    pharmacy = next(row for row in table.rows if "5.000.000 | Môn QPAN" in row.cells[1])
    assert pharmacy.cells[2:] == ["980k / 5.000k", ""]
    assert pharmacy.ancestors[-2:] == ["3.1 Khóa tuyển sinh năm học 2025-2026", "- Ngành Dược:"]
    # the first row of page 3 is a row of its own, not the tail of the last row
    assert by_label["Khóa tuyển sinh năm học 2024-2025"].cells[0] == "3.2"
    assert not any("geometry_text_mismatch" in row.warnings for row in table.rows)
    # the numbering carries on across the early page break
    assert by_label["Đại học liên thông, văn bằng 2 (Trụ sở chính)"].ancestors == [
        "A ĐỐI VỚI TRỤ SỞ CHÍNH"
    ]
    assert by_label["PHÂN HIỆU QUẢNG NGÃI, CƠ SỞ THANH HÓA"].ancestors == []


ADMISSION = _sample("Thong Bao 867*.pdf")


@pytest.mark.skipif(ADMISSION is None, reason="sample PDF not available")
def test_admission_notice_fills_merged_combination_cells() -> None:
    assert ADMISSION is not None
    tables = [
        region
        for region in _tables(ADMISSION)
        if region.heading_path[-1].startswith("a) Ngành/nhóm ngành")
    ]

    first = tables[0].table
    assert first.header_row == [
        "Stt",
        "Tên ngành / Nhóm ngành",
        "Mã ngành > CT chuẩn",
        "Mã ngành > CT TC Tiếng Anh",
        "Tổ hợp (gồm 3 môn) > Bắt buộc",
        "Tổ hợp (gồm 3 môn) > Tự chọn (chọn 1)",
    ]
    assert check_table_invariants(first) == []
    codes = ["7510301", "7510303", "7510302", "7480108", "7510201"]
    codes += ["7510203", "7510202", "7510205", "7510206"]
    # section a) is one table over pages 2-4 (each page breaks it early)
    assert len(tables) == 1
    assert (tables[0].page_start, tables[0].page_end) == (2, 4)
    page_two = first.rows[:9]
    assert [row.cells[0] for row in page_two] == [str(n) for n in range(1, 10)]
    assert [row.cells[2] for row in page_two] == codes
    assert [row.cells[3] for row in page_two] == [f"{code}C" for code in codes]
    # every row of both merged `Toán, Vật lí | Nhóm môn TC1` blocks carries them
    assert all(row.cells[4:] == ["Toán, Vật lí", "Nhóm môn TC1"] for row in page_two)
    assert first.rows[6].cells[1] == "Công nghệ chế tạo máy"  # no glyph shards
    assert not any(row.warnings for row in first.rows)

    # page 3: shuffled markdown rows still line up with the geometry
    rows = first.rows[9:]
    it_rows = [row for row in rows if row.cells[0] == "13"]
    assert [row.cells[4:] for row in it_rows] == [
        ["Toán, Vật lí", "Nhóm môn TC1"],
        ["Toán, Ngữ văn", "Nhóm môn TC12"],
    ]
    assert "KT Phần mềm**" in it_rows[1].cells[1]  # the document's own mark, kept
    assert next(row for row in rows if row.cells[0] == "19").cells[4] == (
        "Toán, Hóa học hoặc Toán, Sinh học"
    )


GUIDE = _sample("HDSD_Thisinh*.pdf")


@pytest.mark.skipif(GUIDE is None, reason="sample PDF not available")
def test_user_guide_tables_have_their_real_columns() -> None:
    assert GUIDE is not None
    regions = split_regions(GUIDE.read_bytes(), GUIDE.name, "pdf")
    tables = {
        region.heading_path[-1]: region.table
        for region in regions
        if region.region_type == RegionType.TABLE and region.table
    }

    terms = tables["1.3. Các thuật ngữ và từ viết tắt"]
    assert terms.header_row == ["STT", "Cụm từ", "Từ viết tắt"]  # not "Cụm từ<br>Từ viết tắt"
    assert terms.rows[0].cells == ["1", "Điểm tiếp nhận hồ sơ", "Điểm TNHS"]

    functions = tables["3.1. Các chức năng trong Phân hệ"]
    # double borders cut 12 grid columns; the table has 4
    assert functions.header_row == ["STT", "Chức năng", "Mô tả", "Đối tượng sử dụng"]
    assert functions.rows[-1].cells == [
        "14",
        "Lịch sửa giao dịch",
        "Lịch sửa giao dịch",
        "Thí sinh",
    ]
    for table in (terms, functions):
        assert check_table_invariants(table) == []
        assert not any(row.warnings for row in table.rows)
    # the heading the markdown glued into the last row is kept, as text after the table
    after = regions[regions.index(next(r for r in regions if r.table is functions)) + 1]
    assert after.region_type == RegionType.TEXT
    assert after.content == "4. HƯỚNG DẪN SỬ DỤNG CÁC CHỨC NĂNG HỆ THỐNG"


FINTECH = _sample("KHUNG FINTECH K20.pdf")


@pytest.mark.skipif(FINTECH is None, reason="sample PDF not available")
def test_fintech_curriculum_elective_group_is_one_row() -> None:
    assert FINTECH is not None
    table = _tables(FINTECH)[0].table

    assert check_table_invariants(table) == []
    labels = [row.cells[0] for row in table.rows]
    # the header repeated on page 2 is not a data row
    assert "STT" not in labels
    elective = next(row for row in table.rows if row.cells[0].startswith("Học phần tự chọn (Sinh"))
    assert elective.cells[0] == "Học phần tự chọn (Sinh viên chọn 1 trong các học phần sau đây)"
    assert elective.cells[4] == "3"
    # two courses the markdown glued into one row (`1<br>2`) are two rows again
    position = table.rows.index(elective)
    first, second = table.rows[position + 1], table.rows[position + 2]
    assert (first.cells[0], first.cells[1]) == ("1", "2112011")
    assert (second.cells[0], second.cells[1]) == ("2", "2111491")
    assert second.ancestors[-1] == elective.cells[0]
    assert not any(row.warnings for row in table.rows)
