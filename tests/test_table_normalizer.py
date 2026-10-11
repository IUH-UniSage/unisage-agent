import pymupdf
import pytest

from app.rag.ingestion.canonical_table import (
    EventKind,
    RowDisposition,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import (
    EvidenceCell,
    EvidenceRow,
    TableEvidence,
    collect_table_evidence,
)
from app.rag.ingestion.table_normalizer import _align_rows_fuzzy, build_page_table
from app.schemas.ingestion import RegionType
from tests.test_table_evidence import SAMPLE, draw_table


def _table(lines: list[str], page: int = 1):
    return build_page_table(lines, page, None)


def test_single_tier_header_is_unchanged() -> None:
    table = _table(["|Name|Score|", "|---|---|", "|Alice|90|", "|Bob|85|"])

    assert table.header_row == ["Name", "Score"]
    assert table.header_levels == [["Name", "Score"]]
    assert table.data_rows == [["Alice", "90"], ["Bob", "85"]]
    assert check_table_invariants(table) == []


def test_two_tier_markdown_header_with_cut_merged_cell_is_repaired() -> None:
    table = _table(
        [
            "|TT|Label|Year 2|025 - 2026|",
            "|---|---|---|---|",
            "|||Per credit|Per year|",
            "|1|Alpha|100|200|",
            "||Beta|300||",
        ]
    )

    assert table.header_row == [
        "TT",
        "Label",
        "Year 2025 - 2026 > Per credit",
        "Year 2025 - 2026 > Per year",
    ]
    assert len(table.header_levels) == 2
    assert table.data_rows == [["1", "Alpha", "100", "200"], ["", "Beta", "300", ""]]
    kinds = {event.kind for event in table.normalization_events}
    assert EventKind.FILL_MERGED_HEADER in kinds
    assert EventKind.MERGE_CELLS in kinds
    assert check_table_invariants(table) == []


def test_three_column_row_is_padded_to_the_header_width() -> None:
    table = _table(["|A|B|C|", "|---|---|---|", "|1|2|", "|3|4|5|"])

    assert table.data_rows == [["1", "2", ""], ["3", "4", "5"]]
    assert "padded_cells" in table.rows[0].warnings
    assert table.rows[0].confidence < 1.0
    assert check_table_invariants(table) == []


def test_row_with_extra_cells_is_merged_into_the_last_cell_without_losing_text() -> None:
    table = _table(["|A|B|", "|---|---|", "|1|2|3|"])

    assert table.data_rows == [["1", "2 3"]]
    assert "merged_extra_cells" in table.rows[0].warnings
    assert check_table_invariants(table) == []


def test_blank_rows_are_dropped_with_a_ledger_entry() -> None:
    table = _table(["|A|B|", "|---|---|", "|||", "|1|2|"])

    assert table.data_rows == [["1", "2"]]
    dropped = [s for s in table.source_rows if s.disposition == RowDisposition.DROPPED_EMPTY]
    assert len(dropped) == 1
    assert check_table_invariants(table) == []


def test_garbled_row_is_kept_with_raw_text_warning_and_low_confidence() -> None:
    table = _table(["|Name|Fee|", "|---|---|", "|~~o~~<br>~~h~~<br>~~ha h~~|~~1 130 000~~|"])

    assert len(table.rows) == 1
    assert "garbled_text_raw_kept" in table.rows[0].warnings
    assert table.rows[0].confidence <= 0.3
    assert check_table_invariants(table) == []


def test_data_row_with_number_in_second_row_is_not_taken_for_a_header_tier() -> None:
    table = _table(
        ["|TT|Label|Fee|", "|---|---|---|", "||Alpha|100|", "||Beta|200|", "||Gamma|300|"]
    )

    assert len(table.header_levels) == 1
    assert table.data_rows == [["", "Alpha", "100"], ["", "Beta", "200"], ["", "Gamma", "300"]]


def _ruled_pdf(rows: list[list[str]]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    draw_table(page, rows)
    data = doc.tobytes()
    doc.close()
    return data


def test_geometry_supplies_clean_cell_text_and_events_for_a_real_pdf_table() -> None:
    content = _ruled_pdf(
        [["TT", "Name", "Fee", "Year"], ["1", "Alpha beta", "10", "20"], ["2", "Gamma", "30", "40"]]
    )

    regions = split_regions(content, "t.pdf", "pdf")
    table = next(r.table for r in regions if r.region_type == RegionType.TABLE)

    assert table is not None
    assert table.header_row == ["TT", "Name", "Fee", "Year"]
    assert table.data_rows == [["1", "Alpha beta", "10", "20"], ["2", "Gamma", "30", "40"]]
    assert check_table_invariants(table) == []


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_sample_pdf_pages_share_columns_two_tier_header_and_hold_all_invariants() -> None:
    regions = split_regions(SAMPLE.read_bytes(), SAMPLE.name, "pdf")
    tables = [r.table for r in regions if r.region_type == RegionType.TABLE and r.table]

    assert len(tables) == 1  # the 3 page fragments are one logical table (see test_table_merger)
    table = tables[0]
    assert table.header_row == [
        "TT",
        "HỆ ĐÀO TẠO",
        "Năm học 2025 - 2026 > Mức thu 01 tín chỉ",
        "Năm học 2025 - 2026 > Mức thu theo năm học",
    ]
    assert len(table.header_levels) == 2
    assert all(len(row.cells) == 4 for row in table.rows)
    assert check_table_invariants(table) == []
    # the second header tier must not leak into the data
    assert not any("Mức thu 01 tín chỉ" in row.cells for row in table.rows)

    # page 3: markdown collapsed to 3 cells (TT glued to the label); geometry restored 4
    page3_rows = [row for row in table.rows if row.page_start == 3]
    assert page3_rows[0].cells == ["", "Môn lý thuyết", "980.000", ""]
    assert any(e.kind == EventKind.SPLIT_CELL for e in table.normalization_events)
    garbled = [row for row in table.rows if "garbled_text_raw_kept" in row.warnings]
    assert len(garbled) == 1
    assert garbled[0].cells[2] == "1 130 000"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_evidence_that_does_not_align_falls_back_to_markdown_with_a_warning() -> None:
    doc = pymupdf.open(SAMPLE)
    evidence = collect_table_evidence(doc)
    doc.close()
    table_evidence = evidence[2][0]
    lines = ["|TT|HỆĐÀO TẠO|Năm học 2|025 - 2026|", "|---|---|---|---|", "|1|x||1|"]

    table = build_page_table(lines, 2, table_evidence)

    assert "evidence_unaligned" in table.warnings
    assert check_table_invariants(table) == []


def _cell(x0: float, y0: float, x1: float, y1: float, text: str) -> EvidenceCell:
    return EvidenceCell(bbox=(x0, y0, x1, y1), text=text)


def test_fuzzy_alignment_keeps_a_garbled_header_off_a_label_merged_down_several_rows() -> None:
    # The label cell is merged down over three geometry rows and holds nearly all
    # the text; the markdown adds a garbled caption row the geometry never saw.
    # Its letters also occur in the label, but pairing it with the label's rows
    # would leave the header's own text unmatched.
    label = " ".join(["Phat trien nguon nhan luc ke toan kiem toan du ve so luong"] * 6)
    rows = [
        EvidenceRow([None, _cell(200, 0, 220, 10, "1"), _cell(220, 0, 240, 10, "2")], 0, 10),
        EvidenceRow(
            [
                _cell(50, 10, 200, 40, label),
                _cell(200, 10, 220, 20, ""),
                _cell(220, 10, 240, 20, ""),
            ],
            10,
            20,
        ),
        EvidenceRow(
            [None, _cell(200, 20, 220, 30, "x"), _cell(220, 20, 240, 30, "")],
            20,
            30,
            [(1, 0), None, None],
        ),
        EvidenceRow(
            [None, _cell(200, 30, 220, 40, ""), _cell(220, 30, 240, 40, "x")],
            30,
            40,
            [(1, 0), None, None],
        ),
    ]
    evidence = TableEvidence(
        page_number=1,
        bbox=(50, 0, 240, 40),
        page_height=800,
        rows=rows,
        column_bounds=[50, 200, 220],
    )
    md = [["Chien luoc chinh sach", "Ch", "uan dau ra"], ["", "1", "2"], [label, "x", "x"]]

    assert _align_rows_fuzzy(md, evidence) == [[0, 1], [2], [2], [2]]


def test_line_break_inside_a_header_cell_is_not_part_of_the_column_name() -> None:
    table = build_page_table(["|**Học**<br>**kỳ**|Tên<br>môn|", "|---|---|", "|1|a|"], 1, None)

    assert table.header_row == ["Học kỳ", "Tên môn"]
