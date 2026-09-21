import pymupdf
import pytest

from app.rag.ingestion.canonical_table import (
    EventKind,
    RowDisposition,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import collect_table_evidence
from app.rag.ingestion.table_normalizer import build_page_table
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
