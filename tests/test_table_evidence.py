from pathlib import Path

import pymupdf
import pytest

from app.rag.ingestion.table_evidence import collect_table_evidence

SAMPLE = (
    Path(__file__).parent.parent / "_to_delete" / "Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf"
)


def draw_table(
    page: pymupdf.Page,
    rows: list[list[str]],
    *,
    top: float = 100,
    col_x: tuple[float, ...] = (50, 120, 300, 450),
    row_h: float = 20,
) -> None:
    """Draw a simple ruled table (one cell per rule box) for evidence tests."""

    for r, row in enumerate(rows):
        y = top + r * row_h
        for c, text in enumerate(row):
            x0 = col_x[c]
            x1 = col_x[c + 1] if c + 1 < len(col_x) else 545
            page.draw_rect(pymupdf.Rect(x0, y, x1, y + row_h), color=(0, 0, 0), width=0.8)
            if text:
                page.insert_text((x0 + 3, y + 14), text, fontsize=9)


def test_collects_rows_cells_and_clean_text_from_a_ruled_table() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    draw_table(page, [["TT", "Name", "Fee", "Year"], ["1", "Alpha beta", "10", "20"]])

    evidence = collect_table_evidence(doc)

    assert list(evidence) == [1]
    table = evidence[1][0]
    assert len(table.rows) == 2
    assert [cell.text for cell in table.rows[1].cells if cell] == ["1", "Alpha beta", "10", "20"]
    assert table.column_count == 4
    assert table.header_row_count == 1


def test_page_without_tables_has_no_evidence() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "just a sentence", fontsize=11)

    assert collect_table_evidence(doc) == {}


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_sample_pdf_evidence_finds_two_tier_header_and_merged_span() -> None:
    doc = pymupdf.open(SAMPLE)
    evidence = collect_table_evidence(doc)
    doc.close()

    for page_number in (2, 3, 4):
        table = evidence[page_number][0]
        assert table.column_count == 4
        assert table.header_row_count == 2
        first = table.rows[0].cells
        assert first[0] is not None and first[0].text == "TT"
        assert first[2] is not None and first[2].text == "Năm học 2025 - 2026"
        assert first[3] is None  # horizontal merge swallowed the 4th column
        assert [c.text for c in table.rows[1].cells if c] == [
            "Mức thu 01 tín chỉ",
            "Mức thu theo năm học",
        ]
    # page 3 has a TT column geometrically even though markdown drops it
    assert evidence[3][0].rows[2].cells[0] is not None
