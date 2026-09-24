from pathlib import Path

import pymupdf
import pytest

from app.rag.ingestion.table_evidence import (
    EvidenceCell,
    EvidenceRow,
    _trim_trailing_garbage_rows,
    collect_table_evidence,
)

SAMPLE = (
    Path(__file__).parent.parent / "_to_delete" / "Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf"
)
HDSD_SAMPLE = Path(__file__).parent.parent / "_to_delete" / "HDSD_Thisinh_DKXT_T72025_IUH.pdf"


def _row(y0: float, y1: float, *texts: str | None) -> EvidenceRow:
    cells = [
        None if text is None else EvidenceCell(bbox=(0.0, y0, 10.0, y1), text=text)
        for text in texts
    ]
    return EvidenceRow(cells=cells, y0=y0, y1=y1)


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


def test_trim_drops_a_trailing_row_that_is_tall_and_nearly_empty() -> None:
    """The exact shape confirmed on HDSD_Thisinh_DKXT_T72025_IUH.pdf: 5 normal
    ~15pt rows followed by one ~290pt row holding a single populated cell
    (absorbed paragraph text), all other cells empty."""

    rows = [
        _row(0, 15, "STT", "STT", "Cụm từ", "Từ viết tắt"),
        _row(15, 30, None, "1", "Điểm tiếp nhận hồ sơ", "Điểm TNHS"),
        _row(30, 45, None, "2", "Chứng minh thư nhân dân", "CMND"),
        _row(45, 320, None, "TỔNG QUAN VỀ SẢN PHẨM...", None, None),
    ]

    trimmed = _trim_trailing_garbage_rows(rows)

    assert trimmed == rows[:-1]


def test_trim_keeps_a_genuine_tall_full_width_note_row() -> None:
    """A real note row inside a table is at most a couple of wrapped lines
    taller than its neighbours - not an order of magnitude taller - so it
    must survive even though it also has just one populated cell."""

    rows = [
        _row(0, 15, "STT", "Name"),
        _row(15, 30, "1", "Alpha"),
        _row(30, 55, "Ghi chú: áp dụng cho khóa 2026 trở đi", None),
    ]

    assert _trim_trailing_garbage_rows(rows) == rows


def test_trim_never_reduces_a_table_below_two_rows() -> None:
    rows = [
        _row(0, 15, "STT", "Name"),
        _row(15, 320, "chỉ một dòng", None),
    ]

    assert _trim_trailing_garbage_rows(rows) == rows


@pytest.mark.skipif(not HDSD_SAMPLE.exists(), reason="sample PDF not available")
def test_hdsd_terminology_table_evidence_has_no_absorbed_paragraph_row() -> None:
    doc = pymupdf.open(HDSD_SAMPLE)
    evidence = collect_table_evidence(doc)
    doc.close()

    table = next(
        t
        for tables in evidence.values()
        for t in tables
        if any(cell is not None and cell.text == "Cụm từ" for row in t.rows for cell in row.cells)
    )

    # header row + 4 terms - the absorbed-paragraph 7th row is gone, and the
    # section title the box took in above the table is kept as a caption
    assert len(table.rows) == 5
    assert table.captions == ["Các thuật ngữ và từ viết tắt"]
    # the empty slivers the double borders cut at both edges are folded away
    assert [c.text if c else None for c in table.rows[1].cells] == [
        "1",
        "Điểm tiếp nhận hồ sơ",
        "Điểm TNHS",
    ]
