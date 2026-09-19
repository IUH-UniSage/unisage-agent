from dataclasses import replace

import pytest

from app.rag.chunking.table_row import TableRowChunker
from app.rag.ingestion.canonical_table import (
    TableBlock,
    build_table_from_plain_rows,
)
from app.rag.ingestion.table_aware_parser import ParsedRegion, split_regions
from app.schemas.ingestion import HeaderSource, RegionType, SourceType
from tests.test_table_evidence import SAMPLE

HEADER = ["Label", "Fee"]


def _table(rows: list[tuple[list[str], list[str], int]], header: list[str] = HEADER) -> TableBlock:
    """rows = (cells, ancestors, page)."""

    base = build_table_from_plain_rows(
        header, [cells for cells, _, _ in rows], HeaderSource.INFERRED, 0.75
    )
    canonical = [
        replace(row, ancestors=list(ancestors), page_start=page, page_end=page)
        for row, (_, ancestors, page) in zip(base.rows, rows, strict=True)
    ]
    return replace(base, rows=canonical, table_id="table-3")


def _region(table: TableBlock, heading: list[str] | None = None) -> ParsedRegion:
    return ParsedRegion(
        RegionType.TABLE,
        "",
        heading_path=heading or [],
        page_start=1,
        page_end=2,
        block_index=7,
        source_type=SourceType.PDF,
        table=table,
    )


def test_two_parents_in_one_chunk_are_both_kept_with_their_own_children() -> None:
    table = _table(
        [
            (["Theory", "980"], ["2025", "Technology"], 1),
            (["Practice", "1600"], ["2025", "Technology"], 1),
            (["Theory", "750"], ["2025", "Economics"], 1),
            (["Practice", "790"], ["2025", "Economics"], 1),
        ]
    )

    chunks = TableRowChunker(max_tokens=400).split([_region(table)])

    assert len(chunks) == 1
    lines = chunks[0].content.splitlines()
    tech = lines.index("[2025 > Technology]")
    eco = lines.index("[2025 > Economics]")
    assert lines[tech + 1 : tech + 3] == ["| Theory | 980 |", "| Practice | 1600 |"]
    assert lines[eco + 1 : eco + 3] == ["| Theory | 750 |", "| Practice | 790 |"]
    assert tech < eco


def test_rows_with_the_same_ancestors_share_one_breadcrumb_line() -> None:
    table = _table(
        [
            (["A", "1"], ["Top", "Mid"], 1),
            (["B", "2"], ["Top", "Mid"], 1),
            (["C", "3"], ["Top", "Mid"], 1),
        ]
    )

    content = TableRowChunker(max_tokens=400).split([_region(table)])[0].content

    assert content.count("[Top > Mid]") == 1


def test_a_row_without_ancestors_after_a_group_gets_an_explicit_marker() -> None:
    table = _table(
        [
            (["A", "1"], ["Top"], 1),
            (["B", "2"], [], 1),
            (["C", "3"], [], 1),
        ]
    )

    lines = TableRowChunker(max_tokens=400).split([_region(table)])[0].content.splitlines()

    assert lines[-5:] == ["[Top]", "| A | 1 |", "[-]", "| B | 2 |", "| C | 3 |"]


def test_content_carries_every_flattened_column_name_and_every_ancestor() -> None:
    header = ["TT", "Name", "Year 2025 > Per credit", "Year 2025 > Per year"]
    table = _table(
        [
            (["", "Theory", "980", ""], ["A. Main campus", "3 Regular", "3.1 Intake 2025"], 1),
            (["", "Practice", "1600", ""], ["A. Main campus", "3 Regular", "3.1 Intake 2025"], 2),
        ],
        header=header,
    )

    chunk = TableRowChunker(max_tokens=400).split([_region(table, ["Decision", "Fees"])])[0]

    for name in header:
        assert name in chunk.content
    for ancestor in ("A. Main campus", "3 Regular", "3.1 Intake 2025"):
        assert ancestor in chunk.content
    assert chunk.content.startswith("Decision > Fees\n\n")
    assert chunk.column_names == header


def test_chunk_pages_and_rows_follow_the_rows_it_holds() -> None:
    rows = [([f"row{i}", str(i)], ["Top"], 1 if i <= 3 else 2) for i in range(1, 7)]
    table = _table(rows)

    chunks = TableRowChunker(max_tokens=400).split([_region(table)])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    locator = chunk.source_locator
    assert (locator.table_id, locator.row_start, locator.row_end, locator.row_count) == (
        "table-3",
        1,
        6,
        6,
    )


def test_a_small_budget_splits_rows_and_each_chunk_reopens_its_breadcrumb() -> None:
    rows = [([f"row number {i}", "x" * 20], ["Top > Sub"], 1 if i <= 4 else 2) for i in range(1, 9)]
    table = _table(rows)

    chunks = TableRowChunker(max_tokens=90).split([_region(table)])

    assert len(chunks) > 1
    assert all("[Top > Sub]" in chunk.content for chunk in chunks)
    starts = [c.source_locator.row_start for c in chunks]
    ends = [c.source_locator.row_end for c in chunks]
    assert starts[0] == 1 and ends[-1] == 8
    assert all(starts[i + 1] == ends[i] + 1 for i in range(len(chunks) - 1))
    page_of_row = {i: 1 if i <= 4 else 2 for i in range(1, 9)}
    for chunk in chunks:
        locator = chunk.source_locator
        assert chunk.page_start == page_of_row[locator.row_start]
        assert chunk.page_end == page_of_row[locator.row_end]


def test_a_very_long_breadcrumb_is_shortened_from_the_root_and_flagged() -> None:
    ancestors = [f"Level {i} " + "word " * 12 for i in range(1, 7)]
    table = _table([(["Leaf", "1"], ancestors, 1), (["Leaf2", "2"], ancestors, 1)])

    chunks = TableRowChunker(max_tokens=200).split([_region(table)])

    for chunk in chunks:
        assert "[… > " in chunk.content
        assert "Level 6" in chunk.content  # the nearest ancestor is kept
        assert "Level 1 " not in chunk.content  # the farthest one is dropped
        assert "breadcrumb_truncated" in chunk.parse_warnings


def test_oversized_row_parts_keep_the_full_breadcrumb() -> None:
    table = _table([(["Big", "word " * 900], ["Top", "Sub"], 1)])

    chunks = TableRowChunker(max_tokens=200).split([_region(table)])

    assert len(chunks) > 1
    for part in chunks:
        assert part.source_locator.is_partial_row
        assert "[Top > Sub]" in part.content
        assert (part.source_locator.row_start, part.source_locator.row_end) == (1, 1)


def test_structure_confidence_and_warnings_are_taken_from_the_rows() -> None:
    base = _table([(["A", "1"], [], 1), (["B", "2"], [], 1)])
    rows = [
        replace(base.rows[0], confidence=0.9, warnings=["padded_cells"]),
        replace(base.rows[1], confidence=0.3, warnings=["garbled_text_raw_kept", "padded_cells"]),
    ]
    table = replace(base, rows=rows, warnings=["merge_borderline"])

    chunk = TableRowChunker(max_tokens=400).split([_region(table)])[0]

    assert chunk.structure_confidence == pytest.approx(0.3)
    assert chunk.parse_warnings == ["padded_cells", "garbled_text_raw_kept", "merge_borderline"]


def test_hand_built_tables_without_provenance_keep_their_old_output() -> None:
    table = TableBlock(
        header_row=HEADER,
        data_rows=[["a", "1"]],
        header_source=HeaderSource.EXPLICIT,
        header_confidence=1.0,
    )

    chunk = TableRowChunker(max_tokens=400).split([_region(table)])[0]

    assert chunk.content == "| Label | Fee |\n| --- | --- |\n| a | 1 |"
    assert chunk.structure_confidence is None
    assert chunk.parse_warnings == []


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_sample_pdf_chunk_crossing_the_page_break_keeps_full_context() -> None:
    regions = split_regions(SAMPLE.read_bytes(), SAMPLE.name, "pdf")
    chunks = TableRowChunker(max_tokens=400).split(regions)

    crossing = [c for c in chunks if c.page_start != c.page_end]
    assert crossing
    for chunk in chunks:
        assert "Năm học 2025 - 2026 > Mức thu 01 tín chỉ" in chunk.content
        assert "Năm học 2025 - 2026 > Mức thu theo năm học" in chunk.content
        assert chunk.source_locator.table_id == "table-0"
        assert chunk.source_locator.row_count == (
            chunk.source_locator.row_end - chunk.source_locator.row_start + 1
        )

    first_page3 = next(c for c in chunks if c.page_start == 3)
    assert "3 Đại học chính quy" in first_page3.content
    assert "3.1 Khóa tuyển sinh năm học 2025-2026" in first_page3.content

    garbled = [c for c in chunks if "garbled_text_raw_kept" in c.parse_warnings]
    assert len(garbled) == 1
    assert garbled[0].structure_confidence <= 0.3
    assert "1 130 000" in garbled[0].content  # kept, never dropped
