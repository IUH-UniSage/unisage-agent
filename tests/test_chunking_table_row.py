import pytest
import tiktoken

from app.core.errors.exceptions import ChunkingConfigException
from app.rag.chunking.table_row import TableRowChunker, TableStructureError
from app.rag.ingestion.table_aware_parser import ParsedRegion, TableBlock
from app.schemas.ingestion import HeaderSource, RegionType, SourceType

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _table_region(
    header_row: list[str] | None,
    data_rows: list[list[str]],
    *,
    heading_path: list[str] | None = None,
    header_source: HeaderSource = HeaderSource.EXPLICIT,
    header_confidence: float = 1.0,
    block_index: int = 0,
) -> ParsedRegion:
    return ParsedRegion(
        RegionType.TABLE,
        "unused-raw-content",
        heading_path=heading_path or [],
        block_index=block_index,
        source_type=SourceType.HTML,
        table=TableBlock(
            header_row=header_row,
            data_rows=data_rows,
            header_source=header_source,
            header_confidence=header_confidence,
        ),
    )


def test_all_rows_are_reassembled_with_full_header_and_heading() -> None:
    header = ["Name", "Score"]
    rows = [[f"Person{i}", str(i)] for i in range(20)]
    region = _table_region(header, rows, heading_path=["Hoc phi"])
    chunker = TableRowChunker(max_tokens=60)

    chunks = chunker.split([region])

    assert len(chunks) > 1
    assert all(chunk.has_header for chunk in chunks)
    assert all(chunk.column_names == header for chunk in chunks)
    assert all(chunk.content.startswith("Hoc phi\n\n") for chunk in chunks)
    reassembled_row_count = 0
    seen_rows: set[int] = set()
    for chunk in chunks:
        locator = chunk.source_locator
        assert locator is not None
        assert locator.row_count is not None
        assert locator.row_start is not None
        assert locator.row_end is not None
        reassembled_row_count += locator.row_count
        for row_number in range(locator.row_start, locator.row_end + 1):
            assert row_number not in seen_rows
            seen_rows.add(row_number)
    assert reassembled_row_count == len(rows)
    assert seen_rows == set(range(1, len(rows) + 1))


def test_missing_header_table_keeps_all_rows_as_data() -> None:
    rows = [["a", "1"], ["b", "2"]]
    region = _table_region(None, rows, header_source=HeaderSource.MISSING, header_confidence=0.0)
    chunker = TableRowChunker(max_tokens=200)

    chunks = chunker.split([region])

    assert all(not chunk.has_header for chunk in chunks)
    assert all(chunk.column_names is None for chunk in chunks)
    assert all(chunk.header_source == HeaderSource.MISSING for chunk in chunks)


def test_normal_row_is_never_split() -> None:
    header = ["Name"]
    rows = [["Alice"], ["Bob"], ["Carol"]]
    region = _table_region(header, rows)
    chunker = TableRowChunker(max_tokens=30)

    chunks = chunker.split([region])

    for chunk in chunks:
        locator = chunk.source_locator
        assert locator is not None
        assert locator.is_partial_row is False
        assert locator.row_part is None
        assert locator.row_part_count is None


def test_content_never_exceeds_max_tokens_with_long_heading_path() -> None:
    header = ["Name", "Score"]
    rows = [[f"Person{i}", str(i)] for i in range(10)]
    heading_path = [f"Section level {i}" for i in range(8)]
    region = _table_region(header, rows, heading_path=heading_path)
    chunker = TableRowChunker(max_tokens=80)

    chunks = chunker.split([region])

    for chunk in chunks:
        assert len(_ENCODING.encode(chunk.content)) <= 80


def test_usable_budget_zero_or_negative_raises_config_exception() -> None:
    header = ["Name", "Score"]
    rows = [["Alice", "90"]]
    heading_path = [f"Very long heading section number {i} with lots of words" for i in range(20)]
    region = _table_region(header, rows, heading_path=heading_path)
    chunker = TableRowChunker(max_tokens=50)

    with pytest.raises(ChunkingConfigException):
        chunker.split([region])


def test_max_tokens_not_positive_raises_config_exception() -> None:
    with pytest.raises(ChunkingConfigException):
        TableRowChunker(max_tokens=0)


def test_structure_error_raised_on_mismatched_cell_count() -> None:
    header = ["Name", "Score"]
    rows = [["Alice", "90"], ["OnlyOneCell"]]
    region = _table_region(header, rows)
    chunker = TableRowChunker(max_tokens=200)

    with pytest.raises(TableStructureError) as excinfo:
        chunker.split([region])

    error = excinfo.value
    assert error.expected_cell_count == 2
    assert error.actual_cell_count == 1
    assert error.raw_row == ["OnlyOneCell"]
    # the redacted message/str must never contain the full raw row content
    assert "OnlyOneCell" not in str(error)


def test_oversized_row_beyond_hard_cap_is_split_into_marked_parts() -> None:
    header = ["Name", "Notes"]
    giant_notes = " ".join(f"word{i}" for i in range(500))
    rows = [["Alice", giant_notes]]
    region = _table_region(header, rows)
    chunker = TableRowChunker(max_tokens=40)

    chunks = chunker.split([region])

    assert len(chunks) > 1
    for chunk in chunks:
        locator = chunk.source_locator
        assert locator is not None
        assert locator.is_partial_row is True
        assert locator.row_part is not None
        assert locator.row_part_count == len(chunks)
        assert locator.table_id == "table-0"
        assert locator.row_start == 1
        assert locator.row_end == 1
    assert [c.source_locator.row_part for c in chunks] == list(  # type: ignore[union-attr]
        range(1, len(chunks) + 1)
    )


def test_escapes_pipe_and_normalizes_newline_in_cells() -> None:
    header = ["Name", "Note"]
    rows = [["Alice", "has | pipe\nand newline"]]
    region = _table_region(header, rows)
    chunker = TableRowChunker(max_tokens=200)

    [chunk] = chunker.split([region])

    assert "\\|" in chunk.content
    assert "<br>" in chunk.content
    # every content line should have the same number of unescaped pipes as
    # the header (no phantom extra column from an unescaped cell `|`)
    for line in chunk.content.splitlines():
        if line.startswith("|"):
            unescaped_pipe_count = len(line.replace("\\|", "")) - len(
                line.replace("\\|", "").replace("|", "")
            )
            assert unescaped_pipe_count == len(header) + 1
    locator = chunk.source_locator
    assert locator is not None
    assert locator.row_start == 1
    assert locator.row_end == 1
    assert locator.row_count == 1
