from app.rag.chunking.excel_rows import ExcelRowChunker
from app.schemas.ingestion import HeaderSource, RegionType, SourceType
from tests.fixtures.documents import make_xlsx_bytes


def test_split_produces_one_chunk_per_row_with_header_prepended() -> None:
    content = make_xlsx_bytes(
        header=["Name", "Score"],
        rows=[["Alice", 90], ["Bob", 85], ["Carol", 78]],
    )
    chunker = ExcelRowChunker(rows_per_chunk=1)

    chunks = chunker.split(content)

    assert len(chunks) == 3
    assert all(chunk.region_type == RegionType.EXCEL_ROW for chunk in chunks)
    assert all(chunk.content.startswith("Name | Score") for chunk in chunks)
    assert "Alice | 90" in chunks[0].content
    assert "Bob | 85" in chunks[1].content
    assert "Carol | 78" in chunks[2].content


def test_split_groups_multiple_rows_per_chunk() -> None:
    content = make_xlsx_bytes(
        header=["Name", "Score"],
        rows=[["Alice", 90], ["Bob", 85], ["Carol", 78]],
    )
    chunker = ExcelRowChunker(rows_per_chunk=2)

    chunks = chunker.split(content)

    assert len(chunks) == 2
    assert "Alice | 90" in chunks[0].content
    assert "Bob | 85" in chunks[0].content
    assert "Carol | 78" in chunks[1].content


def test_split_sets_full_structural_metadata() -> None:
    content = make_xlsx_bytes(
        header=["Name", "Score"],
        rows=[["Alice", 90], ["Bob", 85]],
    )
    chunker = ExcelRowChunker(rows_per_chunk=1)

    chunks = chunker.split(content)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.source_type == SourceType.XLSX
        assert chunk.block_index == 0
        assert chunk.heading_path == []
        assert chunk.column_names == ["Name", "Score"]
        assert chunk.has_header is True
        assert chunk.header_source == HeaderSource.EXPLICIT
        assert chunk.header_confidence == 1.0
        assert chunk.source_locator is not None
        assert chunk.source_locator.table_id == "table-0"
    assert chunks[0].source_locator.row_start == 1  # type: ignore[union-attr]
    assert chunks[0].source_locator.row_end == 1  # type: ignore[union-attr]
    assert chunks[1].source_locator.row_start == 2  # type: ignore[union-attr]
    assert chunks[1].source_locator.row_end == 2  # type: ignore[union-attr]
