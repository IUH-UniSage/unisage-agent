from app.rag.chunking.excel_rows import ExcelRowChunker
from app.schemas.ingestion import RegionType
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
