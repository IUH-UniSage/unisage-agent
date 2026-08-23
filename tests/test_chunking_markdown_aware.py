from app.rag.chunking.markdown_aware import MarkdownAwareChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import RegionType


def test_split_emits_one_chunk_per_table_region_verbatim() -> None:
    chunker = MarkdownAwareChunker(text_chunker=RecursiveChunker(chunk_size=1000, overlap=0))
    table_markdown = "|Name|Score|\n|---|---|\n|Alice|90|"

    chunks = chunker.split([ParsedRegion(RegionType.TABLE, table_markdown)])

    assert len(chunks) == 1
    assert chunks[0].content == table_markdown
    assert chunks[0].region_type == RegionType.TABLE


def test_split_chunks_text_regions_independently_of_tables() -> None:
    chunker = MarkdownAwareChunker(text_chunker=RecursiveChunker(chunk_size=1000, overlap=0))
    regions = [
        ParsedRegion(RegionType.TEXT, "Intro paragraph."),
        ParsedRegion(RegionType.TABLE, "|a|b|\n|---|---|\n|1|2|"),
        ParsedRegion(RegionType.TEXT, "Outro paragraph."),
    ]

    chunks = chunker.split(regions)

    assert [chunk.region_type for chunk in chunks] == [
        RegionType.TEXT,
        RegionType.TABLE,
        RegionType.TEXT,
    ]
    assert chunks[0].content == "Intro paragraph."
    assert chunks[2].content == "Outro paragraph."
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
