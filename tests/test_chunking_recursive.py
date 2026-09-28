import pytest

from app.core.errors.exceptions import ChunkingConfigException
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import RegionType


def test_split_preserves_trailing_content_and_overlap() -> None:
    chunker = RecursiveChunker(chunk_size=5, overlap=1)

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, "abcdefghij")])

    assert chunks[-1].content == "ij"
    reconstructed = chunks[0].content + "".join(chunk.content[1:] for chunk in chunks[1:])
    assert reconstructed == "abcdefghij"


def test_split_carries_region_type_through_to_chunks() -> None:
    chunker = RecursiveChunker(chunk_size=100, overlap=0)

    chunks = chunker.split([ParsedRegion(RegionType.TABLE, "|a|b|")])

    assert all(chunk.region_type == RegionType.TABLE for chunk in chunks)


def test_split_skips_empty_regions() -> None:
    chunker = RecursiveChunker(chunk_size=5, overlap=1)

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, "   ")])

    assert chunks == []


def test_split_assigns_sequential_chunk_index_across_regions() -> None:
    chunker = RecursiveChunker(chunk_size=100, overlap=0)

    chunks = chunker.split(
        [ParsedRegion(RegionType.TEXT, "first"), ParsedRegion(RegionType.TEXT, "second")]
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_rejects_overlap_greater_than_or_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RecursiveChunker(chunk_size=5, overlap=5)


def test_every_sub_chunk_of_a_long_region_carries_the_heading_prefix() -> None:
    chunker = RecursiveChunker(chunk_size=40, overlap=2)
    region = ParsedRegion(
        RegionType.TEXT,
        "a" * 100,
        heading_path=["Section A", "Sub B"],
    )

    chunks = chunker.split([region])

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.content.startswith("Section A > Sub B\n\n")
        assert len(chunk.content) <= 40


def test_heading_prefix_too_long_for_chunk_size_raises_config_exception() -> None:
    chunker = RecursiveChunker(chunk_size=10, overlap=1)
    region = ParsedRegion(
        RegionType.TEXT,
        "some body text",
        heading_path=["A very very very long heading section"],
    )

    with pytest.raises(ChunkingConfigException):
        chunker.split([region])


def test_split_copies_structural_fields_from_region_to_every_chunk() -> None:
    from app.schemas.ingestion import SourceType

    chunker = RecursiveChunker(chunk_size=100, overlap=0)
    region = ParsedRegion(
        RegionType.TEXT,
        "body text",
        heading_path=["A"],
        page_start=3,
        page_end=3,
        block_index=5,
        source_type=SourceType.PDF,
    )

    [chunk] = chunker.split([region])

    assert chunk.source_type == SourceType.PDF
    assert chunk.block_index == 5
    assert chunk.page_start == 3
    assert chunk.page_end == 3
    assert chunk.heading_path == ["A"]
    assert chunk.source_locator is not None
    assert chunk.source_locator.section == "A"
    assert chunk.chunking_version != "legacy"
