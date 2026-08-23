import pytest

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
