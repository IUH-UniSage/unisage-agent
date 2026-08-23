import tiktoken

from app.rag.chunking.token_based import TokenBasedChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import RegionType

_ENCODING = tiktoken.get_encoding("cl100k_base")


def test_split_respects_token_based_chunk_size() -> None:
    chunker = TokenBasedChunker(chunk_size=10, overlap=2)
    long_text = " ".join(f"word{i}" for i in range(50))

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, long_text)])

    assert chunks
    assert all(len(_ENCODING.encode(chunk.content)) <= 10 for chunk in chunks)


def test_split_produces_overlapping_token_windows() -> None:
    chunker = TokenBasedChunker(chunk_size=10, overlap=3)
    long_text = " ".join(f"word{i}" for i in range(30))

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, long_text)])

    assert len(chunks) > 1
    first_tail_tokens = _ENCODING.encode(chunks[0].content)[-3:]
    second_head_tokens = _ENCODING.encode(chunks[1].content)[:3]
    assert first_tail_tokens == second_head_tokens


def test_split_carries_region_type_through_to_chunks() -> None:
    chunker = TokenBasedChunker(chunk_size=100, overlap=0)

    chunks = chunker.split([ParsedRegion(RegionType.TABLE, "|a|b|")])

    assert all(chunk.region_type == RegionType.TABLE for chunk in chunks)
