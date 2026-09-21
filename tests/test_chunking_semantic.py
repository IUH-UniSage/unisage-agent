import pytest
import tiktoken

from app.core.exceptions import ChunkingConfigException
from app.rag.chunking.semantic import SemanticChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import RegionType

_ENCODING = tiktoken.get_encoding("cl100k_base")


class _DeterministicEmbedder:
    """Fake embedder giving each sentence a similarity based on its first word."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(text.split()[0]) % 1000) / 1000.0, 1.0] for text in texts]


def _fixture_text(sentence_count: int, words_per_sentence: int = 20) -> str:
    sentences = []
    for i in range(sentence_count):
        words = " ".join(f"tok{i}_{j}" for j in range(words_per_sentence))
        sentences.append(f"{words}.")
    return " ".join(sentences)


def test_split_produces_chunks_averaging_close_to_target_tokens() -> None:
    chunker = SemanticChunker(
        target_tokens=100,
        overlap_ratio=0.2,
        similarity_threshold=-1.0,  # disable semantic-break splitting for this test
        embedder=_DeterministicEmbedder(),
    )
    text = _fixture_text(sentence_count=20, words_per_sentence=10)

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, text)])

    assert chunks
    token_counts = [len(_ENCODING.encode(chunk.content)) for chunk in chunks]
    average = sum(token_counts) / len(token_counts)
    assert 50 <= average <= 150


def test_split_carries_region_type_through_to_chunks() -> None:
    chunker = SemanticChunker(embedder=_DeterministicEmbedder())

    chunks = chunker.split([ParsedRegion(RegionType.TABLE, "One sentence. Another sentence.")])

    assert all(chunk.region_type == RegionType.TABLE for chunk in chunks)


def test_split_skips_regions_with_no_sentences() -> None:
    chunker = SemanticChunker(embedder=_DeterministicEmbedder())

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, "   ")])

    assert chunks == []


def test_no_chunk_exceeds_the_derived_hard_cap() -> None:
    chunker = SemanticChunker(
        target_tokens=100,
        overlap_ratio=0.2,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )
    # One un-splittable "sentence" (no . ! ?) far larger than the cap.
    giant = " ".join(f"tok_{j}" for j in range(600))

    chunks = chunker.split([ParsedRegion(RegionType.TEXT, giant)])

    assert chunks
    assert all(len(_ENCODING.encode(chunk.content)) <= chunker.max_tokens for chunk in chunks)


def test_tiny_trailing_region_is_merged_not_emitted_alone() -> None:
    chunker = SemanticChunker(
        target_tokens=100,
        min_tokens=48,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )
    body = _fixture_text(sentence_count=6, words_per_sentence=10)

    chunks = chunker.split(
        [
            ParsedRegion(RegionType.TEXT, body),
            ParsedRegion(RegionType.TEXT, "a. Den ngay co."),
        ]
    )

    assert chunks
    assert all(len(_ENCODING.encode(chunk.content)) >= chunker.min_tokens for chunk in chunks)
    assert any("Den ngay co" in chunk.content for chunk in chunks)


def test_every_chunk_carries_heading_prefix_within_budget() -> None:
    chunker = SemanticChunker(
        target_tokens=60,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )
    text = _fixture_text(sentence_count=20, words_per_sentence=10)
    region = ParsedRegion(RegionType.TEXT, text, heading_path=["Section A"], block_index=0)

    chunks = chunker.split([region])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.content.startswith("Section A\n\n")
        assert len(_ENCODING.encode(chunk.content)) <= chunker.max_tokens


def test_heading_prefix_too_long_raises_config_exception() -> None:
    chunker = SemanticChunker(
        target_tokens=10,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )
    heading = " ".join(f"heading_word_{i}" for i in range(30))
    region = ParsedRegion(RegionType.TEXT, "short body.", heading_path=[heading])

    with pytest.raises(ChunkingConfigException):
        chunker.split([region])


def test_small_chunks_from_different_regions_are_not_merged_together() -> None:
    """Two small TEXT regions with different headings/block_index must never
    be glued into one chunk - that would make the resulting chunk falsely
    claim a single heading for content from two different sections."""

    chunker = SemanticChunker(
        target_tokens=1000,
        min_tokens=48,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )
    region_a = ParsedRegion(
        RegionType.TEXT, "a. Noi dung A.", heading_path=["Section A"], block_index=0
    )
    region_b = ParsedRegion(
        RegionType.TEXT, "a. Noi dung B.", heading_path=["Section B"], block_index=1
    )

    chunks = chunker.split([region_a, region_b])

    assert len(chunks) == 2
    assert chunks[0].content.startswith("Section A\n\n")
    assert chunks[1].content.startswith("Section B\n\n")
    assert "Noi dung B" not in chunks[0].content
    assert "Noi dung A" not in chunks[1].content


def test_lone_list_marker_is_not_split_into_its_own_sentence() -> None:
    chunker = SemanticChunker(
        target_tokens=100,
        min_tokens=1,
        similarity_threshold=-1.0,
        embedder=_DeterministicEmbedder(),
    )

    chunks = chunker.split(
        [ParsedRegion(RegionType.TEXT, "Nguoi lao dong can lam gi? a. Den ngay co so y te.")]
    )

    assert chunks
    joined = " ".join(chunk.content for chunk in chunks)
    assert "a. Den ngay co so y te." in joined
