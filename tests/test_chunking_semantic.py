import tiktoken

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
