from collections.abc import Sequence

from app.schemas.retrieval import RetrievedChunk


def rerank(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the base ranking deterministic until a cross-encoder is configured."""

    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
