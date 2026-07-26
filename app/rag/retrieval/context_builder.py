from collections.abc import Sequence

from app.schemas.retrieval import RetrievedChunk


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks into the context format consumed by generation."""

    return "\n\n".join(f"[{chunk.source}] {chunk.content}" for chunk in chunks)
