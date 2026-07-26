from collections.abc import Sequence

from app.schemas.chat import Citation
from app.schemas.retrieval import RetrievedChunk


def build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Convert internal chunks into the public citation contract."""

    return [
        Citation(title=chunk.source, chunk_id=chunk.chunk_id, source=chunk.source)
        for chunk in chunks
    ]
