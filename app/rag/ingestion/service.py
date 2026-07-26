from dataclasses import dataclass

from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.parser import parse_text_document


@dataclass(frozen=True)
class IngestionResult:
    source: str
    chunks: list[str]


class IngestionService:
    """Coordinate parsing and chunking for a document."""

    def __init__(self, chunker: RecursiveChunker | None = None) -> None:
        self._chunker = chunker or RecursiveChunker()

    def ingest(
        self,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> IngestionResult:
        document = parse_text_document(source, content, metadata)
        return IngestionResult(document.source, self._chunker.split(document.content))
