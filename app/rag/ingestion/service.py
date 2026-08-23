from dataclasses import dataclass

from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.parser import parse_text_document
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import RegionType


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
        chunks = self._chunker.split([ParsedRegion(RegionType.TEXT, document.content)])
        return IngestionResult(document.source, [chunk.content for chunk in chunks])
