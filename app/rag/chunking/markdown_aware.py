from dataclasses import dataclass, field
from typing import Protocol

from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk, RegionType


class _TextChunker(Protocol):
    def split(self, regions: list[ParsedRegion]) -> list[Chunk]: ...


@dataclass(frozen=True)
class MarkdownAwareChunker:
    """Chunk text regions normally; keep each table region as one verbatim chunk."""

    text_chunker: _TextChunker = field(default_factory=RecursiveChunker)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for region in regions:
            if region.region_type == RegionType.TABLE:
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=region.content,
                        region_type=RegionType.TABLE,
                    )
                )
                continue
            for chunk in self.text_chunker.split([region]):
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=chunk.content,
                        region_type=chunk.region_type,
                    )
                )
        return chunks
