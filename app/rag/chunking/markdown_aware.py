from dataclasses import dataclass, field

from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk


@dataclass(frozen=True)
class MarkdownAwareChunker:
    """DEPRECATED alias of `RecursiveChunker`.

    Before Phase 3 (see changes/13-09-2026-Chunking-Structural-Metadata),
    this kept every TABLE region as one verbatim chunk, bypassing size-based
    splitting entirely - which is exactly the "table gets cut mid-row by a
    naive size-based chunker" failure mode this whole change set exists to
    fix at the source (`strategy.dispatch()` now always routes TABLE
    regions through `TableRowChunker`, never through this or any other text
    chunker - see Task 3.1). With that special-casing now redundant,
    `ChunkingStrategyName.MARKDOWN_AWARE` is kept only for API/enum
    backward compatibility and behaves identically to `RECURSIVE`.
    """

    text_chunker: RecursiveChunker = field(default_factory=RecursiveChunker)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        return self.text_chunker.split(regions)
