from dataclasses import dataclass, field

from app.rag.chunking.recursive import RecursiveChunker
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk


@dataclass(frozen=True)
class MarkdownAwareChunker:
    """DEPRECATED alias of `RecursiveChunker`.

    `strategy.dispatch()` always routes TABLE regions through
    `TableRowChunker`, so this chunker's old table special-casing is
    redundant. `ChunkingStrategyName.MARKDOWN_AWARE` is kept only for
    API/enum backward compatibility and behaves identically to `RECURSIVE`.
    """

    text_chunker: RecursiveChunker = field(default_factory=RecursiveChunker)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        return self.text_chunker.split(regions)
