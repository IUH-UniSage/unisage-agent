from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import ChunkingConfigException
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk, SourceLocator


def _heading_prefix(heading_path: list[str]) -> str:
    joined = " > ".join(heading_path)
    return f"{joined}\n\n" if joined else ""


@dataclass(frozen=True)
class RecursiveChunker:
    """Deterministic character-based chunker with configurable overlap."""

    chunk_size: int = 800
    overlap: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be between zero and chunk_size - 1")

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        """Split each region into overlapping chunks without dropping
        trailing content. Every chunk gets `region.heading_path` prepended
        into its own `content` (not just the region's first chunk) - the
        budget used for cutting is `chunk_size` MINUS the prefix's length,
        computed once per region, so the final prefixed content never
        exceeds `chunk_size`."""

        chunks: list[Chunk] = []
        for region in regions:
            clean_text = region.content.strip()
            if not clean_text:
                continue

            prefix = _heading_prefix(region.heading_path)
            effective_chunk_size = self.chunk_size - len(prefix)
            if effective_chunk_size <= self.overlap:
                raise ChunkingConfigException(
                    f"heading_path prefix is {len(prefix)} chars, leaving "
                    f"effective_chunk_size={effective_chunk_size} <= overlap={self.overlap} "
                    f"for chunk_size={self.chunk_size}. Increase chunk_size or "
                    "shorten heading_path."
                )
            step = effective_chunk_size - self.overlap

            for start in range(0, len(clean_text), step):
                body = clean_text[start : start + effective_chunk_size]
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=f"{prefix}{body}",
                        region_type=region.region_type,
                        source_type=region.source_type,
                        block_index=region.block_index,
                        heading_path=list(region.heading_path),
                        page_start=region.page_start,
                        page_end=region.page_end,
                        source_locator=SourceLocator(section=_section(region.heading_path)),
                        chunking_version=settings.INGEST_CHUNKING_VERSION,
                    )
                )
        return chunks


def _section(heading_path: list[str]) -> str | None:
    joined = " > ".join(heading_path)
    return joined or None
