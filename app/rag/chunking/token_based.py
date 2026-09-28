from dataclasses import dataclass

import tiktoken

from app.core.config import settings
from app.core.errors.exceptions import ChunkingConfigException
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk, SourceLocator

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _heading_prefix(heading_path: list[str]) -> str:
    joined = " > ".join(heading_path)
    return f"{joined}\n\n" if joined else ""


def _section(heading_path: list[str]) -> str | None:
    joined = " > ".join(heading_path)
    return joined or None


@dataclass(frozen=True)
class TokenBasedChunker:
    """Chunker that splits on real token counts (via tiktoken) instead of characters."""

    chunk_size: int = 400
    overlap: int = 40

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be between zero and chunk_size - 1")

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        """Split each region into overlapping chunks bounded by token count.

        `region.heading_path` is prepended into EVERY sub-chunk's content
        (not just the first), with its token cost subtracted from
        `chunk_size` BEFORE cutting - `usable_chunk_size` - so the final
        prefixed content's real token count never exceeds `chunk_size`.
        """

        chunks: list[Chunk] = []
        for region in regions:
            clean_text = region.content.strip()
            if not clean_text:
                continue

            prefix = _heading_prefix(region.heading_path)
            prefix_tokens = len(_ENCODING.encode(prefix)) if prefix else 0
            usable_chunk_size = self.chunk_size - prefix_tokens
            if usable_chunk_size <= self.overlap:
                raise ChunkingConfigException(
                    f"heading_path prefix is {prefix_tokens} tokens, leaving "
                    f"usable_max_tokens={usable_chunk_size} <= overlap={self.overlap} "
                    f"for chunk_size={self.chunk_size}. Increase chunk_size or "
                    "shorten heading_path."
                )
            step = usable_chunk_size - self.overlap

            tokens = _ENCODING.encode(clean_text)
            for start in range(0, len(tokens), step):
                token_slice = tokens[start : start + usable_chunk_size]
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=f"{prefix}{_ENCODING.decode(token_slice)}",
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
