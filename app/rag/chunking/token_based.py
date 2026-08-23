from dataclasses import dataclass

import tiktoken

from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")


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
        """Split each region into overlapping chunks bounded by token count."""

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for region in regions:
            clean_text = region.content.strip()
            if not clean_text:
                continue
            tokens = _ENCODING.encode(clean_text)
            for start in range(0, len(tokens), step):
                token_slice = tokens[start : start + self.chunk_size]
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=_ENCODING.decode(token_slice),
                        region_type=region.region_type,
                    )
                )
        return chunks
