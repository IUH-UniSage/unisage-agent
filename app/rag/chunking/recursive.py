from dataclasses import dataclass

from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk


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
        """Split each region into overlapping chunks without dropping trailing content."""

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for region in regions:
            clean_text = region.content.strip()
            if not clean_text:
                continue
            for start in range(0, len(clean_text), step):
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=clean_text[start : start + self.chunk_size],
                        region_type=region.region_type,
                    )
                )
        return chunks
