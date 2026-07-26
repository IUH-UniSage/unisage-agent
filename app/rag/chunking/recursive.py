from dataclasses import dataclass


@dataclass(frozen=True)
class RecursiveChunker:
    """Small deterministic character chunker for the initial ingestion flow."""

    chunk_size: int = 800
    overlap: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be between zero and chunk_size - 1")

    def split(self, text: str) -> list[str]:
        """Split text into overlapping chunks without dropping trailing content."""

        clean_text = text.strip()
        if not clean_text:
            return []

        step = self.chunk_size - self.overlap
        return [
            clean_text[start : start + self.chunk_size] for start in range(0, len(clean_text), step)
        ]
