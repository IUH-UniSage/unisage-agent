from typing import Protocol


class EmbeddingProvider(Protocol):
    """Swap point for the embedding backend (OpenAI today, self-hosted later)."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...
