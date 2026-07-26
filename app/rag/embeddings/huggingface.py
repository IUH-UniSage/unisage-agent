from dataclasses import dataclass


@dataclass(frozen=True)
class HuggingFaceEmbeddingConfig:
    """Configuration kept separate so the provider can be enabled later."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: int = 384


class HuggingFaceEmbedder:
    """Lazy adapter boundary for an optional local Hugging Face model."""

    def __init__(self, config: HuggingFaceEmbeddingConfig | None = None) -> None:
        self.config = config or HuggingFaceEmbeddingConfig()

    def embed(self, text: str) -> list[float]:
        """Fail clearly until the optional sentence-transformers package is installed."""

        if not text.strip():
            return [0.0] * self.config.dimensions
        raise RuntimeError(
            "HuggingFace embeddings are not enabled in the base scaffold. "
            "Install the provider dependency before using this adapter."
        )
