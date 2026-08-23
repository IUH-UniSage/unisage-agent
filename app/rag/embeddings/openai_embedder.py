from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import settings


@dataclass(frozen=True)
class OpenAIEmbedder:
    """Embedding provider backed by the OpenAI embeddings API, called in one batch."""

    model: str = field(default_factory=lambda: settings.OPENAI_EMBEDDING_MODEL)
    client: OpenAI | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in a single batched OpenAI call, preserving input order.

        The OpenAI client is built lazily on first use (not at construction
        time) so that constructing an `OpenAIEmbedder` never requires
        `OPENAI_API_KEY` to be set - only actually calling `embed` does.
        """

        if not texts:
            return []
        client = self.client or OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
