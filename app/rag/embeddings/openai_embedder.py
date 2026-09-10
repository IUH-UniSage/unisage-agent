from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import settings

# The OpenAI embeddings endpoint rejects an `input` array longer than 2048
# items with a 400. A single parsed region can produce more sentences than
# that, so requests are split into batches of this size and stitched back
# together in input order.
_MAX_INPUTS_PER_REQUEST = 2048


@dataclass(frozen=True)
class OpenAIEmbedder:
    """Embedding provider backed by the OpenAI embeddings API."""

    model: str = field(default_factory=lambda: settings.OPENAI_EMBEDDING_MODEL)
    client: OpenAI | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts, preserving input order.

        Sent in as few requests as the endpoint's 2048-item cap allows
        (usually one). The OpenAI client is built lazily on first use (not
        at construction time) so that constructing an `OpenAIEmbedder`
        never requires `OPENAI_API_KEY` to be set - only actually calling
        `embed` does.
        """

        if not texts:
            return []
        client = self.client or OpenAI(api_key=settings.OPENAI_API_KEY)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            response = client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
