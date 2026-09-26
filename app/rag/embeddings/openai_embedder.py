from dataclasses import dataclass

from openai import OpenAI

from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.model_registry import require_top_priority_credential

# The OpenAI embeddings endpoint rejects an `input` array longer than 2048
# items with a 400. A single parsed region can produce more sentences than
# that, so requests are split into batches of this size and stitched back
# together in input order.
_MAX_INPUTS_PER_REQUEST = 2048


@dataclass(frozen=True)
class OpenAIEmbedder:
    """Embedding provider backed by the OpenAI-compatible embeddings API.

    Model name, API key and base URL come from the model registry's ACTIVE EMBEDDING
    credential (plan.md "Cutover khỏi cấu hình `.env` tĩnh") - never `.env`. Both are resolved
    lazily on first `embed()` call, not at construction time, so building an `OpenAIEmbedder()`
    never itself requires a loaded registry snapshot - only actually calling `embed` does
    (matches the previous lazy-client behavior; tests inject `model`/`client` directly to skip
    the registry entirely - see tests/test_embedding_provider.py, tests/test_retrieval.py).
    """

    model: str | None = None
    client: OpenAI | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts, preserving input order.

        Sent in as few requests as the endpoint's 2048-item cap allows
        (usually one).
        """

        if not texts:
            return []

        model = self.model
        client = self.client
        if model is None or client is None:
            credential = require_top_priority_credential("EMBEDDING")
            model = model or credential.model_name or ""
            client = client or OpenAI(
                api_key=credential.api_key,
                base_url=credential.api_base_url or None,
                http_client=build_provider_http_client_sync(
                    ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
                ),
            )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
