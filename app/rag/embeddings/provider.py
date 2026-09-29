from typing import TYPE_CHECKING, Protocol

from app.core.budget.tracker import BudgetTracker
from app.core.registry.model_registry import get_current_snapshot

if TYPE_CHECKING:
    from app.core.usage.usage_recorder import UsageRecorder


class EmbeddingProvider(Protocol):
    """Swap point for the embedding backend - `OpenAIEmbedder` (OpenAI, self-hosted OpenAI-
    compatible) or `GoogleEmbedder` (Gemini's native embedContent API), picked by
    `build_embedder()` below from the ACTIVE EMBEDDING credential's `provider`."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_tracked(
        self,
        texts: list[str],
        usage_recorder: "UsageRecorder",
        budget_tracker: BudgetTracker,
    ) -> list[list[float]]: ...


def build_embedder() -> EmbeddingProvider:
    """Picks `OpenAIEmbedder` or `GoogleEmbedder` for the current ACTIVE EMBEDDING credential's
    `provider` - a Google credential speaks Gemini's native `embedContent` API, everything else
    (OpenAI, a `SELF_HOSTED` OpenAI-compatible server) speaks the OpenAI wire format.

    Only peeks at the snapshot's `provider` string here; the returned instance still resolves
    the full credential (api key, model, identity guard) itself, lazily, on first `embed()`/
    `embed_tracked()` call - same as constructing either class directly. If no snapshot/
    credential is loaded yet, defaults to `OpenAIEmbedder`, which raises its own
    `EmbeddingProviderError` on first use - matching the error a caller would see today if it
    just constructed `OpenAIEmbedder()` directly.

    Local imports (not top-level) - `openai_embedder.py`/`google_embedder.py` don't need to
    import each other, and this module stays the one place a caller picks between them.
    """

    from app.rag.embeddings.google_embedder import GoogleEmbedder
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    snapshot = get_current_snapshot()
    # Java enforces a unique partial index on (model_purpose='EMBEDDING', status='ACTIVE') -
    # at most one candidate ever exists; `OpenAIEmbedder`/`GoogleEmbedder`'s own
    # `_resolve_from_registry` still defensively logs if that's ever violated.
    candidates = snapshot.credentials_for("EMBEDDING") if snapshot is not None else ()
    provider = candidates[0].provider if candidates else None

    if provider == "google":
        return GoogleEmbedder()
    return OpenAIEmbedder()
