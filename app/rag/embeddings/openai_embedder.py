import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.core.embedding_identity import ensure_embedding_identity
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.llm_error_classifier import EmbeddingProviderError
from app.core.model_registry import (
    CredentialConfig,
    ModelRegistryError,
    active_credentials_for,
    get_current_snapshot,
    require_top_priority_credential,
)
from app.core.redaction import safe_error_message
from app.core.usage_recorder import UsageRecorder

logger = logging.getLogger(__name__)

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
    the registry entirely, and skip the identity guard below with it - see
    tests/test_embedding_provider.py, tests/test_retrieval.py).

    When `model`/`client` are resolved from the registry (not injected), every `embed()` call is
    gated by `app.core.embedding_identity.ensure_embedding_identity` (plan.md "Embedding identity
    guard") - this is the one chokepoint both ingest (`app.worker.celery_app.embed_chunks`) and
    query-time retrieval (`app.rag.retrieval.service.RetrievalService`, which builds its embedder
    from this same class) go through, so neither path can silently embed with a credential that
    doesn't match the Qdrant collection's established vector space. A failed provider call, a
    missing credential, or a failed guard check all raise `EmbeddingProviderError` - embedding
    never auto-fails-over, so there is nothing to catch and retry here; callers must let it
    escape.
    """

    model: str | None = None
    client: OpenAI | None = None
    # Mutable cache for the lazily-resolved model/client/credential/identity_key - a dict's
    # *contents* can be mutated on a frozen dataclass even though the field itself can't be
    # reassigned. Keeps credential resolution + the identity guard check to once per instance
    # instead of once per `embed()` call (this class is constructed once per Celery task
    # invocation / once per `RetrievalService`, but `embed()` used to be called once per chunk).
    _resolved: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def identity_key(self) -> str | None:
        """The `embedding_identity_key` established/verified for whichever credential this
        instance resolved from the registry - `None` if `model`/`client` were injected directly
        (guard skipped) or `embed()` hasn't been called yet."""

        return self._resolved.get("identity_key")

    @property
    def credential(self) -> CredentialConfig | None:
        """The registry credential this instance resolved, if any - lets a caller several
        frames away (`app.worker.celery_app.embed_chunks`) build a `report_health` call after an
        `EmbeddingProviderError` without having to re-derive which credential failed."""

        return self._resolved.get("credential")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts, preserving input order.

        Sent in as few requests as the endpoint's 2048-item cap allows
        (usually one).

        Cost Tracking plan.md Task 8: one `embed()` call is one business request
        (`purpose=EMBEDDING`, no conversation/message ids - plan.md "1 request = 1
        batch embed"). Self-contained: builds and closes its own `UsageRecorder`,
        so `app.worker.celery_app.embed_chunks` (the one production caller) needs
        no changes. Only recorded when a registry credential was actually resolved
        - a test that injects `model`/`client` directly (skipping the registry
        entirely) has no credential to snapshot and records nothing, same as the
        identity-probe call in `_resolve_from_registry` below, which never passes
        a recorder at all.
        """

        if not texts:
            return []

        model = self.model
        client = self.client
        if model is None or client is None:
            if "model" not in self._resolved:
                self._resolve_from_registry()
            model = self._resolved["model"]
            client = self._resolved["client"]

        credential = self._resolved.get("credential")
        if credential is None:
            return self._call_provider(client, model, texts)

        recorder = UsageRecorder(request_id=str(uuid.uuid4()), purpose="EMBEDDING")
        status = "ERROR"
        try:
            vectors = self._call_provider(client, model, texts, usage_recorder=recorder)
            status = "SUCCESS"
            return vectors
        finally:
            asyncio.run(recorder.close(status=status))

    def _resolve_from_registry(self) -> None:
        embedding_credentials = active_credentials_for("EMBEDDING")
        if len(embedding_credentials) > 1:
            # Shouldn't happen - Java enforces a unique partial index on
            # `(model_purpose='EMBEDDING', status='ACTIVE')` - but embedding is the one purpose
            # that must never silently pick "a" credential if this invariant is ever violated.
            logger.error(
                "model registry snapshot has %d ACTIVE EMBEDDING credentials (expected at most "
                "1) - using the highest-priority one defensively",
                len(embedding_credentials),
            )

        try:
            credential = require_top_priority_credential("EMBEDDING")
        except ModelRegistryError as exc:
            raise EmbeddingProviderError(str(exc)) from exc

        snapshot = get_current_snapshot()
        if snapshot is None:
            raise EmbeddingProviderError(
                "no model registry snapshot loaded", credential=credential
            )

        model = credential.model_name or ""
        client = OpenAI(
            api_key=credential.api_key,
            base_url=credential.api_base_url or None,
            http_client=build_provider_http_client_sync(
                ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
            ),
        )

        def _embed_probe(probe_texts: list[str]) -> list[list[float]]:
            return self._call_provider(client, model, probe_texts)

        identity_key = ensure_embedding_identity(
            credential, snapshot=snapshot, embed_probe=_embed_probe
        )

        self._resolved["credential"] = credential
        self._resolved["model"] = model
        self._resolved["client"] = client
        self._resolved["identity_key"] = identity_key

    def _call_provider(
        self,
        client: OpenAI,
        model: str,
        texts: list[str],
        *,
        usage_recorder: UsageRecorder | None = None,
    ) -> list[list[float]]:
        """`usage_recorder`, when given, gets exactly one line for this call (summed
        across every 2048-item sub-batch it takes to embed `texts`) - `None` here
        (the identity-probe call in `_resolve_from_registry`, and any test that
        constructs this class with `model`/`client` injected directly) means no
        line at all, not a line with zero tokens."""

        credential = self._resolved.get("credential")
        started_at = time.monotonic()
        try:
            vectors: list[list[float]] = []
            prompt_tokens = 0
            for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
                batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
                response = client.embeddings.create(model=model, input=batch)
                vectors.extend(item.embedding for item in response.data)
                if response.usage is not None:
                    prompt_tokens += response.usage.prompt_tokens
            if usage_recorder is not None:
                usage_recorder.record(
                    node_name="embed_batch",
                    attempt=0,
                    credential=credential,
                    status="SUCCESS",
                    input_tokens=prompt_tokens,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                )
            return vectors
        except EmbeddingProviderError:
            if usage_recorder is not None:
                usage_recorder.record(
                    node_name="embed_batch",
                    attempt=0,
                    credential=credential,
                    status="ERROR",
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                    error_code="EmbeddingProviderError",
                )
            raise
        except Exception as exc:
            if usage_recorder is not None:
                usage_recorder.record(
                    node_name="embed_batch",
                    attempt=0,
                    credential=credential,
                    status="ERROR",
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                    error_code=type(exc).__name__,
                )
            message = safe_error_message(exc, credential.api_key if credential else None)
            raise EmbeddingProviderError(message, credential=credential) from exc
