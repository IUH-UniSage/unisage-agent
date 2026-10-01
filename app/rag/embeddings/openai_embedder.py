import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from openai import OpenAI
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker
from app.core.config import settings
from app.core.errors.provider_errors import EmbeddingBudgetRejectedError, EmbeddingProviderError
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.registry.embedding_identity import ensure_embedding_identity
from app.core.registry.model_registry import (
    CredentialConfig,
    ModelRegistryError,
    active_credentials_for,
    get_current_snapshot,
    require_top_priority_credential,
)
from app.core.security.redaction import safe_error_message
from app.core.usage.usage_recorder import UsageRecorder

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
    credential - never `.env`. Both are resolved
    lazily on first `embed()` call, not at construction time, so building an `OpenAIEmbedder()`
    never itself requires a loaded registry snapshot - only actually calling `embed` does
    (matches the previous lazy-client behavior; tests inject `model`/`client` directly to skip
    the registry entirely, and skip the identity guard below with it - see
    tests/test_embedding_provider.py, tests/test_retrieval.py).

    When `model`/`client` are resolved from the registry (not injected), every `embed()` call is
    gated by `app.core.registry.embedding_identity.ensure_embedding_identity` - this is the
    one chokepoint both ingest (`app.worker.celery_app.embed_chunks`) and
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

        One `embed()` call is one business request (`purpose=EMBEDDING`, no
        conversation/message ids). Self-contained: builds and closes its own `UsageRecorder`,
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

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._embed_recorded(client, model, texts, credential))
        # Chat retrieval calls this sync method from inside the server's event loop, where
        # asyncio.run() is not allowed.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, self._embed_recorded(client, model, texts, credential)
            ).result()

    async def embed_tracked(
        self,
        texts: list[str],
        usage_recorder: UsageRecorder,
        budget_tracker: BudgetTracker,
    ) -> list[list[float]]:
        """Embed within a CALLER-managed `UsageRecorder`/`BudgetTracker` - the ingestion
        pipeline's shared per-document recorder (`app.worker.celery_app.embed_chunks`), which
        reserves/settles the request-level budget once for the whole document rather than once
        per `embed()` call. Unlike `embed()`, never builds/closes its own recorder or reserves/
        settles a request-level budget - only the one PROVIDER-scope acquire/release pair
        `_call_provider_tracked` already does per call. Must be awaited from within the caller's
        own event loop (no `asyncio.run`/thread-pool wrapping here, unlike `embed()`)."""

        if not texts:
            return []

        model = self.model
        client = self.client
        if model is None or client is None:
            if "model" not in self._resolved:
                self._resolve_from_registry()
            model = self._resolved["model"]
            client = self._resolved["client"]

        return await self._call_provider_tracked(
            client, model, texts, usage_recorder, budget_tracker
        )

    async def _embed_recorded(
        self, client: OpenAI, model: str, texts: list[str], credential: CredentialConfig
    ) -> list[list[float]]:
        # One event loop and one Redis client for the whole call: an asyncio Redis connection
        # is bound to the loop that opened it, so reusing a shared client across separate
        # asyncio.run() calls fails and the budget calls silently fail open.
        redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
        budget_tracker = BudgetTracker(redis_client=redis_client)
        recorder = UsageRecorder(
            request_id=str(uuid.uuid4()), purpose="EMBEDDING", budget_tracker=budget_tracker
        )
        status = "ERROR"
        try:
            reserve_result = await budget_tracker.reserve_request(
                request_id=recorder.request_id,
                purpose="EMBEDDING",
                estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)),
            )
            if reserve_result != "OK":
                # Reuses `EmbeddingProviderError` (not `RequestBudgetRejectedError`) so
                # `app.worker.celery_app.embed_chunks`'s existing "abort the whole job"
                # handling for that type applies here too - a budget rejection is not a
                # per-chunk data-quality problem, it means embedding should stop entirely.
                raise EmbeddingBudgetRejectedError(
                    f"EMBEDDING budget reservation rejected: {reserve_result}",
                    reason=reserve_result,
                    credential=credential,
                )
            vectors = await self._call_provider_tracked(
                client, model, texts, recorder, budget_tracker
            )
            status = "SUCCESS"
            return vectors
        finally:
            try:
                await recorder.close(status=status)
            finally:
                try:
                    await redis_client.aclose()
                except Exception:
                    logger.debug("embedder: closing the Redis client failed", exc_info=True)

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
            raise EmbeddingProviderError("no model registry snapshot loaded", credential=credential)

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

    def _call_provider(self, client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
        """Untracked call - the identity probe, and any caller that injected `model`/`client`
        directly (no registry credential, so nothing to record or budget)."""

        credential = self._resolved.get("credential")
        try:
            vectors, _ = self._request_embeddings(client, model, texts)
            return vectors
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            message = safe_error_message(exc, credential.api_key if credential else None)
            raise EmbeddingProviderError(message, credential=credential) from exc

    async def _call_provider_tracked(
        self,
        client: OpenAI,
        model: str,
        texts: list[str],
        usage_recorder: UsageRecorder,
        budget_tracker: BudgetTracker,
    ) -> list[list[float]]:
        """Records exactly one line for this call (summed across every 2048-item sub-batch)
        and gates it against the PROVIDER-scope budget - embedding never fails over, so there
        is a single acquire/release pair, no candidate loop."""

        credential = self._resolved.get("credential")
        budget_seq = usage_recorder.reserve_budget_seq()
        acquire_result = await budget_tracker.acquire_provider(
            request_id=usage_recorder.request_id,
            seq=budget_seq,
            provider=credential.provider if credential else "",
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)),
        )
        if acquire_result != "OK":
            raise EmbeddingBudgetRejectedError(
                f"EMBEDDING provider budget denied: {acquire_result}",
                reason=acquire_result,
                credential=credential,
            )

        started_at = time.monotonic()
        committed_usd = Decimal("0")
        try:
            # The OpenAI client is sync; keep it off the event loop.
            vectors, prompt_tokens = await asyncio.to_thread(
                self._request_embeddings, client, model, texts
            )
            committed_usd = usage_recorder.record(
                node_name="embed_batch",
                attempt=0,
                credential=credential,
                status="SUCCESS",
                input_tokens=prompt_tokens,
                latency_ms=int((time.monotonic() - started_at) * 1000),
            )
            return vectors
        except Exception as exc:
            committed_usd = usage_recorder.record(
                node_name="embed_batch",
                attempt=0,
                credential=credential,
                status="ERROR",
                latency_ms=int((time.monotonic() - started_at) * 1000),
                error_code=type(exc).__name__,
            )
            if isinstance(exc, EmbeddingProviderError):
                raise
            message = safe_error_message(exc, credential.api_key if credential else None)
            raise EmbeddingProviderError(message, credential=credential) from exc
        finally:
            await budget_tracker.release_provider(
                request_id=usage_recorder.request_id,
                seq=budget_seq,
                actual_usd=committed_usd,
            )

    @staticmethod
    def _request_embeddings(
        client: OpenAI, model: str, texts: list[str]
    ) -> tuple[list[list[float]], int]:
        vectors: list[list[float]] = []
        prompt_tokens = 0
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            if response.usage is not None:
                prompt_tokens += response.usage.prompt_tokens
        return vectors, prompt_tokens
