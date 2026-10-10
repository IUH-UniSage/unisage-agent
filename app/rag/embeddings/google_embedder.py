import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import tiktoken
from google.genai import Client
from google.genai.types import HttpOptions
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker
from app.core.config import settings
from app.core.errors.provider_errors import EmbeddingBudgetRejectedError, EmbeddingProviderError
from app.core.llm.http_client import (
    ProviderConnectionInfo,
    build_provider_http_client,
    build_provider_http_client_sync,
)
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

# Gemini's embedContent response carries no token-usage field at all (unlike generateContent's
# usageMetadata, which pydantic-ai's GoogleModel reads for CHAT/EXTRACTION) - there is nothing
# real to report for PRICED-cost accounting, so input tokens are approximated the same way
# `app.rag.chunking.semantic` already estimates tokens for chunk-sizing. Not exact, but close
# enough for cost history, and consistent with the rest of the codebase's own token estimates.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _estimate_input_tokens(texts: list[str]) -> int:
    return sum(len(_ENCODING.encode(text)) for text in texts)


def request_embeddings_sync(
    client: Client, model: str, texts: list[str]
) -> tuple[list[list[float]], int]:
    """The actual `embedContent` call(s) + response extraction - a public module-level function
    (not just a method on `GoogleEmbedder`) so `app.worker.verification_tasks` can fingerprint a
    CANDIDATE credential (not yet in the registry snapshot, so there's no `GoogleEmbedder`
    instance to resolve one for) with the exact same request/response handling, not a second
    hand-rolled copy of it.

    One `embed_content` call PER TEXT, never a `contents=[text1, text2, ...]` batch in one call:
    the SDK's own `embed_content` special-cases any model name containing `gemini-embedding-2`
    by running the whole list through `_transformers.t_contents` (the generic
    multi-turn-conversation transformer, which merges every item into ONE `Content`'s parts) -
    not `t_contents_for_embed` (the one that actually keeps a list of independent items
    independent). The result: a 3-text batch call silently returns exactly 1 combined
    embedding instead of 3 - confirmed empirically (`ValueError: embedding probe expected 3
    vectors, provider returned 1`), not just a docs read. One call per text sidesteps that
    entirely and is correct for every Gemini embedding model, not just the ones that happen to
    batch correctly.
    """

    vectors: list[list[float]] = []
    for text in texts:
        response = client.models.embed_content(model=model, contents=text)
        vectors.extend(embedding.values or [] for embedding in (response.embeddings or []))
    return vectors, _estimate_input_tokens(texts)


async def request_embeddings_async(
    client: Client, model: str, texts: list[str]
) -> tuple[list[list[float]], int]:
    """Async counterpart of `request_embeddings_sync` - same one-call-per-text reasoning (see
    that function's docstring). Issued concurrently (`asyncio.gather`), not sequentially - unlike
    the sync path (verification's one-shot fingerprint probe, 3 texts, latency doesn't matter),
    this is the production ingest path (`GoogleEmbedder.embed_tracked`), where a chunk's
    content/summary/questions texts embedding one-at-a-time would triple real latency for no
    reason now that they can't be combined into one provider call."""

    async def _embed_one(text: str) -> list[list[float]]:
        response = await client.aio.models.embed_content(model=model, contents=text)
        return [embedding.values or [] for embedding in (response.embeddings or [])]

    results = await asyncio.gather(*(_embed_one(text) for text in texts))
    vectors = [vector for result in results for vector in result]
    return vectors, _estimate_input_tokens(texts)


@dataclass(frozen=True)
class GoogleEmbedder:
    """Embedding provider backed by Google's native `embedContent` API (`google-genai`), for an
    EMBEDDING credential with `provider="google"`.

    `app.rag.embeddings.openai_embedder.OpenAIEmbedder` always speaks the OpenAI wire format
    (`POST {api_base_url}/embeddings`) regardless of `credential.provider` - against Gemini's
    native REST API that 404s (embeddings live at a completely different path/shape,
    `models/{model}:embedContent`, not the OpenAI-compatible `/embeddings` OpenAI's SDK calls).
    This class is the Google-native counterpart, selected by
    `app.rag.embeddings.provider.build_embedder()` purely from the ACTIVE EMBEDDING credential's
    `provider` field - same public contract as `OpenAIEmbedder` (`embed()`/`embed_tracked()`/
    `identity_key`/`credential`), so every caller (`app.worker.tasks.ingestion.embed_chunks`,
    `app.rag.chunking.semantic.SemanticChunker`, `app.rag.retrieval.service.RetrievalService`)
    can use either interchangeably without knowing which one it got.

    Model name, API key and base URL come from the model registry's ACTIVE EMBEDDING credential
    - never `.env`. Resolved lazily on first `embed()`/`embed_tracked()` call, not at construction
    time, matching `OpenAIEmbedder`'s identical precedent; tests inject `model`/`client` directly
    to skip the registry entirely.

    When resolved from the registry, every call is gated by
    `app.core.registry.embedding_identity.ensure_embedding_identity` - the same chokepoint
    `OpenAIEmbedder` goes through, so a Google embedding credential can't silently embed into a
    Qdrant collection established by a different credential's vector space either.
    """

    model: str | None = None
    client: Client | None = None
    # Mutable cache for the lazily-resolved model/client/credential/identity_key - see
    # OpenAIEmbedder's identical field for why a dict's contents (not the field itself) is what
    # gets mutated on this frozen dataclass.
    _resolved: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def identity_key(self) -> str | None:
        return self._resolved.get("identity_key")

    @property
    def credential(self) -> CredentialConfig | None:
        return self._resolved.get("credential")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts, preserving input order - self-contained, builds and closes its own
        `UsageRecorder` (`purpose=EMBEDDING`). See `OpenAIEmbedder.embed()` for the full
        contract this mirrors."""

        if not texts:
            return []

        model, client = self._ensure_resolved()
        credential = self._resolved.get("credential")
        if credential is None:
            return self._call_provider(client, model, texts)

        return _run_sync(self._embed_recorded(client, model, texts, credential))

    async def embed_tracked(
        self,
        texts: list[str],
        usage_recorder: UsageRecorder,
        budget_tracker: BudgetTracker,
    ) -> list[list[float]]:
        """Embed within a CALLER-managed `UsageRecorder`/`BudgetTracker` - see
        `OpenAIEmbedder.embed_tracked()` for the full contract this mirrors."""

        if not texts:
            return []

        model, client = self._ensure_resolved()
        return await self._call_provider_tracked(
            client, model, texts, usage_recorder, budget_tracker
        )

    def _ensure_resolved(self) -> tuple[str, Client]:
        model = self.model
        client = self.client
        if model is None or client is None:
            if "model" not in self._resolved:
                self._resolve_from_registry()
            model = self._resolved["model"]
            client = self._resolved["client"]
        return model, client

    async def _embed_recorded(
        self, client: Client, model: str, texts: list[str], credential: CredentialConfig
    ) -> list[list[float]]:
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
                    logger.debug("GoogleEmbedder: closing the Redis client failed", exc_info=True)

    def _resolve_from_registry(self) -> None:
        embedding_credentials = active_credentials_for("EMBEDDING")
        if len(embedding_credentials) > 1:
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
        connection_info = ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
        client = Client(
            vertexai=False,
            api_key=credential.api_key,
            http_options=HttpOptions(
                base_url=credential.api_base_url or None,
                httpx_client=build_provider_http_client_sync(connection_info),
                httpx_async_client=build_provider_http_client(connection_info),
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

    def _call_provider(self, client: Client, model: str, texts: list[str]) -> list[list[float]]:
        """Untracked call - the identity probe, and any caller that injected `model`/`client`
        directly (no registry credential, so nothing to record or budget)."""

        credential = self._resolved.get("credential")
        try:
            vectors, _ = self._request_embeddings_sync(client, model, texts)
            return vectors
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            message = safe_error_message(exc, credential.api_key if credential else None)
            raise EmbeddingProviderError(message, credential=credential) from exc

    async def _call_provider_tracked(
        self,
        client: Client,
        model: str,
        texts: list[str],
        usage_recorder: UsageRecorder,
        budget_tracker: BudgetTracker,
    ) -> list[list[float]]:
        """Records exactly one line for this call and gates it against the PROVIDER-scope
        budget - see `OpenAIEmbedder._call_provider_tracked()`'s identical contract."""

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
            # Unlike OpenAIEmbedder (one client, sync-only SDK - needs asyncio.to_thread to stay
            # off the event loop), this client was built with BOTH a sync and an async SSRF-pinned
            # transport, so the native async path is used directly, no thread hop needed.
            vectors, input_tokens = await self._request_embeddings_async(client, model, texts)
            committed_usd = usage_recorder.record(
                node_name="embed_batch",
                attempt=0,
                credential=credential,
                status="SUCCESS",
                input_tokens=input_tokens,
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
    def _request_embeddings_sync(
        client: Client, model: str, texts: list[str]
    ) -> tuple[list[list[float]], int]:
        return request_embeddings_sync(client, model, texts)

    @staticmethod
    async def _request_embeddings_async(
        client: Client, model: str, texts: list[str]
    ) -> tuple[list[list[float]], int]:
        return await request_embeddings_async(client, model, texts)


def _run_sync(coro: Any) -> Any:
    """Bridges an async call from a sync call site - see `OpenAIEmbedder.embed()`'s identical
    "already inside a running loop" handling (Chat retrieval calls `embed()` from inside the
    server's event loop, where `asyncio.run()` is not allowed)."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
