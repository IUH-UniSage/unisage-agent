"""Exceptions raised around AI-provider calls that callers several frames away must
recognise (the ingest task, the API error handlers, the circuit breaker's classifier).

Classification of a failure lives in `llm_error_classifier` (TRANSIENT/PERMANENT, for the
circuit breaker) and `llm_failure` (the cause shown to the client); this module only
defines the exception types.
"""

from __future__ import annotations

from typing import Any


class EmbeddingProviderError(Exception):
    """Raised when the ACTIVE EMBEDDING credential itself is unusable — the actual provider call
    failed (auth/connection/rate-limit/...), there is no ACTIVE EMBEDDING credential at all, or
    the embedding identity guard (`app.core.registry.embedding_identity`) refused to use
    it. Embedding never auto-fails-over — there is no other credential to
    route to, so this is always terminal for the job. Every caller
    (`OpenAIEmbedder.embed`, and transitively `app.worker.tasks.ingestion.embed_chunks` and
    `app.rag.retrieval.service.RetrievalService`) must let this escape uncaught rather than
    treat it as a per-chunk data problem.

    `credential` (when known) is attached so a caller several frames away (`embed_chunks`) can
    still build a `report_health` call without having to re-derive which credential failed.
    """

    def __init__(self, message: str, *, credential: Any = None) -> None:
        super().__init__(message)
        self.credential = credential


class EmbeddingBudgetRejectedError(EmbeddingProviderError):
    """An embedding call was refused by budget enforcement (request-level reservation or the
    PROVIDER-scope acquire) before any provider call was made. A subclass so every existing
    "abort the whole job" handling of `EmbeddingProviderError` still applies, but distinct so
    the client can be told it is a budget limit, not a broken credential. `reason` is the
    tracker's result string (e.g. "REJECT_EXCEEDED", "DENY_THROTTLED")."""

    def __init__(self, message: str, *, reason: str, credential: Any = None) -> None:
        super().__init__(message, credential=credential)
        self.reason = reason


class MalformedExtractionResponseError(Exception):
    """Raised by a caller (never by a provider SDK itself) when a credential's response parsed
    fine at the transport level but didn't match the shape the caller actually needed - e.g. a
    fallback EXTRACTION credential's JSON missing the expected keys (see
    `app/rag/enrichment/multi_representation.py`). Always PERMANENT: a credential that returns
    the wrong shape isn't a transient blip, it needs SA attention rather than an automatic retry
    against the exact same credential."""
