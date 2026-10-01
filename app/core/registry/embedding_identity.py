"""Embedding identity guard — plan.md "Embedding identity guard".

Refuses to embed (ingest OR query-time retrieval) with an ACTIVE EMBEDDING credential whose
identity doesn't match the vector space already established for the Qdrant collection. Two
different embedding models produce vectors that are not comparable even at the same dimension —
a silent swap corrupts retrieval instead of erroring, which is exactly what this guard exists to
prevent. There is no fallback path here: a mismatch always raises, and it's the same exception
type (`EmbeddingProviderError`, via the `EmbeddingIdentityMismatchError` subclass below) that
`app.worker.celery_app.embed_chunks` already lets escape the per-chunk try/except.

Wired into `app.rag.embeddings.openai_embedder.OpenAIEmbedder.embed()` — the single class both
ingest (`embed_chunks`) and query-time retrieval (`app.rag.retrieval.service.RetrievalService`)
build their embedder from, so both call paths get this guard for free without either module
having to call it directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.errors.provider_errors import EmbeddingProviderError
from app.core.llm.embedding_probe import (
    EmbeddingFingerprint,
    fingerprints_match,
    measure_fingerprint,
    unflatten_fingerprint,
)
from app.core.registry.model_registry import (
    CredentialConfig,
    EmbeddingIndexIdentity,
    ModelRegistrySnapshot,
)
from app.integrations.backend_java_client import BackendJavaClient, BackendJavaHTTPError
from app.rag.vectorstore import qdrant_store

logger = logging.getLogger(__name__)


class EmbeddingIdentityMismatchError(EmbeddingProviderError):
    """The ACTIVE EMBEDDING credential's identity doesn't match the collection's registered
    identity (or the collection already has vectors with no identity registered at all). Always
    terminal — never fall back, never use the credential anyway. This is exactly the "detect and
    stop" failure mode plan.md's "Embedding identity guard" exists for; an operator needs to
    resolve this (register the correct identity, or re-index) before ingest/retrieval can run
    again — alerting on this is a separate concern; this only guarantees the log signal exists.
    """


def identity_key(
    provider: str, model_name: str, source_ref: str | None, api_base_url: str | None, dimension: int
) -> str:
    """Hash of the identity actually used for one embedding write — attached to every Qdrant
    point's payload as `embedding_identity_key` so operators can later audit which vectors came
    from which identity."""

    raw = "|".join(
        str(part)
        for part in (provider, model_name, source_ref or "", api_base_url or "", dimension)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# One entry per (credential.id, credential.revision, snapshot.version) that has already passed
# the guard in this process — avoids re-probing the provider (an extra embeddings-API call) on
# every single `embed()` call. A hot-reloaded snapshot gets a new `version`, and a rotated
# credential gets a new `revision`, so either always forces a fresh check.
_verified: dict[tuple[str, int, int], str] = {}


def reset_verified_cache_for_tests() -> None:
    """Test-only escape hatch — the module-level cache above is process-global by design, but
    tests construct many snapshots/credentials with reused ids across cases."""

    _verified.clear()


def _run_async(coro):
    """Runs `coro` to completion from this deliberately-sync module (both
    `embed_chunks`/Celery and `OpenAIEmbedder.embed()`/the sync chunking
    pipeline call into this file synchronously).
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _measure(embed_probe: Callable[[list[str]], list[list[float]]]) -> EmbeddingFingerprint:
    try:
        return measure_fingerprint(embed_probe)
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        raise EmbeddingProviderError(f"embedding identity probe failed: {exc}") from exc


def ensure_embedding_identity(
    credential: CredentialConfig,
    *,
    snapshot: ModelRegistrySnapshot,
    embed_probe: Callable[[list[str]], list[list[float]]],
    qdrant_client: QdrantClient | None = None,
    backend_client: BackendJavaClient | None = None,
) -> str:
    """Verifies (or, for an empty collection with no identity yet, establishes) that
    `credential` matches the vector space currently in `settings.QDRANT_COLLECTION`.

    Returns the `embedding_identity_key` to attach to every point written with this credential.
    Raises `EmbeddingIdentityMismatchError` on any mismatch (never falls back), and
    `EmbeddingProviderError` if the probe call itself fails.
    """

    cache_key = (credential.id, credential.revision, snapshot.version)
    cached = _verified.get(cache_key)
    if cached is not None:
        return cached

    client = qdrant_client or qdrant_store.get_client()
    collection_dimension = qdrant_store.get_collection_dimension(client)
    has_points = qdrant_store.collection_has_points(client)
    identity = snapshot.embedding_index_identity
    bj_client = backend_client or BackendJavaClient()

    if identity is None:
        if has_points:
            logger.error(
                "embedding identity guard: collection '%s' already has vectors but no "
                "embedding identity is registered - refusing to embed. Run "
                "`python -m app.tools.register_embedding_index_identity` before enabling the "
                "registry for embedding (no automatic alert wired up yet).",
                settings.QDRANT_COLLECTION,
            )
            raise EmbeddingIdentityMismatchError(
                f"collection '{settings.QDRANT_COLLECTION}' already has vectors but no embedding "
                "identity is registered yet",
                credential=credential,
            )
        key = _bootstrap_identity(credential, embed_probe, bj_client)
        _verified[cache_key] = key
        return key

    if credential.provider != identity.provider or credential.model_name != identity.model_name:
        logger.error(
            "embedding identity guard: ACTIVE EMBEDDING credential (provider=%s, model=%s) does "
            "not match the collection's registered identity (provider=%s, model=%s) - refusing "
            "to embed (no automatic alert wired up yet).",
            credential.provider,
            credential.model_name,
            identity.provider,
            identity.model_name,
        )
        raise EmbeddingIdentityMismatchError(
            "ACTIVE EMBEDDING credential does not match the collection's registered identity",
            credential=credential,
        )

    if (
        identity.dimension is not None
        and collection_dimension is not None
        and identity.dimension != collection_dimension
    ):
        logger.error(
            "embedding identity guard: registered identity dimension=%s does not match "
            "collection '%s''s actual configured dimension=%s (no automatic alert wired up yet).",
            identity.dimension,
            settings.QDRANT_COLLECTION,
            collection_dimension,
        )
        raise EmbeddingIdentityMismatchError(
            "registered embedding identity dimension does not match the collection's actual "
            "vector dimension",
            credential=credential,
        )

    dimension = identity.dimension or collection_dimension or 0
    if identity.fingerprint is not None:
        measured = _measure(embed_probe)
        if measured.dimension != identity.dimension or not fingerprints_match(
            measured, identity.fingerprint
        ):
            logger.error(
                "embedding identity guard: measured fingerprint for the ACTIVE EMBEDDING "
                "credential does not match collection '%s''s registered fingerprint - the "
                "provider may have silently swapped models behind the same name/dimension "
                "(no automatic alert wired up yet).",
                settings.QDRANT_COLLECTION,
            )
            raise EmbeddingIdentityMismatchError(
                "measured embedding fingerprint does not match the collection's registered "
                "identity",
                credential=credential,
            )
        dimension = measured.dimension

    key = identity_key(
        credential.provider or "",
        credential.model_name or "",
        identity.model_source_ref,
        credential.api_base_url,
        dimension,
    )
    _verified[cache_key] = key
    return key


def _bootstrap_identity(
    credential: CredentialConfig,
    embed_probe: Callable[[list[str]], list[list[float]]],
    bj_client: BackendJavaClient,
) -> str:
    """Collection is empty and no identity is registered yet — the first ingest batch of the
    currently-running job establishes it itself (plan.md "Khởi tạo danh tính index"): measure
    the fingerprint from the credential actually being used, then PUT it.
    201 -> proceed. 409 -> someone else won the race (another worker, possibly on a different
    snapshot); GET and compare - match -> proceed, mismatch -> job FAILED, nothing upserted.
    """

    fingerprint = _measure(embed_probe)
    try:
        _run_async(
            bj_client.put_embedding_index_identity(
                collection=settings.QDRANT_COLLECTION,
                provider=credential.provider or "",
                model_name=credential.model_name or "",
                model_source_ref=None,
                api_base_url=credential.api_base_url,
                dimension=fingerprint.dimension,
                fingerprint=fingerprint.flattened(),
                established_by="first-upsert",
            )
        )
        return identity_key(
            credential.provider or "",
            credential.model_name or "",
            None,
            credential.api_base_url,
            fingerprint.dimension,
        )
    except BackendJavaHTTPError as exc:
        if exc.status_code != 409:
            raise EmbeddingProviderError(
                f"failed to register embedding identity: {exc}", credential=credential
            ) from exc
        return _reconcile_after_lost_race(credential, fingerprint, bj_client, exc)


def _reconcile_after_lost_race(
    credential: CredentialConfig,
    fingerprint: EmbeddingFingerprint,
    bj_client: BackendJavaClient,
    cause: Exception,
) -> str:
    existing = _run_async(
        bj_client.get_embedding_index_identity(collection=settings.QDRANT_COLLECTION)
    )
    if existing is None:
        raise EmbeddingProviderError(
            "embedding identity PUT returned 409 but a follow-up GET found none",
            credential=credential,
        ) from cause

    registered_dimension = existing.get("dimension")
    registered_fingerprint = (
        unflatten_fingerprint(existing["fingerprint"], int(registered_dimension))
        if existing.get("fingerprint") and registered_dimension is not None
        else None
    )
    registered = EmbeddingIndexIdentity(
        provider=existing.get("provider"),
        model_name=existing.get("modelName"),
        model_source_ref=existing.get("modelSourceRef"),
        api_base_url=existing.get("apiBaseUrl"),
        dimension=registered_dimension,
        fingerprint=registered_fingerprint,
    )

    if registered.provider != credential.provider or registered.model_name != credential.model_name:
        logger.error(
            "embedding identity guard: lost the race to establish collection '%s''s embedding "
            "identity, and the winner used a different provider/model (provider=%s, model=%s) "
            "than this credential (provider=%s, model=%s).",
            settings.QDRANT_COLLECTION,
            registered.provider,
            registered.model_name,
            credential.provider,
            credential.model_name,
        )
        raise EmbeddingIdentityMismatchError(
            "lost the race to establish the collection's embedding identity, and the winner "
            "used a different provider/model",
            credential=credential,
        ) from cause

    if registered.fingerprint is not None and not fingerprints_match(
        fingerprint, registered.fingerprint
    ):
        logger.error(
            "embedding identity guard: lost the race to establish collection '%s''s embedding "
            "identity, and the winner's fingerprint does not match this credential's.",
            settings.QDRANT_COLLECTION,
        )
        raise EmbeddingIdentityMismatchError(
            "lost the race to establish the collection's embedding identity, and the winner's "
            "fingerprint does not match this credential",
            credential=credential,
        ) from cause

    return identity_key(
        credential.provider or "",
        credential.model_name or "",
        registered.model_source_ref,
        credential.api_base_url,
        registered.dimension or fingerprint.dimension,
    )
