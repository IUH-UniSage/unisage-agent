"""The retrieval cache's view of document changes (docs/specs/SPEC-retrieval-cache.md,
"Hook invalidation"). Every hook is a no-op when the cache is off and fails open: a
Redis error here must never fail the ingest, edit or delete that triggered it."""

from __future__ import annotations

import logging

from app.integrations.retrieval_cache import PUBLIC_SCOPE, default_cache, department_scope

logger = logging.getLogger(__name__)


async def invalidate_document(document_id: str) -> int:
    """The document was edited, deleted, re-chunked, or its access payload changed: every
    cached turn that referenced one of its chunks goes. Returns the entries dropped."""

    cache = default_cache()
    if cache is None:
        return 0
    try:
        return await cache.purge_document(document_id)
    except Exception:
        logger.warning("retrieval cache: purge for %s failed", document_id, exc_info=True)
        return 0


async def invalidate_after_embed(document_id: str, *, department_id: str, is_public: bool) -> None:
    """New or re-embedded content: drop what referenced the document (its chunk set may
    have changed) and bump the generation of the scope that can see it, so only keys of
    users who could retrieve the new chunks stop matching."""

    cache = default_cache()
    if cache is None:
        return
    try:
        await cache.purge_document(document_id)
        await cache.bump_generation(PUBLIC_SCOPE if is_public else department_scope(department_id))
    except Exception:
        logger.warning(
            "retrieval cache: invalidation after embed of %s failed", document_id, exc_info=True
        )
