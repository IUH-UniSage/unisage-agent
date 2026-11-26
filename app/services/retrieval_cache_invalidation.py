"""The retrieval cache's view of document changes (docs/specs/SPEC-retrieval-cache.md,
"Hook invalidation"). Every hook is a no-op when the cache is off and fails open: a
Redis error here must never fail the ingest, edit or delete that triggered it."""

from __future__ import annotations

import logging

from app.integrations.retrieval_cache import PUBLIC_SCOPE, default_cache, department_scope
from app.rag.vectorstore.qdrant_store import WILDCARD_DEPARTMENT_ID

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


def visibility_scopes(department_id: str | None, *, is_public: bool) -> list[str]:
    """The generation scopes whose keys may now retrieve a document with this access payload.

    A public document is read through `public`, which every key depends on. A department's
    document is read by that department's users and by wildcard (`*`) grants such as
    SUPER_ADMIN, whose keys depend on `dept:*`, not on `dept:{id}`: bumping only the
    department left a wildcard key serving its pre-ingest "no information" answer until the
    TTL ran out (25/11/2026, SA-001 asking about a document just embedded)."""

    if is_public:
        return [PUBLIC_SCOPE]
    scopes = [department_scope(department_id)] if department_id else []
    return [*scopes, department_scope(WILDCARD_DEPARTMENT_ID)]


async def invalidate_after_embed(document_id: str, *, department_id: str, is_public: bool) -> None:
    """New or re-embedded content: drop what referenced the document (its chunk set may
    have changed) and bump the generation of every scope that can see it, so only keys of
    users who could retrieve the new chunks stop matching."""

    await invalidate_visibility_change(
        document_id, department_id=department_id, is_public=is_public
    )


async def invalidate_visibility_change(
    document_id: str, *, department_id: str | None, is_public: bool
) -> None:
    """The document's chunks are (newly) visible under this access payload - an embed, or an
    edit of department/level/public. Entries that referenced it are purged; keys of everyone
    who can now see it change, so a cached "no information" answer of a user who just gained
    access is not served again."""

    cache = default_cache()
    if cache is None:
        return
    try:
        await cache.purge_document(document_id)
        for scope in visibility_scopes(department_id, is_public=is_public):
            await cache.bump_generation(scope)
    except Exception:
        logger.warning(
            "retrieval cache: invalidation after a visibility change of %s failed",
            document_id,
            exc_info=True,
        )
