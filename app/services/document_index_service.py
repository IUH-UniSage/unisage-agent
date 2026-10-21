"""Keeps a document's points in step with edits made in unisage-backend. Every change reaches
every collection in use (`index_collections()`): the live one and an open re-index target, so
the cutover cannot bring back a stale payload or a deleted document."""

import logging

from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.registry.model_registry import index_collections
from app.database.repositories.ingestion_job import delete_draft, set_draft_department
from app.rag.vectorstore import qdrant_store
from app.schemas.document_index import IndexMetadataUpdateRequest
from app.services.retrieval_cache_invalidation import invalidate_document

logger = logging.getLogger(__name__)


async def update_index_metadata(
    client: QdrantClient,
    session: AsyncSession,
    document_id: str,
    request: IndexMetadataUpdateRequest,
) -> int:
    """Rewrites `department`/`access_level`/`is_public` on every point of the document."""

    payload = {
        "department": request.department_id,
        "access_level": request.access_level,
        "is_public": request.is_public,
    }
    updated = 0
    for collection in index_collections():
        updated += qdrant_store.set_document_payload(
            client, document_id, payload, collection=collection
        )
    if request.department_id is not None:
        await set_draft_department(session, document_id, request.department_id)
    await invalidate_document(document_id)
    logger.info(
        "document %s: access payload rewritten on %d point(s) (department=%s, level=%d, public=%s)",
        document_id,
        updated,
        request.department_id,
        request.access_level,
        request.is_public,
    )
    return updated


async def delete_index(client: QdrantClient, session: AsyncSession, document_id: str) -> int:
    """Drops every point and the chunking draft of the document."""

    deleted = 0
    for collection in index_collections():
        deleted += qdrant_store.delete_document_points(client, document_id, collection=collection)
    await delete_draft(session, document_id)
    await invalidate_document(document_id)
    logger.info("document %s: index dropped (%d point(s))", document_id, deleted)
    return deleted
