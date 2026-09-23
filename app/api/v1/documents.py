import logging

from fastapi import APIRouter, Depends, Query
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TrustedContext, require_department_membership, require_document_permission
from app.core.security import verify_internal_secret
from app.database.repositories.chunk import ChunkRepository
from app.database.session import get_db_session
from app.rag.vectorstore import qdrant_store
from app.schemas.common import ApiResponse, PageResponse
from app.schemas.ingestion import Chunk, IndexedChunk
from app.services.chunk_service import ChunkService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"], dependencies=[Depends(verify_internal_secret)])


def get_qdrant_client() -> QdrantClient:
    return qdrant_store.get_client()


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ApiResponse[PageResponse[list[Chunk]]],
)
async def list_document_chunks(
    document_id: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[PageResponse[list[Chunk]]]:
    """Return one page of a document's chunks, ordered by `chunk_index`.

    404s (via `DocumentChunksNotFoundException`) if the document has no
    chunking draft yet - same "no draft" outcome `GET /ingestion/jobs/{id}`
    already treats as a normal 404, not a server error.
    """

    service = ChunkService(ChunkRepository(db_session))
    chunk_page = await service.list_document_chunks(document_id, page=page, limit=limit)

    if chunk_page.department_id is not None:
        require_department_membership(chunk_page.department_id, context)
    else:
        logger.warning(
            "Process log for %s has no department_id (pre-migration row); "
            "skipping the membership check",
            document_id,
        )

    return ApiResponse.success(
        PageResponse.of(
            chunk_page.chunks, page=page, limit=limit, total_items=chunk_page.total
        )
    )


@router.get(
    "/documents/{document_id}/chunks/indexed",
    response_model=ApiResponse[PageResponse[list[IndexedChunk]]],
)
async def list_indexed_chunks(
    document_id: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
    client: QdrantClient = Depends(get_qdrant_client),
) -> ApiResponse[PageResponse[list[IndexedChunk]]]:
    """Return one page of `document_id`'s chunks as actually indexed in
    Qdrant - including `summary`/`questions` (generated at embed time by
    `MultiRepresentationEnricher`), which the plain `/chunks` draft endpoint
    above never has.

    Empty (never 404) when the document has no indexed chunks yet, since
    that's a normal state (draft chunked but not embedded, or nothing
    ingested at all).
    """

    service = ChunkService(ChunkRepository(db_session))
    indexed_page = service.list_indexed_chunks(client, document_id, page=page, limit=limit)

    if indexed_page.department_id is not None:
        require_department_membership(indexed_page.department_id, context)

    return ApiResponse.success(
        PageResponse.of(
            indexed_page.chunks, page=page, limit=limit, total_items=indexed_page.total
        )
    )


@router.delete(
    "/documents/{document_id}/chunks/indexed/{chunk_id}",
    response_model=ApiResponse[None],
)
async def delete_indexed_chunk(
    document_id: str,
    chunk_id: str,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
    client: QdrantClient = Depends(get_qdrant_client),
) -> ApiResponse[None]:
    """Remove one chunk's point from the live Qdrant index (management action
    for the "indexed chunks" view) - does not touch the Postgres chunking
    draft, only what the RAG pipeline can actually retrieve.

    `chunk_id` must belong to `document_id` - membership is verified against
    the target point's own payload before deleting, so a caller can't delete
    a point in a department they aren't granted access to just by knowing its
    `chunk_id`.
    """

    records = qdrant_store.scroll_chunks_by_document(client, document_id)
    target = next(
        (record for record in records if (record.payload or {}).get("chunk_id") == chunk_id),
        None,
    )
    if target is not None and target.payload:
        require_department_membership(str(target.payload["department"]), context)

    service = ChunkService(ChunkRepository(db_session))
    service.delete_indexed_chunk(client, document_id, chunk_id)
    return ApiResponse.success_without_data()
