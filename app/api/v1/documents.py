import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TrustedContext, require_department_membership, require_document_permission
from app.core.security import verify_internal_secret
from app.database.repositories.chunk import ChunkRepository
from app.database.session import get_db_session
from app.schemas.common import ApiResponse, PageResponse
from app.schemas.ingestion import Chunk
from app.services.chunk_service import ChunkService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"], dependencies=[Depends(verify_internal_secret)])


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
