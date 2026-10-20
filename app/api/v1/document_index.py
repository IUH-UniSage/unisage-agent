"""Backend-only routes (unisage-backend -> agent). Under `/api/v1/internal/` because the gateway
adds `X-Internal-Secret` to every request it proxies to this service but blocks
`/api/v1/ai/internal/**` - so only a direct internal caller can reach them. No user context:
unisage-backend has already authorized the document edit."""

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.security import verify_internal_secret
from app.database.session import get_db_session
from app.rag.vectorstore import qdrant_store
from app.schemas.common import ApiResponse
from app.schemas.document_index import (
    IndexDeleteResponse,
    IndexMetadataUpdateRequest,
    IndexMetadataUpdateResponse,
)
from app.services import document_index_service

router = APIRouter(
    prefix="/internal", tags=["Internal"], dependencies=[Depends(verify_internal_secret)]
)


def get_qdrant_client() -> QdrantClient:
    return qdrant_store.get_client()


@router.patch(
    "/documents/{document_id}/index-metadata",
    response_model=ApiResponse[IndexMetadataUpdateResponse],
)
async def update_index_metadata(
    document_id: str,
    request: IndexMetadataUpdateRequest,
    db_session: AsyncSession = Depends(get_db_session),
    client: QdrantClient = Depends(get_qdrant_client),
) -> ApiResponse[IndexMetadataUpdateResponse]:
    """The document's department / access level / public flag changed: rewrite its points."""

    updated = await document_index_service.update_index_metadata(
        client, db_session, document_id, request
    )
    return ApiResponse.success(IndexMetadataUpdateResponse(updated_points=updated))


@router.delete(
    "/documents/{document_id}/index",
    response_model=ApiResponse[IndexDeleteResponse],
)
async def delete_index(
    document_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    client: QdrantClient = Depends(get_qdrant_client),
) -> ApiResponse[IndexDeleteResponse]:
    """The document got a new file or was deleted: drop its points and chunking draft."""

    deleted = await document_index_service.delete_index(client, db_session, document_id)
    return ApiResponse.success(IndexDeleteResponse(deleted_points=deleted))
