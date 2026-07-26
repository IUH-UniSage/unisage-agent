from fastapi import APIRouter

from app.rag.ingestion.service import IngestionService
from app.schemas.document import DocumentIngestionRequest, DocumentIngestionResponse

router = APIRouter(tags=["Ingestion"])
ingestion_service = IngestionService()


@router.post("/ingestion", response_model=DocumentIngestionResponse)
async def ingest_document(
    request: DocumentIngestionRequest,
) -> DocumentIngestionResponse:
    """Parse and chunk text; persistence and embeddings are later pipeline stages."""

    result = ingestion_service.ingest(
        source=request.source,
        content=request.content,
        metadata=request.metadata,
    )
    return DocumentIngestionResponse(
        source=result.source,
        chunk_count=len(result.chunks),
        chunks=result.chunks,
    )
