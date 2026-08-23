import asyncio

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TrustedContext, get_trusted_context
from app.core.security import verify_internal_secret
from app.database.repositories.ingestion_job import delete_draft, get_draft, upsert_chunking_draft
from app.database.session import get_db_session
from app.rag.chunking import strategy
from app.rag.ingestion import minio_client
from app.rag.ingestion.parser import extract_raw_text
from app.rag.ingestion.service import IngestionService
from app.schemas.document import DocumentIngestionRequest, DocumentIngestionResponse
from app.schemas.ingestion import (
    ChunkingRequest,
    ChunkingResponse,
    EmbeddingAcceptedResponse,
    EmbeddingRequest,
    IngestionJobResponse,
    PreviewRequest,
    PreviewResponse,
)
from app.worker.celery_app import celery_app, embed_chunks

_TERMINAL_STATES = {"SUCCESS", "FAILURE"}
_PROGRESS_POLL_INTERVAL_SECONDS = 0.5

router = APIRouter(tags=["Ingestion"], dependencies=[Depends(verify_internal_secret)])
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


@router.post("/ingestion/preview", response_model=PreviewResponse)
async def preview_document(request: PreviewRequest) -> PreviewResponse:
    """Fetch the stored object and return its raw extracted text."""

    content = minio_client.get_object_bytes(request.object_key)
    raw_text = extract_raw_text(content, request.object_key)
    return PreviewResponse(raw_text=raw_text)


@router.post("/ingestion/chunking", response_model=ChunkingResponse)
async def chunk_document(
    request: ChunkingRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> ChunkingResponse:
    """Fetch the stored object fresh and chunk it with the requested strategy."""

    content = minio_client.get_object_bytes(request.object_key)
    chunks = strategy.dispatch(request.strategy, request.params, content, request.object_key)
    response = ChunkingResponse(chunks=chunks)
    await upsert_chunking_draft(
        db_session,
        document_id=request.document_id,
        object_key=request.object_key,
        strategy=request.strategy.value,
        params=request.params,
        chunks=chunks,
    )
    return response


@router.get("/ingestion/jobs/{document_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    document_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> IngestionJobResponse:
    """Return the resumable chunking draft for a document, or 404 if none exists."""

    draft = await get_draft(db_session, document_id)
    if draft is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No draft found.")
    return IngestionJobResponse(
        object_key=draft.object_key,
        current_step=draft.current_step.value,
        chunking_strategy=draft.chunking_strategy,
        chunking_params=draft.chunking_params,
        chunks=draft.chunks,
    )


@router.post(
    "/ingestion/embedding",
    response_model=EmbeddingAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def embed_document(
    request: EmbeddingRequest,
    context: TrustedContext = Depends(get_trusted_context),
    db_session: AsyncSession = Depends(get_db_session),
) -> EmbeddingAcceptedResponse:
    """Dispatch the client-approved chunk list for background enrichment + embedding."""

    task = embed_chunks.delay(
        request.document_id,
        request.object_key,
        [chunk.model_dump(mode="json") for chunk in request.chunks],
        context.department,
        context.access_level,
    )
    response = EmbeddingAcceptedResponse(task_id=task.id)
    await delete_draft(db_session, request.document_id)
    return response


@router.websocket("/ingestion/embedding/{task_id}/progress")
async def embedding_progress(websocket: WebSocket, task_id: str) -> None:
    """Push percent-complete progress frames until the embedding task finishes or fails."""

    await websocket.accept()
    result = AsyncResult(task_id, app=celery_app)
    try:
        while True:
            state = result.state
            percent = 0
            info = result.info
            if isinstance(info, dict) and "percent" in info:
                percent = int(info["percent"])
            await websocket.send_json({"percent": percent, "state": state})
            if state in _TERMINAL_STATES:
                break
            await asyncio.sleep(_PROGRESS_POLL_INTERVAL_SECONDS)
    finally:
        await websocket.close()
