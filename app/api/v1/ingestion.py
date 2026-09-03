import asyncio
import contextlib

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TrustedContext, require_department_membership, require_document_permission
from app.core.exceptions import DepartmentAccessDeniedException
from app.core.security import verify_internal_secret
from app.database.repositories.ingestion_job import (
    delete_draft,
    get_draft,
    mark_embedding,
    upsert_chunking_draft,
)
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
    EmbeddingStatusResponse,
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
async def preview_document(
    request: PreviewRequest,
    context: TrustedContext = Depends(require_document_permission),
) -> PreviewResponse:
    """Fetch the stored object and return its raw extracted text."""

    require_department_membership(request.department_id, context)
    content = minio_client.get_object_bytes(request.object_key)
    raw_text = extract_raw_text(content, request.object_key)
    return PreviewResponse(raw_text=raw_text)


@router.post("/ingestion/chunking", response_model=ChunkingResponse)
async def chunk_document(
    request: ChunkingRequest,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChunkingResponse:
    """Fetch the stored object fresh and chunk it with the requested strategy."""

    require_department_membership(request.department_id, context)
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
        task_id=draft.celery_task_id,
    )


@router.delete("/ingestion/jobs/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ingestion_job(
    document_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> None:
    """Clear a document's draft/in-flight-embedding row once its client has
    observed the embed task reach a terminal state (see
    `DocumentProcessStep.EMBEDDING`'s docstring for why this can't be done
    from the Celery worker itself). A no-op if there's no row."""

    await delete_draft(db_session, document_id)


@router.post(
    "/ingestion/embedding",
    response_model=EmbeddingAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def embed_document(
    request: EmbeddingRequest,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> EmbeddingAcceptedResponse:
    """Dispatch the client-approved chunk list for background enrichment + embedding."""

    _require_department_access_within_grant(request, context)

    task = embed_chunks.delay(
        request.document_id,
        request.object_key,
        [chunk.model_dump(mode="json") for chunk in request.chunks],
        request.department_id,
        request.access_level,
    )
    response = EmbeddingAcceptedResponse(task_id=task.id)
    await mark_embedding(db_session, document_id=request.document_id, celery_task_id=task.id)
    return response


def _require_department_access_within_grant(
    request: EmbeddingRequest, context: TrustedContext
) -> None:
    """Raise 403 unless the caller's granted access_level for the department covers the request."""

    granted_level = next(
        (
            entry.access_level
            for entry in context.department_access
            if entry.department_id == request.department_id
        ),
        None,
    )
    if granted_level is None or request.access_level > granted_level:
        raise DepartmentAccessDeniedException(request.department_id)


def _read_task_progress(task_id: str) -> EmbeddingStatusResponse:
    """Read one embedding task's current {percent, state} from the Celery result backend.

    Shared by the WebSocket loop below (one push per poll interval) and
    `GET /ingestion/embedding/{task_id}/status` (one pull per HTTP call) -
    same read, two transports.
    """

    result = AsyncResult(task_id, app=celery_app)
    percent = 0
    info = result.info
    if isinstance(info, dict) and "percent" in info:
        percent = int(info["percent"])
    return EmbeddingStatusResponse(percent=percent, state=result.state)


@router.get("/ingestion/embedding/{task_id}/status", response_model=EmbeddingStatusResponse)
async def get_embedding_status(task_id: str) -> EmbeddingStatusResponse:
    """Poll-friendly HTTP equivalent of one WebSocket progress frame.

    Lets a client detect a task finishing without keeping a WebSocket (and
    therefore a wizard page) open - e.g. the Processing queue polling every
    document it knows has an in-flight embed, so `Document.status` still
    gets updated even if nobody reopens that document's wizard to watch the
    WebSocket directly.
    """

    return _read_task_progress(task_id)


@router.websocket("/ingestion/embedding/{task_id}/progress")
async def embedding_progress(websocket: WebSocket, task_id: str) -> None:
    """Push percent-complete progress frames until the embedding task finishes or fails.

    A client can disconnect at any point mid-loop (tab closed, page
    navigated away, resumed-then-abandoned view) - `send_json` then raises
    `WebSocketDisconnect` because the peer is already gone. That's an
    expected shutdown path here, not an error: swallow it and skip the
    `close()` call entirely, since closing an already-disconnected socket
    itself raises (`Cannot call "send" once a close message has been
    sent.`), which would otherwise surface as a second, misleading
    exception in the logs for what is just a normal disconnect.
    """

    await websocket.accept()
    try:
        while True:
            progress = _read_task_progress(task_id)
            await websocket.send_json({"percent": progress.percent, "state": progress.state})
            if progress.state in _TERMINAL_STATES:
                break
            await asyncio.sleep(_PROGRESS_POLL_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return
    finally:
        with contextlib.suppress(RuntimeError):
            await websocket.close()
