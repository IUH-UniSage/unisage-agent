import contextlib
import logging

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    Depends,
    Header,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    TrustedContext,
    get_trusted_context,
    require_department_membership,
    require_document_permission,
)
from app.core.config import settings
from app.core.events import ingestion_event_stream
from app.core.exceptions import (
    DepartmentAccessDeniedException,
    EmbeddingChunkSetMismatchException,
    EmbeddingDraftLegacyException,
    EmbeddingDraftMismatchException,
    EmptyDocumentTextException,
    IngestionJobNotFoundException,
    UniSageException,
)
from app.core.security import verify_internal_secret
from app.database.models import DocumentProcessStep
from app.database.repositories.ingestion_job import (
    DraftDTO,
    get_draft,
    mark_embedding,
    upsert_chunking_draft,
)
from app.database.session import get_db_session
from app.rag.chunking import strategy
from app.rag.chunking.validation import validate_chunks
from app.rag.ingestion import minio_client
from app.rag.ingestion.parser import extract_raw_text
from app.schemas.common import ApiResponse
from app.schemas.ingestion import (
    Chunk,
    ChunkingRequest,
    ChunkingResponse,
    EmbeddingAcceptedResponse,
    EmbeddingRequest,
    IngestionJobResponse,
    PreviewRequest,
    PreviewResponse,
    TaskProgress,
)
from app.worker.celery_app import celery_app, embed_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"], dependencies=[Depends(verify_internal_secret)])


@router.post("/ingestion/preview", response_model=ApiResponse[PreviewResponse])
async def preview_document(
    request: PreviewRequest,
    context: TrustedContext = Depends(require_document_permission),
) -> ApiResponse[PreviewResponse]:
    """Fetch the stored object and return its raw extracted text."""

    require_department_membership(request.department_id, context)
    content = minio_client.get_object_bytes(request.object_key)
    raw_text = extract_raw_text(content, request.object_key)
    return ApiResponse.success(PreviewResponse(raw_text=raw_text))


@router.post("/ingestion/chunking", response_model=ApiResponse[ChunkingResponse])
async def chunk_document(
    request: ChunkingRequest,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ChunkingResponse]:
    """Fetch the stored object fresh and chunk it with the requested strategy."""

    require_department_membership(request.department_id, context)
    content = minio_client.get_object_bytes(request.object_key)
    chunks = strategy.dispatch(request.strategy, request.params, content, request.object_key)
    if not chunks:
        raise EmptyDocumentTextException()
    validate_chunks(chunks)
    await upsert_chunking_draft(
        db_session,
        document_id=request.document_id,
        object_key=request.object_key,
        department_id=request.department_id,
        strategy=request.strategy.value,
        params=request.params,
        chunks=chunks,
    )
    return ApiResponse.success(ChunkingResponse(chunks=chunks))


@router.get("/ingestion/jobs/{document_id}", response_model=ApiResponse[IngestionJobResponse])
async def get_ingestion_job(
    document_id: str,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[IngestionJobResponse]:
    """Return the resumable process-log record for a document, or 404 if none exists.

    When the record is at the `embedding` step, the response carries the
    live Celery task progress (`task_state` / `task_percent`) so the
    client's reconciliation sweep needs only this one authorized call.
    """

    draft = await get_draft(db_session, document_id)
    if draft is None:
        raise IngestionJobNotFoundException(document_id)

    if draft.department_id is not None:
        require_department_membership(draft.department_id, context)
    else:
        logger.warning(
            "Process log for %s has no department_id (pre-migration row); "
            "skipping the membership check",
            document_id,
        )

    task_progress = _draft_task_progress(draft)
    return ApiResponse.success(IngestionJobResponse.from_draft(draft, task_progress))


@router.post(
    "/ingestion/embedding",
    response_model=ApiResponse[EmbeddingAcceptedResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
async def embed_document(
    request: EmbeddingRequest,
    context: TrustedContext = Depends(require_document_permission),
    db_session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[EmbeddingAcceptedResponse]:
    """Dispatch a chunk list for background enrichment + embedding.

    Execution order (deliberately NOT task-number order - see plan.md v5
    Task 4.4's note): Task 4.5's canonical-metadata-from-draft merge runs
    FIRST, then Task 4.4's `validate_chunks()` runs on the MERGED result,
    right before dispatching Celery. `request.chunks` is never trusted for
    anything but `content` - every other field (heading_path, page_start,
    source_locator, column_names, ...) is taken from the stored draft, so a
    client (or a script bypassing the UI wizard) cannot forge structural
    metadata.
    """

    _require_department_access_within_grant(request, context)

    draft = await get_draft(db_session, request.document_id)
    merged_chunks = _merge_canonical_chunks(request, draft)
    validate_chunks(merged_chunks)

    task = embed_chunks.delay(
        request.document_id,
        request.object_key,
        [chunk.model_dump(mode="json") for chunk in merged_chunks],
        request.department_id,
        request.access_level,
    )
    await mark_embedding(db_session, document_id=request.document_id, celery_task_id=task.id)
    return ApiResponse.success(EmbeddingAcceptedResponse(task_id=task.id))


def _merge_canonical_chunks(request: EmbeddingRequest, draft: DraftDTO | None) -> list[Chunk]:
    """Task 4.5: build the chunk list actually sent for embedding, trusting
    ONLY `content` from `request.chunks` - every other field comes from the
    stored chunking draft (the canonical source of truth), never the client.

    Raises (see plan.md v5, Task 4.5's acceptance criteria):
    - `IngestionJobNotFoundException` - no draft at all for `document_id`.
    - `EmbeddingDraftMismatchException` - draft exists but its
      `department_id`/`object_key` doesn't match this request (never use a
      draft belonging to a different context, even if `document_id` matches).
    - `EmbeddingDraftLegacyException` (HTTP 409) - the draft (or any of its
      chunks) was not produced by the current chunking logic
      (`chunking_version != settings.CHUNKING_VERSION`, which includes
      `"legacy"`, or `source_type`/`block_index` still `None`) - cannot be
      embedded safely; the document must be re-chunked. Chunks already
      embedded in Qdrant are never touched.
    - `EmbeddingChunkSetMismatchException` - `request.chunks` isn't a valid
      subset of the canonical draft: a duplicate `chunk_index`, or an index
      the draft doesn't have. Never merges partially.

    The client may send FEWER chunks than the draft: chunks the user deleted
    in the wizard before embedding are simply left out and never embedded.
    """

    if draft is None:
        raise IngestionJobNotFoundException(request.document_id)

    if draft.department_id != request.department_id or draft.object_key != request.object_key:
        raise EmbeddingDraftMismatchException(request.document_id)

    canonical_chunks = draft.chunks
    is_outdated_draft = any(
        chunk.chunking_version != settings.CHUNKING_VERSION
        or chunk.source_type is None
        or chunk.block_index is None
        for chunk in canonical_chunks
    )
    if is_outdated_draft:
        raise EmbeddingDraftLegacyException(request.document_id)

    canonical_by_index = {chunk.chunk_index: chunk for chunk in canonical_chunks}

    request_indexes = [chunk.chunk_index for chunk in request.chunks]
    if len(set(request_indexes)) != len(request_indexes):
        raise EmbeddingChunkSetMismatchException(
            request.document_id, "chunk_index bị trùng lặp trong yêu cầu"
        )
    unknown_indexes = sorted(set(request_indexes) - canonical_by_index.keys())
    if unknown_indexes:
        raise EmbeddingChunkSetMismatchException(
            request.document_id,
            f"chunk_index {unknown_indexes} không có trong bản nháp",
        )

    return [
        canonical_by_index[chunk.chunk_index].model_copy(update={"content": chunk.content})
        for chunk in sorted(request.chunks, key=lambda chunk: chunk.chunk_index)
    ]


def _require_department_access_within_grant(
    request: EmbeddingRequest, context: TrustedContext
) -> None:
    """Raise 403 unless the caller's granted access_level for the department covers the request."""

    granted_level = context.granted_access_level(request.department_id)
    if granted_level is None or request.access_level > granted_level:
        raise DepartmentAccessDeniedException(request.department_id)


def _read_task_progress(task_id: str) -> TaskProgress:
    """Read one embedding task's current {percent, state} from the Celery result backend."""

    result = AsyncResult(task_id, app=celery_app)
    percent = 0
    info = result.info
    if isinstance(info, dict) and "percent" in info:
        percent = int(info["percent"])
    return TaskProgress(percent=percent, state=result.state)


def _draft_task_progress(draft: DraftDTO) -> TaskProgress | None:
    """The live task progress for a draft that has an embed in flight, else None."""

    if draft.current_step != DocumentProcessStep.EMBEDDING or draft.celery_task_id is None:
        return None
    return _read_task_progress(draft.celery_task_id)


@router.websocket("/ingestion/events")
async def ingestion_events(
    websocket: WebSocket,
    x_internal_secret: str | None = Header(default=None),
    x_user_department_access: str | None = Header(default=None),
    x_user_permissions: str | None = Header(default=None),
) -> None:
    """Fan out ingestion progress/completion frames to one browser client.

    Auth mirrors every other route: the API Gateway injects `X-Internal-Secret`
    plus the JWT-derived `X-User-*` headers on the handshake (a browser's
    native WebSocket can't set headers - the gateway reads the identity from
    the `accessToken` cookie). We never trust a department list the client
    controls. Each frame is forwarded only if its `department_id` is one the
    caller is granted access to.
    """

    if not x_internal_secret or x_internal_secret != settings.INTERNAL_SECRET_KEY:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        context = await get_trusted_context(x_user_department_access, x_user_permissions)
    except UniSageException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    allowed_departments = {entry.department_id for entry in context.department_access}
    forward_all = context.has_wildcard_department_access

    await websocket.accept()
    try:
        async for frame in ingestion_event_stream():
            if forward_all or frame.get("department_id") in allowed_departments:
                await websocket.send_json(frame)
    except WebSocketDisconnect:
        return
    finally:
        with contextlib.suppress(RuntimeError):
            await websocket.close()
