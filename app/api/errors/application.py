"""Handlers for the application's own errors: `UniSageException`, request validation,
the chunker's internal `TableStructureError`, and the catch-all fallback."""

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.errors.envelope import error_content, new_reference, public_chat_error
from app.core.config import settings
from app.core.errors.error_codes import ErrorCode
from app.core.errors.exceptions import UniSageException
from app.core.errors.llm_failure import LLMCallException
from app.rag.chunking.table_row import TableStructureError

logger = logging.getLogger(__name__)


async def unisage_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map application exceptions to the same envelope backend-java's
    `GlobalExceptionHandler` produces for its `AppException`."""

    assert isinstance(exc, UniSageException)
    message, errors = exc.message, exc.errors
    if isinstance(exc, LLMCallException):
        # e.g. "no CHAT credential" raised before the chat stream starts.
        reference = new_reference()
        logger.error("AI model failure (%s, ref=%s): %s", exc.failure.reason, reference, message)
        public = public_chat_error(request, exc.failure, reference=reference)
        if public is not None:
            message, errors = public
    return JSONResponse(
        status_code=exc.error_code.http_status,
        content=error_content(exc.error_code.code, message, errors),
    )


async def table_structure_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """`TableStructureError` is an internal chunker BUG/invariant failure
    (a row's structure doesn't match its table's expectations) - never a
    user input/config problem, unlike `ChunkValidationException`/
    `ChunkingConfigException`, which get their own `UniSageException` 4xx
    handling above. Log this at CRITICAL with full structured
    context (document_id, block_index, table_id, row_index, expected vs
    actual cell count, a non-reversible digest of the offending row) so an
    on-call engineer can actually debug it - but the row's raw/untruncated
    text (which may hold PII pulled straight from an uploaded document) is
    only ever logged when `settings.APP_DEBUG` is on (local/test runs), never
    in a production log line. The client still only sees a generic 500.
    """

    assert isinstance(exc, TableStructureError)
    document_id = "unknown"
    try:
        body = await request.json()
        if isinstance(body, dict):
            document_id = str(body.get("document_id", "unknown"))
    except Exception:  # pragma: no cover - defensive only, body may be unreadable/non-JSON
        pass

    context: dict[str, Any] = {
        "document_id": document_id,
        "block_index": exc.block_index,
        "table_id": exc.table_id,
        "row_index": exc.row_index,
        "expected_cell_count": exc.expected_cell_count,
        "actual_cell_count": exc.actual_cell_count,
        "row_sample_digest": exc.row_sample_digest,
    }
    if settings.APP_DEBUG:
        # Debug/test only - never reached in production, where APP_DEBUG=False.
        context["raw_row"] = exc.raw_row
    logger.critical("TableStructureError: internal chunker invariant violated: %s", context)

    return JSONResponse(
        status_code=ErrorCode.INTERNAL_ERROR.http_status,
        content=error_content(ErrorCode.INTERNAL_ERROR.code, ErrorCode.INTERNAL_ERROR.message),
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map Pydantic/FastAPI request-validation failures, mirroring Java's
    `MethodArgumentNotValidException` handler: one `errors` entry per
    invalid field."""

    del request
    assert isinstance(exc, RequestValidationError)
    field_errors = {
        ".".join(str(part) for part in error["loc"][1:]) or str(error["loc"][-1]): error["msg"]
        for error in exc.errors()
    }
    return JSONResponse(
        status_code=ErrorCode.VALIDATION_ERROR.http_status,
        content=error_content(
            ErrorCode.VALIDATION_ERROR.code, ErrorCode.VALIDATION_ERROR.message, field_errors
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all fallback, mirroring Java's `Exception.class` handler: log the
    full traceback server-side, return a generic 500 to the client."""

    del request
    reference = new_reference()
    logger.exception("Unhandled exception (ref=%s)", reference, exc_info=exc)
    return JSONResponse(
        status_code=ErrorCode.INTERNAL_ERROR.http_status,
        content=error_content(
            ErrorCode.INTERNAL_ERROR.code,
            f"{ErrorCode.INTERNAL_ERROR.message} (Mã tham chiếu: {reference})",
            {"reference": reference},
        ),
    )
