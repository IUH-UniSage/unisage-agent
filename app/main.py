import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1 import chat, documents, health, ingestion
from app.core.budget.poller import start_budget_snapshot_poller
from app.core.budget.snapshot import refresh_budget_snapshot
from app.core.config import settings
from app.core.errors.error_codes import ErrorCode
from app.core.errors.exceptions import UniSageException
from app.core.errors.llm_error_classifier import EmbeddingProviderError
from app.core.observability.logging_config import configure_logging
from app.core.observability.middleware import request_logging_middleware
from app.core.pricing.poller import start_pricing_snapshot_poller
from app.core.pricing.snapshot import refresh_pricing_snapshot
from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
from app.core.registry.model_registry import init_model_registry
from app.core.registry.registry_subscriber import start_asyncio_registry_subscriber
from app.core.security.redaction import safe_error_message
from app.rag.chunking.table_row import TableStructureError

# Sets the root format, silences noisy/secret-leaking third-party loggers
# (httpx/httpcore/openai/anthropic/...) and attaches the redaction filter that
# scrubs every log record before it's written - see app/core/observability/logging_config.py.
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log application startup and shutdown boundaries."""

    del app
    logger.info("Starting %s in [%s] mode", settings.APP_NAME, settings.APP_ENV)
    # One-time load of the model registry snapshot from backend-java. No-op when
    # MODEL_REGISTRY_ENABLED=false; when true, raises (and is deliberately left
    # uncaught, failing startup) if there is no ACTIVE CHAT credential.
    await init_model_registry()
    # Hot-reload the cached snapshot without a restart - subscribes to Java's
    # after-commit pub/sub signal and independently polls /version as a self-healing
    # fallback. No-op when the flag above is off.
    subscriber = start_asyncio_registry_subscriber()
    # Same soft-limit posture as the reservation itself: an empty/never-loaded
    # snapshot means no budgets are enforced, fail-open by absence - never fatal to
    # startup even if Java is unreachable right now. Gated on the same flag as the
    # model registry above - a process running on static .env credentials has no
    # live backend-java to fetch a budget snapshot from either.
    # Prices follow the same gate: without a live backend-java every call is UNPRICED.
    budget_poller: asyncio.Task[None] | None = None
    pricing_poller: asyncio.Task[None] | None = None
    if settings.MODEL_REGISTRY_ENABLED:
        await refresh_budget_snapshot()
        budget_poller = start_budget_snapshot_poller()
        await refresh_pricing_snapshot()
        pricing_poller = start_pricing_snapshot_poller()
    yield
    if budget_poller is not None:
        budget_poller.cancel()
    if pricing_poller is not None:
        pricing_poller.cancel()
    await subscriber.stop()
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="UniSage AI Agent Service - RAG Engine and Pydantic Graph state machine",
    lifespan=lifespan,
)

# No CORSMiddleware here - only the API Gateway sets CORS headers (see
# api-gateway/application.yml's globalcors config), same pattern
# backend-java already follows ("disable direct cors in backend and rely
# on gateway"). This service is only ever reached through the Gateway
# (verify_internal_secret gates every route), so a browser never talks to
# it directly - and stacking a second CORSMiddleware here made every
# response carry two Access-Control-Allow-Origin values
# ("http://localhost:5173, *"), which browsers reject outright.
app.middleware("http")(request_logging_middleware)


def _error_content(code: int, message: str, errors: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the `{code, message, errors}` envelope - `data` and `errors` are
    omitted when absent, matching Java's `@JsonInclude(NON_NULL)` on
    `ApiResponse`."""

    content: dict[str, Any] = {"code": code, "message": message}
    if errors:
        content["errors"] = errors
    return content


@app.exception_handler(UniSageException)
async def unisage_exception_handler(request: Request, exc: UniSageException) -> JSONResponse:
    """Map application exceptions to the same envelope backend-java's
    `GlobalExceptionHandler` produces for its `AppException`."""

    del request
    return JSONResponse(
        status_code=exc.error_code.http_status,
        content=_error_content(exc.error_code.code, exc.message, exc.errors),
    )


@app.exception_handler(TableStructureError)
async def table_structure_error_handler(request: Request, exc: TableStructureError) -> JSONResponse:
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
        content=_error_content(ErrorCode.INTERNAL_ERROR.code, ErrorCode.INTERNAL_ERROR.message),
    )


@app.exception_handler(EmbeddingIdentityMismatchError)
async def embedding_identity_mismatch_handler(
    request: Request, exc: EmbeddingIdentityMismatchError
) -> JSONResponse:
    """Raised deep inside `app.core.registry.embedding_identity.ensure_embedding_identity` -
    reached from a synchronous embed call (e.g. the "semantic" chunking strategy's own
    embedding step, `POST /ingestion/chunking`), not from an endpoint that can catch and
    translate it itself. A dedicated handler (not a `UniSageException` subclass, same reasoning
    as `TableStructureError` above) is the one place every such call site's mismatch reaches
    the client as a specific, actionable error instead of falling through to
    `unhandled_exception_handler`'s generic 500.
    """

    del request
    api_key = exc.credential.api_key if exc.credential else None
    logger.error("Embedding identity mismatch: %s", safe_error_message(exc, api_key))
    return JSONResponse(
        status_code=ErrorCode.EMBEDDING_IDENTITY_MISMATCH.http_status,
        content=_error_content(
            ErrorCode.EMBEDDING_IDENTITY_MISMATCH.code, ErrorCode.EMBEDDING_IDENTITY_MISMATCH.message
        ),
    )


@app.exception_handler(EmbeddingProviderError)
async def embedding_provider_error_handler(
    request: Request, exc: EmbeddingProviderError
) -> JSONResponse:
    """Catches every OTHER `EmbeddingProviderError` (auth/network/budget-rejection failures
    during a synchronous embed call) - `EmbeddingIdentityMismatchError` (a subclass) is matched
    by the more specific handler above first, since Starlette dispatches to the most specific
    registered handler in the exception's MRO.
    """

    del request
    api_key = exc.credential.api_key if exc.credential else None
    logger.error("Embedding provider error: %s", safe_error_message(exc, api_key))
    return JSONResponse(
        status_code=ErrorCode.EMBEDDING_PROVIDER_ERROR.http_status,
        content=_error_content(
            ErrorCode.EMBEDDING_PROVIDER_ERROR.code, ErrorCode.EMBEDDING_PROVIDER_ERROR.message
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Map Pydantic/FastAPI request-validation failures, mirroring Java's
    `MethodArgumentNotValidException` handler: one `errors` entry per
    invalid field."""

    del request
    field_errors = {
        ".".join(str(part) for part in error["loc"][1:]) or str(error["loc"][-1]): error["msg"]
        for error in exc.errors()
    }
    return JSONResponse(
        status_code=ErrorCode.VALIDATION_ERROR.http_status,
        content=_error_content(
            ErrorCode.VALIDATION_ERROR.code, ErrorCode.VALIDATION_ERROR.message, field_errors
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all fallback, mirroring Java's `Exception.class` handler: log the
    full traceback server-side, return a generic 500 to the client."""

    del request
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=ErrorCode.INTERNAL_ERROR.http_status,
        content=_error_content(ErrorCode.INTERNAL_ERROR.code, ErrorCode.INTERNAL_ERROR.message),
    )


app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }
