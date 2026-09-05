import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1 import chat, documents, health, ingestion
from app.core.config import settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import UniSageException
from app.core.middleware import request_logging_middleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# `settings.DEBUG` only controls our own verbose output (see
# app/core/graph_trace.py's prompt dump) - it must NOT raise the root level,
# or every third-party library's own DEBUG logs (httpx, httpcore, the
# OpenAI SDK's vendored httpx fork, SQLAlchemy's engine echo) drown out the
# one thing worth reading here: which graph node a request went through.
for _noisy_logger in (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "openai",
    "openai._base_client",
    "sqlalchemy.engine",
    "asyncio",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log application startup and shutdown boundaries."""

    del app
    logger.info("Starting %s in [%s] mode", settings.APP_NAME, settings.APP_ENV)
    yield
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
