import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1 import chat, health, ingestion
from app.core.config import settings
from app.core.exceptions import UniSageException
from app.core.middleware import request_logging_middleware

logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger(__name__)


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


@app.exception_handler(UniSageException)
async def unisage_exception_handler(request: Request, exc: UniSageException) -> JSONResponse:
    """Map application exceptions to the stable HTTP error contract."""

    del request
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }
