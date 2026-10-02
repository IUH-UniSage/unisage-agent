import logging

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.v1 import chat, documents, health, ingestion
from app.core.config import settings
from app.core.lifespan import lifespan
from app.core.observability.logging_config import configure_logging
from app.core.observability.middleware import request_logging_middleware

# Sets the root format, silences noisy/secret-leaking third-party loggers
# (httpx/httpcore/openai/anthropic/...) and attaches the redaction filter that
# scrubs every log record before it's written - see app/core/observability/logging_config.py.
configure_logging()
logger = logging.getLogger(__name__)

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

register_exception_handlers(app)

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
