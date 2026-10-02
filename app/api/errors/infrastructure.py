"""Handlers for a backing service (Postgres, Qdrant, Redis/Celery, backend-java) being
down or rejecting a call."""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from kombu.exceptions import (  # type: ignore[import-untyped]
    OperationalError as KombuOperationalError,
)
from qdrant_client.http.exceptions import ApiException as QdrantApiException
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.api.errors.envelope import error_content, new_reference, public_chat_error
from app.core.errors.error_codes import ErrorCode
from app.core.security.redaction import safe_error_message
from app.integrations.backend_java_client import BackendJavaError

logger = logging.getLogger(__name__)

ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]

# (exception type, code to answer with, component name logged and returned)
INFRASTRUCTURE_FAILURES: tuple[tuple[type[Exception], ErrorCode, str], ...] = (
    (SQLAlchemyError, ErrorCode.DATABASE_ERROR, "postgres"),
    (QdrantApiException, ErrorCode.VECTOR_STORE_ERROR, "qdrant"),
    (RedisError, ErrorCode.TASK_QUEUE_UNAVAILABLE, "redis"),
    (KombuOperationalError, ErrorCode.TASK_QUEUE_UNAVAILABLE, "celery-broker"),
    (BackendJavaError, ErrorCode.BACKEND_JAVA_UNAVAILABLE, "backend-java"),
)


def infrastructure_handler(error_code: ErrorCode, component: str) -> ExceptionHandler:
    """A handler for one backing service: logs the real exception, answers with that
    service's own code/message so the client knows WHAT is unavailable instead of a
    generic 500 (a plain system message for a student/guest on a chat route)."""

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        reference = new_reference()
        logger.error(
            "%s failure (ref=%s): %s",
            component,
            reference,
            safe_error_message(exc),
            exc_info=exc,
        )
        message, errors = error_code.message, {"component": component}
        public = public_chat_error(request, None, reference=reference)
        if public is not None:
            message, errors = public
        return JSONResponse(
            status_code=error_code.http_status,
            content=error_content(error_code.code, message, errors),
        )

    return _handler
