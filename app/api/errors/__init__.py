"""Every exception handler of the FastAPI app, registered in one call from `app.main`.

- `application`: our own `UniSageException`s, request validation, internal chunker bugs,
  and the catch-all fallback.
- `ai_models`: CHAT/EMBEDDING/EXTRACTION failures (bad key, quota, not configured, ...).
- `infrastructure`: Postgres, Qdrant, Redis/Celery, backend-java being unavailable.
- `envelope`: the shared `{code, message, errors}` envelope, reference codes, and the
  chat "how much detail may this caller see" decision.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.errors.ai_models import AI_MODEL_EXCEPTION_TYPES, ai_model_failure_handler
from app.api.errors.application import (
    table_structure_error_handler,
    unhandled_exception_handler,
    unisage_exception_handler,
    validation_exception_handler,
)
from app.api.errors.infrastructure import INFRASTRUCTURE_FAILURES, infrastructure_handler
from app.core.errors.exceptions import UniSageException
from app.rag.chunking.table_row import TableStructureError


def register_exception_handlers(app: FastAPI) -> None:
    """Order doesn't matter: Starlette picks the handler of the most specific registered
    class in the raised exception's MRO."""

    # The decorator form, unlike `add_exception_handler`, accepts a handler typed with the
    # concrete exception class it is registered for.
    app.exception_handler(UniSageException)(unisage_exception_handler)
    app.exception_handler(TableStructureError)(table_structure_error_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    for exception_type in AI_MODEL_EXCEPTION_TYPES:
        app.add_exception_handler(exception_type, ai_model_failure_handler)
    for exception_type, error_code, component in INFRASTRUCTURE_FAILURES:
        app.add_exception_handler(exception_type, infrastructure_handler(error_code, component))
    app.add_exception_handler(Exception, unhandled_exception_handler)
