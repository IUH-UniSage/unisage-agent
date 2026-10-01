"""Handler for an AI-model failure (CHAT / EMBEDDING / EXTRACTION) that escapes a
synchronous endpoint."""

import logging

import google.genai.errors as google_errors
import openai
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior

from app.api.errors.envelope import error_content, new_reference, public_chat_error
from app.core.budget.tracker import RequestBudgetRejectedError
from app.core.errors.llm_error_classifier import EmbeddingProviderError
from app.core.errors.llm_failure import describe_llm_failure
from app.core.llm.provider_models import UnsupportedProviderError
from app.core.registry.errors import NoAvailableCredentialError, NoBudgetAvailableError
from app.core.registry.model_registry import ModelRegistryError
from app.core.security.redaction import safe_error_message
from app.core.security.ssrf_guard import SsrfBlockedError

logger = logging.getLogger(__name__)

# Starlette dispatches by MRO, so subclasses (`EmbeddingIdentityMismatchError`,
# `openai.AuthenticationError`, ...) are covered by their base listed here.
AI_MODEL_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    EmbeddingProviderError,
    ModelRegistryError,
    NoAvailableCredentialError,
    NoBudgetAvailableError,
    RequestBudgetRejectedError,
    UnsupportedProviderError,
    SsrfBlockedError,
    ModelAPIError,
    UnexpectedModelBehavior,
    openai.APIError,
    google_errors.APIError,
)


async def ai_model_failure_handler(request: Request, exc: Exception) -> JSONResponse:
    """Every AI-model failure that escapes a synchronous endpoint (e.g. the "semantic"
    chunking strategy's embedding step, `POST /ingestion/chunking`) - a broken/missing
    EMBEDDING/EXTRACTION/CHAT credential, a provider HTTP error, a budget rejection, an
    embedding identity mismatch. `describe_llm_failure` turns it into a specific code and
    message (which purpose, and why: 401, quota, model not found, not configured, ...)
    instead of the catch-all's generic 500 - or, for a student/guest on a chat route, a
    plain message (see `public_chat_error`).
    """

    failure = describe_llm_failure(exc)
    credential = getattr(exc, "credential", None)
    api_key = getattr(credential, "api_key", None)
    reference = new_reference()
    logger.error(
        "AI model failure (%s, purpose=%s, ref=%s): %s",
        failure.reason,
        failure.purpose,
        reference,
        safe_error_message(exc, api_key),
    )
    message, errors = failure.message, failure.details()
    public = public_chat_error(request, failure, reference=reference)
    if public is not None:
        message, errors = public
    return JSONResponse(
        status_code=failure.error_code.http_status,
        content=error_content(failure.error_code.code, message, errors),
    )
