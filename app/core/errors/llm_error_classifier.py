"""Classifies a provider-call failure as `TRANSIENT` or `PERMANENT` — plan.md's
`CredentialHealthErrorType` concept (see `unisage-backend`'s
`entity/enums/CredentialHealthErrorType.java`), mirrored here as a separate Python enum since
Task 10's router and health-report call live entirely on this side.

`provider_models.py` (Task 5) only ever builds a `Model`/`Provider` pair for `openai`, `google`
(`google-genai`), and the OpenAI-compatible `SELF_HOSTED` transport (which is the *same* `openai`
SDK under the hood — there is no separate exception hierarchy to add for it). Each of those
provider SDKs raises its own exception types on a failed call; on top of that, per ADR 0005,
PydanticAI's model-native layer (`pydantic_ai/models/{openai,google}.py`)
sometimes re-wraps an HTTP-status failure into `pydantic_ai.exceptions.ModelHTTPError` before it
reaches the caller, or a connection/timeout failure into the status-code-less
`pydantic_ai.exceptions.ModelAPIError`. `classify_llm_error()` recognizes both the raw SDK
exception and the PydanticAI-wrapped one for every provider above. Providers are added below one
at a time as each one is confirmed against the actual installed SDK.

Unrecognized exceptions classify as `TRANSIENT` — todo.md is explicit that this must fail open
toward retrying, never toward permanently disabling a credential whose failure mode isn't
understood.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import google.genai.errors as google_errors
import httpx
import openai
from pydantic_ai.exceptions import ModelHTTPError

from app.core.security.ssrf_guard import SsrfBlockedError


class EmbeddingProviderError(Exception):
    """Raised when the ACTIVE EMBEDDING credential itself is unusable — the actual provider call
    failed (auth/connection/rate-limit/...), there is no ACTIVE EMBEDDING credential at all, or
    the embedding identity guard (`app.core.registry.embedding_identity`) refused to use
    it. Embedding never auto-fails-over — there is no other credential to
    route to, so this is always terminal for the job. Every caller
    (`OpenAIEmbedder.embed`, and transitively `app.worker.celery_app.embed_chunks` and
    `app.rag.retrieval.service.RetrievalService`) must let this escape uncaught rather than
    treat it as a per-chunk data problem.

    `credential` (when known) is attached so a caller several frames away (`embed_chunks`) can
    still build a `report_health` call without having to re-derive which credential failed.
    """

    def __init__(self, message: str, *, credential: Any = None) -> None:
        super().__init__(message)
        self.credential = credential


class EmbeddingBudgetRejectedError(EmbeddingProviderError):
    """An embedding call was refused by budget enforcement (request-level reservation or the
    PROVIDER-scope acquire) before any provider call was made. A subclass so every existing
    "abort the whole job" handling of `EmbeddingProviderError` still applies, but distinct so
    the client can be told it is a budget limit, not a broken credential. `reason` is the
    tracker's result string (e.g. "REJECT_EXCEEDED", "DENY_THROTTLED")."""

    def __init__(self, message: str, *, reason: str, credential: Any = None) -> None:
        super().__init__(message, credential=credential)
        self.reason = reason


class MalformedExtractionResponseError(Exception):
    """Raised by a caller (never by a provider SDK itself) when a credential's response parsed
    fine at the transport level but didn't match the shape the caller actually needed - e.g. a
    fallback EXTRACTION credential's JSON missing the expected keys (see
    `app/rag/enrichment/multi_representation.py`). Always PERMANENT: a credential that returns
    the wrong shape isn't a transient blip, it needs SA attention rather than an automatic retry
    against the exact same credential."""


class ErrorType(StrEnum):
    """Mirrors the Java-side `CredentialHealthErrorType` enum's two values and their meaning —
    same names, separate enum, since nothing here is serialized directly to that Java type
    (Task 10 builds the health-report body from this)."""

    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"


# Substrings that indicate a rate-limit response is actually "the wallet is empty" rather than
# "you're calling too fast right now" — checked against whatever text each SDK actually exposes
# for a 429 (a `code`/`type`/`status` field where the SDK parses one out, or the raw message/body
# text where it doesn't). Retrying an *out-of-credit* 429 never helps, so it's PERMANENT; a plain
# "too many requests" 429 is TRANSIENT.
_QUOTA_EXHAUSTION_MARKERS = (
    "insufficient_quota",
    "quota_exceeded",
    "quota exceeded",
    "billing_hard_limit_reached",
    "billing_not_active",
    "exceeded_quota",
    "exceeded your current quota",
    "out of credit",
)
# Deliberately NOT included: Google's "RESOURCE_EXHAUSTED" status — the Gemini API returns that
# same status for both an actual out-of-quota 429 and a plain short-term rate limit (there's no
# separate status enum value for the two), so treating it as a quota signal on its own would
# make every Google rate limit PERMANENT. Google 429s are distinguished by the marker phrases
# above appearing in `exc.message`/`exc.details`, same as any other provider's message text.


def _text_signals_quota_exhaustion(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _QUOTA_EXHAUSTION_MARKERS)


def _dict_signals_quota_exhaustion(body: dict[str, Any]) -> bool:
    # Different SDKs hand the 429 body to us at different nesting depths (see provider-specific
    # comments where each is wired below) — check both the top level and a nested "error" key
    # rather than assuming one shape.
    candidates: list[Any] = [
        body.get("code"),
        body.get("status"),
        body.get("type"),
        body.get("message"),
    ]
    nested = body.get("error")
    if isinstance(nested, dict):
        candidates.extend(
            [nested.get("code"), nested.get("status"), nested.get("type"), nested.get("message")]
        )
    return any(
        isinstance(candidate, str) and _text_signals_quota_exhaustion(candidate)
        for candidate in candidates
    )


def is_quota_exhausted(*sources: object | None) -> bool:
    """True if any of `sources` (a 429 response body, a status string, a message, ...) looks
    like a quota/credit exhaustion signal rather than a plain rate limit."""

    for source in sources:
        if source is None:
            continue
        if isinstance(source, str) and _text_signals_quota_exhaustion(source):
            return True
        if isinstance(source, dict) and _dict_signals_quota_exhaustion(source):
            return True
    return False


def _classify_by_status_code(status_code: int, *quota_text_sources: object | None) -> ErrorType:
    """Shared status-code → `ErrorType` mapping, used both for a raw SDK exception's own
    `status_code`/`code` and for a PydanticAI-wrapped `ModelHTTPError.status_code`."""

    if status_code in (401, 403):
        return ErrorType.PERMANENT
    if status_code == 429:
        return (
            ErrorType.PERMANENT if is_quota_exhausted(*quota_text_sources) else ErrorType.TRANSIENT
        )
    if status_code >= 500:
        return ErrorType.TRANSIENT
    # Any other 4xx (400/404/409/422/...) is unrecognized here — fail open toward TRANSIENT
    # per todo.md, rather than guessing it's a permanent credential problem.
    return ErrorType.TRANSIENT


def provider_status_code(exc: BaseException) -> int | None:
    """The HTTP status of a failed provider call, whichever layer raised it - `None` for a
    failure with no HTTP response (connection refused, timeout, ...).

    - PydanticAI (ADR 0005) usually re-wraps an HTTP failure into `ModelHTTPError` before a
      call site sees it; a connection/timeout failure becomes the status-less
      `ModelAPIError` instead.
    - openai SDK (`openai` provider and `SELF_HOSTED`'s OpenAI-compatible transport):
      `APIStatusError` and its subclasses (`AuthenticationError`, `RateLimitError`, ...).
    - google-genai SDK: every failure is `errors.APIError`, told apart only by `exc.code`.
    """

    if isinstance(exc, ModelHTTPError | openai.APIStatusError):
        return exc.status_code
    if isinstance(exc, google_errors.APIError):
        return exc.code if isinstance(exc.code, int) else None
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def provider_error_details(exc: BaseException) -> tuple[object | None, ...]:
    """Whatever text/body a provider attached to its error, for marker checks (quota
    exhausted, context too long, ...).

    openai's `exc.body` is already the unwrapped inner `error` object (so `body["code"]` is
    directly e.g. "insufficient_quota"). For google-genai the free-text explanation lives in
    `exc.message`/`exc.details`; its `exc.status` ("RESOURCE_EXHAUSTED") is deliberately not
    used - Google returns the same value for a quota-exhausted and a plain rate-limited 429.
    """

    if isinstance(exc, ModelHTTPError | openai.APIStatusError):
        return (exc.body,)
    if isinstance(exc, google_errors.APIError):
        return (exc.message, exc.details)
    return (str(exc),)


def classify_llm_error(exc: Exception) -> ErrorType:
    """Classifies one provider-call failure. Never raises — an exception type this function
    doesn't recognize falls through to `ErrorType.TRANSIENT`."""

    if isinstance(exc, SsrfBlockedError):
        # A credential whose apiBaseUrl resolves into a blocked range: no amount of retrying
        # changes where that URL points.
        return ErrorType.PERMANENT
    if isinstance(exc, MalformedExtractionResponseError):
        return ErrorType.PERMANENT

    status_code = provider_status_code(exc)
    if status_code is not None:
        return _classify_by_status_code(status_code, *provider_error_details(exc))
    # No HTTP status: a connection error/timeout (openai `APIConnectionError`, PydanticAI
    # `ModelAPIError`, ...) or something unrecognized - both fail open toward retrying.
    return ErrorType.TRANSIENT
