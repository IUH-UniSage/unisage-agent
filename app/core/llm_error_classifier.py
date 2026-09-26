"""Classifies a provider-call failure as `TRANSIENT` or `PERMANENT` — plan.md's
`CredentialHealthErrorType` concept (see `unisage-backend`'s
`entity/enums/CredentialHealthErrorType.java`), mirrored here as a separate Python enum since
Task 10's router and health-report call live entirely on this side.

`provider_models.py` (Task 5) only ever builds a `Model`/`Provider` pair for `openai`, `google`
(`google-genai`), `groq`, `mistral`, and the OpenAI-compatible `SELF_HOSTED` transport (which is
the *same* `openai` SDK under the hood — there is no separate exception hierarchy to add for it).
Each of those four provider SDKs raises its own exception types on a failed call; on top of that,
per ADR 0005, PydanticAI's model-native layer (`pydantic_ai/models/{openai,groq,mistral,google}.py`)
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

from enum import Enum
from typing import Any

import openai
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from app.core.ssrf_guard import SsrfBlockedError


class ErrorType(str, Enum):
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
    "resource_exhausted",
    "billing_hard_limit_reached",
    "billing_not_active",
    "exceeded_quota",
    "exceeded your current quota",
    "out of credit",
)


def _text_signals_quota_exhaustion(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _QUOTA_EXHAUSTION_MARKERS)


def _dict_signals_quota_exhaustion(body: dict[str, Any]) -> bool:
    # Different SDKs hand the 429 body to us at different nesting depths (see provider-specific
    # comments where each is wired below) — check both the top level and a nested "error" key
    # rather than assuming one shape.
    candidates: list[Any] = [body.get("code"), body.get("status"), body.get("type"), body.get("message")]
    nested = body.get("error")
    if isinstance(nested, dict):
        candidates.extend([nested.get("code"), nested.get("status"), nested.get("type"), nested.get("message")])
    return any(isinstance(candidate, str) and _text_signals_quota_exhaustion(candidate) for candidate in candidates)


def _quota_exhausted(*sources: object | None) -> bool:
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
        return ErrorType.PERMANENT if _quota_exhausted(*quota_text_sources) else ErrorType.TRANSIENT
    if status_code >= 500:
        return ErrorType.TRANSIENT
    # Any other 4xx (400/404/409/422/...) is unrecognized here — fail open toward TRANSIENT
    # per todo.md, rather than guessing it's a permanent credential problem.
    return ErrorType.TRANSIENT


def classify_llm_error(exc: Exception) -> ErrorType:
    """Classifies one provider-call failure. Never raises — an exception type this function
    doesn't recognize falls through to `ErrorType.TRANSIENT`."""

    if isinstance(exc, SsrfBlockedError):
        # A credential whose apiBaseUrl resolves into a blocked range: no amount of retrying
        # changes where that URL points.
        return ErrorType.PERMANENT

    # --- PydanticAI's own wrapping layer (ADR 0005) — checked before any raw SDK type, since a
    # provider call site normally sees these instead of the raw SDK exception. ---
    if isinstance(exc, ModelHTTPError):
        return _classify_by_status_code(exc.status_code, exc.body)
    if isinstance(exc, ModelAPIError):
        # PydanticAI's generic wrap with no HTTP status attached — every provider's
        # `_map_api_errors` (openai.py/groq.py/mistral.py) routes a connection/timeout failure
        # here, never into `ModelHTTPError`. No status code to inspect, so: connection error →
        # TRANSIENT.
        return ErrorType.TRANSIENT

    # --- openai SDK (`openai` provider, and `SELF_HOSTED`'s OpenAI-compatible transport — the
    # same SDK, so the same exception hierarchy applies to both). ---
    if isinstance(exc, openai.APIConnectionError):
        # Covers `openai.APITimeoutError` too (subclasses `APIConnectionError`).
        return ErrorType.TRANSIENT
    if isinstance(exc, openai.APIStatusError):
        # Covers `AuthenticationError` (401), `PermissionDeniedError` (403), `RateLimitError`
        # (429), `InternalServerError` (5xx) and any other status via the generic base class.
        # `exc.body` here is already the unwrapped inner `error` object (openai's own
        # `_make_status_error` does `body.get("error", body)` before constructing the
        # exception), so `exc.body.get("code")` is directly e.g. `"insufficient_quota"`.
        return _classify_by_status_code(exc.status_code, exc.body)

    return ErrorType.TRANSIENT
