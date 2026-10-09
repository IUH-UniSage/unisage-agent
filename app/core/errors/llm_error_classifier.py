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

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import google.genai.errors as google_errors
import httpx
import openai
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from app.core.errors.provider_errors import MalformedExtractionResponseError
from app.core.security.ssrf_guard import SsrfBlockedError


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
    # Z.ai code 1113: "Insufficient balance or no resource package. Please recharge."
    "insufficient balance",
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


class GoogleQuotaWindow(StrEnum):
    """Which rolling window a Gemini 429 says was exceeded - read from the `quotaId` of the
    `google.rpc.QuotaFailure` detail (e.g. `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`,
    `GenerateRequestsPerDayPerProjectPerModel-FreeTier`)."""

    MINUTE = "MINUTE"
    DAY = "DAY"


@dataclass(frozen=True)
class GoogleRateLimit:
    window: GoogleQuotaWindow
    # `google.rpc.RetryInfo.retryDelay` in seconds, when Google sent one.
    retry_delay_seconds: float | None


_QUOTA_FAILURE_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


def _google_error_details(source: object) -> list[dict[str, Any]]:
    """The `error.details` list of a Gemini error body (`{"error": {..., "details": [...]}}`),
    as both `google_errors.APIError.details` and PydanticAI's `ModelHTTPError.body` carry it."""

    if not isinstance(source, dict):
        return []
    error = source.get("error", source)
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, list):
        return []
    return [detail for detail in details if isinstance(detail, dict)]


def _parse_duration_seconds(raw: object) -> float | None:
    # protobuf Duration JSON form: "39s", "0.5s".
    if not isinstance(raw, str) or not raw.endswith("s"):
        return None
    try:
        return max(0.0, float(raw[:-1]))
    except ValueError:
        return None


def google_rate_limit(*sources: object | None) -> GoogleRateLimit | None:
    """The quota window a Gemini 429 body reports as exceeded, or `None` when no source carries a
    `QuotaFailure` detail naming one.

    Gemini answers every free-tier limit hit - per minute and per day alike - with the same
    "You exceeded your current quota, please check your plan and billing details" message, so
    that text says nothing about credit; only the `quotaId` tells the windows apart. A per-day
    violation wins over a per-minute one (it is the longer wait)."""

    window: GoogleQuotaWindow | None = None
    retry_delay: float | None = None
    for source in sources:
        for detail in _google_error_details(source):
            kind = detail.get("@type")
            if kind == _RETRY_INFO_TYPE:
                retry_delay = _parse_duration_seconds(detail.get("retryDelay"))
            elif kind == _QUOTA_FAILURE_TYPE:
                violations = detail.get("violations")
                for violation in violations if isinstance(violations, list) else []:
                    quota_id = violation.get("quotaId") if isinstance(violation, dict) else None
                    if not isinstance(quota_id, str):
                        continue
                    if "PerDay" in quota_id:
                        window = GoogleQuotaWindow.DAY
                    elif window is None and ("PerMinute" in quota_id or "PerSecond" in quota_id):
                        window = GoogleQuotaWindow.MINUTE
    if window is None:
        return None
    return GoogleRateLimit(window=window, retry_delay_seconds=retry_delay)


# Z.ai code 1302 "Rate limit reached for requests": too many requests in flight at once for
# the account and model - clears as soon as one of them finishes.
ZAI_CONCURRENCY_LIMIT_CODE = "1302"


def zai_error_code(*sources: object | None) -> str | None:
    """Z.ai's own business error code (e.g. "1302") from an error body, which arrives either
    unwrapped (`{"code": "1302", ...}`, the openai SDK's `body`) or as `{"error": {...}}`."""

    for source in sources:
        if not isinstance(source, dict):
            continue
        error = source.get("error", source)
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str | int) and str(code).isdigit():
            return str(code)
    return None


def is_quota_exhausted(*sources: object | None) -> bool:
    """True if any of `sources` (a 429 response body, a status string, a message, ...) looks
    like a quota/credit exhaustion signal rather than a plain rate limit.

    A Gemini 429 that names a per-minute/per-day quota window is a rate limit that resets on
    its own, whatever its message says (see `google_rate_limit`)."""

    if google_rate_limit(*sources) is not None:
        return False
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


def is_model_wide_failure(exc: BaseException) -> bool:
    """`True` for a failure that says the provider's MODEL is struggling (5xx, timeout,
    connection error) rather than one credential (429, 401, ...) - every other key on the
    same model would most likely fail the same way, so trying them one by one only adds
    latency. Follows `__cause__` so a wrapped timeout still counts."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = provider_status_code(current)
        if status_code is not None:
            return status_code >= 500
        if isinstance(
            current,
            TimeoutError
            | httpx.TimeoutException
            | httpx.TransportError
            | openai.APIConnectionError
            | ModelAPIError,
        ):
            return True
        current = current.__cause__
    return False
