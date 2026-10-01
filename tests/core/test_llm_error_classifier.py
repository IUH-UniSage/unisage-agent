"""Tests for `classify_llm_error()` — todo.md Task 9.

Every case constructs the real SDK/PydanticAI exception class with realistic constructor
arguments (a real `httpx`/`httpx2` `Request`/`Response`, or the SDK's own `response_json`/`body`
shape), not a bare `Exception("insufficient_quota")` string match — the classifier inspects
typed attributes (`status_code`/`code`, `body`/`details`), so a test double that skips those
attributes would pass for the wrong reason.

Covers every provider `provider_models.py` (Task 5) actually wires: `openai` (also exercises
`SELF_HOSTED`, which reuses the same SDK/exception hierarchy), `google` (`google-genai`) — plus
`pydantic_ai.exceptions.ModelHTTPError`/`ModelAPIError` (the PydanticAI-wrapped form of the same
failures) and `SsrfBlockedError`.
"""

from __future__ import annotations

import google.genai.errors as google_errors
import httpx2
import openai
import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from app.core.errors.llm_error_classifier import ErrorType, classify_llm_error
from app.core.errors.provider_errors import MalformedExtractionResponseError
from app.core.security.ssrf_guard import SsrfBlockedError


def _openai_response(status_code: int, *, error: dict) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx2.Response(status_code, request=request, json={"error": error})


class TestSsrfBlocked:
    def test_ssrf_blocked_is_always_permanent(self) -> None:
        exc = SsrfBlockedError("RESOLVED_IP_BLOCKED", host="169.254.169.254")
        assert classify_llm_error(exc) == ErrorType.PERMANENT


class TestMalformedExtractionResponse:
    def test_malformed_extraction_response_is_always_permanent(self) -> None:
        exc = MalformedExtractionResponseError("missing summary/questions keys")
        assert classify_llm_error(exc) == ErrorType.PERMANENT


class TestOpenAI:
    """Also covers `SELF_HOSTED` credentials — `provider_models.py` builds them with the same
    `OpenAIChatModel`/`OpenAIProvider` pair, so they raise from this exact hierarchy."""

    def test_authentication_error_is_permanent(self) -> None:
        response = _openai_response(
            401,
            error={
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            },
        )
        exc = openai.AuthenticationError(
            "Incorrect API key provided", response=response, body=response.json()["error"]
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_permission_denied_is_permanent(self) -> None:
        response = _openai_response(
            403, error={"message": "Forbidden", "type": "invalid_request_error", "code": None}
        )
        exc = openai.PermissionDeniedError(
            "Forbidden", response=response, body=response.json()["error"]
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_rate_limit_with_insufficient_quota_is_permanent(self) -> None:
        response = _openai_response(
            429,
            error={
                "message": "You exceeded your current quota, please check your plan and billing details.",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            },
        )
        exc = openai.RateLimitError("quota", response=response, body=response.json()["error"])
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_rate_limit_without_quota_code_is_transient(self) -> None:
        response = _openai_response(
            429,
            error={"message": "Rate limit reached for requests", "type": "requests", "code": None},
        )
        exc = openai.RateLimitError(
            "rate limited", response=response, body=response.json()["error"]
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_internal_server_error_is_transient(self) -> None:
        response = _openai_response(
            500, error={"message": "The server had an error", "type": "server_error", "code": None}
        )
        exc = openai.InternalServerError(
            "server error", response=response, body=response.json()["error"]
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_api_connection_error_is_transient(self) -> None:
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = openai.APIConnectionError(request=request)
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_api_timeout_error_is_transient(self) -> None:
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = openai.APITimeoutError(request=request)
        assert classify_llm_error(exc) == ErrorType.TRANSIENT


class TestGoogle:
    def test_permission_denied_is_permanent(self) -> None:
        exc = google_errors.ClientError(
            401, {"error": {"status": "PERMISSION_DENIED", "message": "API key not valid"}}
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_resource_exhausted_with_quota_message_is_permanent(self) -> None:
        # Google's own status string is "RESOURCE_EXHAUSTED" for *both* an out-of-quota 429 and
        # a plain rate-limited 429 - the distinguishing signal has to come from the message text.
        exc = google_errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for quota metric 'requests'",
                }
            },
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_resource_exhausted_without_quota_message_is_transient(self) -> None:
        exc = google_errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Rate limit exceeded, please retry after some time",
                }
            },
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_server_error_is_transient(self) -> None:
        exc = google_errors.ServerError(
            503, {"error": {"status": "UNAVAILABLE", "message": "model overloaded"}}
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT


class TestModelHTTPErrorWrapping:
    """`pydantic_ai.exceptions.ModelHTTPError` — the form each provider's model-native layer
    (`pydantic_ai/models/{openai,google}.py`) actually re-raises as, per ADR 0005."""

    def test_401_is_permanent(self) -> None:
        exc = ModelHTTPError(
            status_code=401,
            model_name="gpt-4o-mini",
            body={"code": "invalid_api_key", "message": "bad key"},
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_403_is_permanent(self) -> None:
        exc = ModelHTTPError(
            status_code=403, model_name="gemini-1.5-flash", body={"message": "forbidden"}
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_429_with_openai_shaped_body_and_quota_code_is_permanent(self) -> None:
        # openai's model-native layer forwards the already-unwrapped inner error dict.
        exc = ModelHTTPError(
            status_code=429,
            model_name="gpt-4o-mini",
            body={"code": "insufficient_quota", "message": "x"},
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_429_with_nested_error_body_and_quota_code_is_permanent(self) -> None:
        # Some model-native layers forward the raw, still-nested `{"error": {...}}` body.
        exc = ModelHTTPError(
            status_code=429,
            model_name="gpt-4o-mini",
            body={"error": {"code": "rate_limit_exceeded"}},
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

        exc_quota = ModelHTTPError(
            status_code=429,
            model_name="gpt-4o-mini",
            body={"error": {"code": "quota_exceeded", "message": "x"}},
        )
        assert classify_llm_error(exc_quota) == ErrorType.PERMANENT

    def test_429_with_string_body_and_quota_text_is_permanent(self) -> None:
        # The body can arrive as a raw string, not a dict.
        exc = ModelHTTPError(
            status_code=429,
            model_name="gpt-4o-mini",
            body="You exceeded your current quota",
        )
        assert classify_llm_error(exc) == ErrorType.PERMANENT

    def test_429_plain_rate_limit_is_transient(self) -> None:
        exc = ModelHTTPError(
            status_code=429,
            model_name="gpt-4o-mini",
            body={"code": None, "message": "too many requests"},
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_5xx_is_transient(self) -> None:
        exc = ModelHTTPError(status_code=503, model_name="gpt-4o-mini", body="overloaded")
        assert classify_llm_error(exc) == ErrorType.TRANSIENT

    def test_unrecognized_4xx_is_transient(self) -> None:
        # e.g. 400 bad request - not one of the acceptance criteria's explicit cases, so this
        # must fail open toward TRANSIENT rather than guessing PERMANENT.
        exc = ModelHTTPError(
            status_code=400, model_name="gpt-4o-mini", body={"message": "invalid request"}
        )
        assert classify_llm_error(exc) == ErrorType.TRANSIENT


class TestModelAPIErrorWrapping:
    """The status-code-less wrap PydanticAI uses for a connection/timeout failure (every
    provider's `_map_api_errors` routes `APIConnectionError` here, never into `ModelHTTPError`)."""

    def test_connection_failure_is_transient(self) -> None:
        exc = ModelAPIError(model_name="gpt-4o-mini", message="Connection error.")
        assert classify_llm_error(exc) == ErrorType.TRANSIENT


class TestUnrecognized:
    def test_unrecognized_exception_type_is_transient(self) -> None:
        assert classify_llm_error(ValueError("totally unrelated failure")) == ErrorType.TRANSIENT

    def test_ordinary_runtime_error_is_transient(self) -> None:
        assert classify_llm_error(RuntimeError()) == ErrorType.TRANSIENT


@pytest.mark.parametrize("error_type", [ErrorType.TRANSIENT, ErrorType.PERMANENT])
def test_error_type_values_match_java_naming(error_type: ErrorType) -> None:
    """Names (not the enum class itself) match `CredentialHealthErrorType` on the Java side, so
    a human reading a health-report payload doesn't have to learn two vocabularies."""

    assert error_type.value in {"TRANSIENT", "PERMANENT"}
