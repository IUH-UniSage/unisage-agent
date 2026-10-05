"""`describe_llm_failure` must turn every AI-model failure shape into a specific reason and
message - never the generic "có lỗi xảy ra" - and never leak provider text into the message."""

from __future__ import annotations

import httpx2
import openai
import pytest
from google.genai import errors as google_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from app.core.budget.tracker import RequestBudgetRejectedError
from app.core.errors.error_codes import ErrorCode
from app.core.errors.llm_failure import (
    FailureReason,
    LLMFailure,
    admin_failure_message,
    describe_llm_failure,
    is_model_failure,
)
from app.core.errors.provider_errors import EmbeddingBudgetRejectedError, EmbeddingProviderError
from app.core.errors.public_errors import (
    can_see_ai_details,
    permissions_from_header,
    public_chat_message,
)
from app.core.llm.provider_models import UnsupportedProviderError
from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
from app.core.registry.errors import NoAvailableCredentialError, NoBudgetAvailableError
from app.core.registry.model_registry import ModelRegistryError
from tests.fixtures.gemini_errors import (
    PER_DAY_QUOTA_ID,
    PER_MINUTE_QUOTA_ID,
    gemini_quota_http_error,
)

_REQUEST = httpx2.Request("POST", "https://api.example.test/v1/chat/completions")


def _openai_status_error(status: int, body: dict[str, object] | None = None) -> openai.APIError:
    response = httpx2.Response(status, request=_REQUEST)
    return openai.APIStatusError("sk-secret-should-not-leak", response=response, body=body)


@pytest.mark.parametrize(
    ("exc", "reason", "error_code"),
    [
        (_openai_status_error(401), "LLM_AUTH_FAILED", ErrorCode.LLM_AUTH_FAILED),
        (_openai_status_error(403), "LLM_AUTH_FAILED", ErrorCode.LLM_AUTH_FAILED),
        (_openai_status_error(404), "LLM_MODEL_NOT_FOUND", ErrorCode.LLM_MODEL_NOT_FOUND),
        (_openai_status_error(409), "LLM_CONFLICT", ErrorCode.LLM_REQUEST_REJECTED),
        (_openai_status_error(400), "LLM_REQUEST_REJECTED", ErrorCode.LLM_REQUEST_REJECTED),
        (
            _openai_status_error(429, {"code": "insufficient_quota"}),
            "LLM_QUOTA_EXHAUSTED",
            ErrorCode.LLM_QUOTA_EXHAUSTED,
        ),
        (_openai_status_error(429), "LLM_RATE_LIMITED", ErrorCode.LLM_RATE_LIMITED),
        (_openai_status_error(503), "LLM_PROVIDER_ERROR", ErrorCode.LLM_PROVIDER_ERROR),
        (ModelHTTPError(401, "gpt-x"), "LLM_AUTH_FAILED", ErrorCode.LLM_AUTH_FAILED),
        (
            google_errors.ClientError(401, {"error": {"message": "API key not valid"}}),
            "LLM_AUTH_FAILED",
            ErrorCode.LLM_AUTH_FAILED,
        ),
        (openai.APITimeoutError(request=_REQUEST), "LLM_TIMEOUT", ErrorCode.LLM_TIMEOUT),
        (
            openai.APIConnectionError(request=_REQUEST),
            "LLM_CONNECTION_ERROR",
            ErrorCode.LLM_CONNECTION_ERROR,
        ),
        (
            ModelAPIError("gpt-x", "conn reset"),
            "LLM_CONNECTION_ERROR",
            ErrorCode.LLM_CONNECTION_ERROR,
        ),
        (
            UnsupportedProviderError("anthropic"),
            "LLM_PROVIDER_UNSUPPORTED",
            ErrorCode.LLM_PROVIDER_UNSUPPORTED,
        ),
        (ModelRegistryError("none"), "LLM_NOT_CONFIGURED", ErrorCode.LLM_NOT_CONFIGURED),
    ],
)
def test_provider_failures_map_to_specific_reasons(
    exc: Exception, reason: str, error_code: ErrorCode
) -> None:
    failure = describe_llm_failure(exc, purpose="CHAT")

    assert failure.reason == reason
    assert failure.error_code is error_code
    assert failure.message.startswith("Mô hình Chat")
    assert "sk-secret" not in failure.message
    assert is_model_failure(exc)


def test_http_status_is_named_in_the_message() -> None:
    failure = describe_llm_failure(_openai_status_error(401), purpose="EXTRACTION")

    assert "Mô hình Extraction" in failure.message
    assert "HTTP 401" in failure.message


def test_missing_extraction_credential_is_not_configured() -> None:
    """`MultiRepresentationEnricher` wraps the registry miss in `NoAvailableCredentialError` -
    the client must still be told the EXTRACTION model isn't configured."""

    try:
        try:
            raise ModelRegistryError("no ACTIVE EXTRACTION credential")
        except ModelRegistryError as cause:
            raise NoAvailableCredentialError("EXTRACTION") from cause
    except NoAvailableCredentialError as exc:
        failure = describe_llm_failure(exc)

    assert failure.reason == "LLM_NOT_CONFIGURED"
    assert failure.purpose == "EXTRACTION"
    assert "Extraction" in failure.message


def test_exhausted_failover_reports_the_last_provider_error() -> None:
    exc = NoAvailableCredentialError("CHAT", last_error=_openai_status_error(401))

    failure = describe_llm_failure(exc)

    assert failure.reason == "LLM_AUTH_FAILED"
    assert failure.purpose == "CHAT"


def test_exhausted_failover_with_unknown_last_error_is_unavailable() -> None:
    exc = NoAvailableCredentialError("CHAT", last_error=RuntimeError("weird"))

    failure = describe_llm_failure(exc)

    assert failure.reason == "LLM_UNAVAILABLE"
    assert "RuntimeError" in failure.message


def test_all_credentials_suspended_when_some_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.errors.llm_failure.active_credentials_for", lambda _purpose: ("cred",)
    )

    failure = describe_llm_failure(NoAvailableCredentialError("CHAT"))

    assert failure.reason == "LLM_UNAVAILABLE"
    assert failure.error_code is ErrorCode.LLM_ALL_CREDENTIALS_SUSPENDED


def test_embedding_wrapper_is_unwrapped_to_its_provider_cause() -> None:
    try:
        try:
            raise _openai_status_error(401)
        except openai.APIError as cause:
            raise EmbeddingProviderError("auth failed") from cause
    except EmbeddingProviderError as exc:
        failure = describe_llm_failure(exc, purpose="CHAT")

    assert failure.reason == "LLM_AUTH_FAILED"
    assert failure.purpose == "EMBEDDING"
    assert failure.message.startswith("Mô hình Embedding")


def test_bare_embedding_error_keeps_the_embedding_provider_code() -> None:
    failure = describe_llm_failure(EmbeddingProviderError("identity PUT failed"))

    assert failure.error_code is ErrorCode.EMBEDDING_PROVIDER_ERROR


def test_identity_mismatch() -> None:
    failure = describe_llm_failure(EmbeddingIdentityMismatchError("mismatch"))

    assert failure.error_code is ErrorCode.EMBEDDING_IDENTITY_MISMATCH


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (RequestBudgetRejectedError("INGEST", "REJECT_EXCEEDED"), "BUDGET_EXCEEDED"),
        (NoBudgetAvailableError("CHAT", "DENY_THROTTLED"), "BUDGET_THROTTLED"),
        (
            EmbeddingBudgetRejectedError("denied", reason="DENY_EXCEEDED"),
            "BUDGET_EXCEEDED",
        ),
    ],
)
def test_budget_rejections(exc: Exception, reason: str) -> None:
    failure = describe_llm_failure(exc)

    assert failure.reason == reason
    assert failure.error_code is ErrorCode.LLM_BUDGET_EXCEEDED


def test_internal_bug_is_not_a_model_failure() -> None:
    assert not is_model_failure(KeyError("x"))


def test_admin_failure_message_puts_the_cause_first_and_redacts() -> None:
    message = admin_failure_message(
        _openai_status_error(401), purpose="EMBEDDING", api_key="sk-secret-should-not-leak"
    )

    assert message.startswith("Mô hình Embedding: API key không hợp lệ")
    assert "Chi tiết:" in message
    assert "sk-secret" not in message
    assert len(message) <= 500


def test_context_length_400_is_input_too_large() -> None:
    exc = _openai_status_error(
        400, {"code": "context_length_exceeded", "message": "maximum context length is 8192"}
    )

    assert describe_llm_failure(exc, purpose="CHAT").reason == "LLM_INPUT_TOO_LARGE"


@pytest.mark.parametrize(
    ("reason", "retryable", "expected_start"),
    [
        ("LLM_CONTENT_FILTERED", False, "Câu hỏi này bị bộ lọc an toàn"),
        ("LLM_INPUT_TOO_LARGE", False, "Câu hỏi hoặc cuộc hội thoại đã quá dài"),
        ("BUDGET_EXCEEDED", False, "Hệ thống đã đạt giới hạn sử dụng"),
        ("LLM_RATE_LIMITED", True, "Trợ lý AI đang bận"),
        ("LLM_AUTH_FAILED", False, "Trợ lý AI đang tạm ngưng do sự cố hệ thống"),
        ("LLM_NOT_CONFIGURED", False, "Trợ lý AI đang tạm ngưng do sự cố hệ thống"),
    ],
)
def test_public_chat_message_categories(reason: str, retryable: bool, expected_start: str) -> None:
    failure = LLMFailure(
        reason=FailureReason(reason),
        error_code=ErrorCode.LLM_PROVIDER_ERROR,
        message="Mô hình Chat: chi tiết kỹ thuật",
        retryable=retryable,
        purpose="CHAT",
    )

    message, _ = public_chat_message(failure, reference="ref12345")

    assert message.startswith(expected_start)
    assert "chi tiết kỹ thuật" not in message
    assert message.endswith("(Mã tham chiếu: ref12345)")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ('["CHAT_MODEL_ALL"]', True),
        ('["LLM_TRACE_LOG_READ"]', True),
        ('["LLM_TRACE_LOG_ALL"]', True),
        # A plain read grant - the USER role held it until V30, so it must not count.
        ('["CHAT_MODEL_READ", "MESSAGE_SEND"]', False),
        ('["DOCUMENT_ALL"]', False),
        ("not-json", False),
        (None, False),
    ],
)
def test_ai_admin_detection(header: str | None, expected: bool) -> None:
    assert can_see_ai_details(permissions_from_header(header)) is expected


# ── Gemini free-tier quota windows ───────────────────────────────────────


def test_gemini_per_minute_quota_429_is_a_rate_limit_not_empty_credit() -> None:
    failure = describe_llm_failure(gemini_quota_http_error(PER_MINUTE_QUOTA_ID), "CHAT")

    assert failure.reason == FailureReason.LLM_RATE_LIMITED
    assert failure.retryable is True
    assert "credit" not in failure.message


def test_gemini_per_day_quota_429_says_when_it_resets() -> None:
    failure = describe_llm_failure(gemini_quota_http_error(PER_DAY_QUOTA_ID), "EXTRACTION")

    assert failure.reason == FailureReason.LLM_RATE_LIMITED
    assert failure.retryable is True
    assert "trong ngày" in failure.message
    assert "credit" not in failure.message
