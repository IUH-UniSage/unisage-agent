"""Turns any AI-model failure (CHAT / EMBEDDING / EXTRACTION) into a specific,
client-facing explanation instead of a generic "có lỗi xảy ra".

`describe_llm_failure()` is the one place that knows how to read every shape such a
failure arrives in - a raw provider SDK exception (openai, google-genai), PydanticAI's
`ModelHTTPError`/`ModelAPIError` wrapping, `EmbeddingProviderError` wrapping (with the
real cause on `__cause__`), `NoAvailableCredentialError` (with the cause on
`last_error`), budget rejections, registry/config problems - and reduce it to an
`LLMFailure`: a stable reason, the matching `ErrorCode`, and a Vietnamese message naming
which model purpose failed and why (HTTP status included when there is one).

Messages never interpolate provider text - only fixed sentences, the purpose label, the
HTTP status and, for an unrecognized failure, the exception's class name - so nothing
credential-bearing can reach the client (same rule as `app.graph.stream_error_codes`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

import httpx
import openai
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    UnexpectedModelBehavior,
)

from app.core.budget.tracker import RequestBudgetRejectedError
from app.core.errors.error_codes import ErrorCode
from app.core.errors.exceptions import UniSageException
from app.core.errors.llm_error_classifier import (
    GoogleQuotaWindow,
    google_rate_limit,
    is_quota_exhausted,
    provider_error_details,
    provider_status_code,
)
from app.core.errors.provider_errors import (
    EmbeddingBudgetRejectedError,
    EmbeddingProviderError,
    MalformedExtractionResponseError,
)
from app.core.llm.provider_models import UnsupportedProviderError
from app.core.registry.embedding_identity import EmbeddingIdentityMismatchError
from app.core.registry.errors import (
    CredentialRpmSaturatedError,
    NoAvailableCredentialError,
    NoBudgetAvailableError,
)
from app.core.registry.model_registry import (
    CredentialConfig,
    ModelRegistryError,
    active_credentials_for,
)
from app.core.security.redaction import safe_error_message
from app.core.security.ssrf_guard import SsrfBlockedError

_PURPOSE_LABELS = {
    "CHAT": "Chat",
    "EMBEDDING": "Embedding",
    "EXTRACTION": "Extraction",
}

_CHECK_CONFIG = "Kiểm tra trang Cấu hình AI."


class FailureReason(StrEnum):
    """Why an AI-model call failed - stable, machine-readable. Sent as the SSE
    `event: error` code and the HTTP `errors.reason`, and stored as a suspended
    credential's circuit-breaker marker (`ModelRouter.record_failure`)."""

    LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_AUTH_FAILED = "LLM_AUTH_FAILED"
    LLM_QUOTA_EXHAUSTED = "LLM_QUOTA_EXHAUSTED"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_MODEL_NOT_FOUND = "LLM_MODEL_NOT_FOUND"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_CONFLICT = "LLM_CONFLICT"
    LLM_INPUT_TOO_LARGE = "LLM_INPUT_TOO_LARGE"
    LLM_REQUEST_REJECTED = "LLM_REQUEST_REJECTED"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_CONNECTION_ERROR = "LLM_CONNECTION_ERROR"
    LLM_CONTENT_FILTERED = "LLM_CONTENT_FILTERED"
    LLM_MALFORMED_RESPONSE = "LLM_MALFORMED_RESPONSE"
    LLM_PROVIDER_UNSUPPORTED = "LLM_PROVIDER_UNSUPPORTED"
    LLM_URL_BLOCKED = "LLM_URL_BLOCKED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    BUDGET_THROTTLED = "BUDGET_THROTTLED"
    EMBEDDING_IDENTITY_MISMATCH = "EMBEDDING_IDENTITY_MISMATCH"
    EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
    LLM_UNKNOWN_ERROR = "LLM_UNKNOWN_ERROR"


@dataclass(frozen=True)
class LLMFailure:
    """`reason` is a stable machine-readable cause (also the SSE `event: error` code);
    `message` is the friendly, purpose-specific sentence to show as-is."""

    reason: FailureReason
    error_code: ErrorCode
    message: str
    retryable: bool
    purpose: str | None

    def details(self) -> dict[str, str]:
        """The `errors` map of an HTTP error response: which purpose failed, and why."""

        details = {"reason": self.reason.value}
        if self.purpose:
            details["purpose"] = self.purpose
        return details


class LLMCallException(UniSageException):
    """A UniSageException built from an `LLMFailure`, for HTTP (non-streaming) endpoints -
    `app.api.errors`' handler renders it with the failure's own status/code/message."""

    def __init__(self, failure: LLMFailure) -> None:
        super().__init__(failure.error_code, message=failure.message, errors=failure.details())
        self.failure = failure


def _label(purpose: str | None, model: str | None = None) -> str:
    base = "Mô hình AI" if purpose is None else f"Mô hình {_PURPOSE_LABELS.get(purpose, purpose)}"
    return f"{base} [{model}]" if model else base


def credential_nickname(credential: CredentialConfig | None) -> str | None:
    """The operator's nickname for the credential, else its model name - what to show
    next to the purpose so an error says WHICH model failed."""

    if credential is None:
        return None
    return credential.display_name or credential.model_name


def stamp_failed_model(exc: BaseException, credential: CredentialConfig) -> None:
    """Remembers which credential's model `exc` came from, so a later
    `describe_llm_failure(exc)` (e.g. the SSE error) can name it."""

    try:
        exc.failed_model = credential_nickname(credential)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - exotic exception types without __dict__
        pass


def _root_cause(exc: BaseException) -> BaseException:
    """Follows wrapper exceptions down to the actual provider/config failure."""

    seen: set[int] = set()
    current = exc
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, NoAvailableCredentialError) and current.last_error is not None:
            current = current.last_error
            continue
        if isinstance(current, EmbeddingIdentityMismatchError | EmbeddingBudgetRejectedError):
            return current
        if isinstance(current, EmbeddingProviderError | UnexpectedModelBehavior):
            cause = current.__cause__
            if cause is not None:
                current = cause
                continue
        if isinstance(current, NoAvailableCredentialError) and isinstance(
            current.__cause__, ModelRegistryError
        ):
            return current.__cause__
        return current
    return current


def _purpose_of(exc: BaseException, default: str | None) -> str | None:
    if isinstance(exc, NoAvailableCredentialError | NoBudgetAvailableError):
        return exc.purpose
    if isinstance(exc, EmbeddingProviderError):
        return "EMBEDDING"
    return default


def _failure(
    reason: FailureReason,
    error_code: ErrorCode,
    message: str,
    *,
    retryable: bool,
    purpose: str | None,
) -> LLMFailure:
    return LLMFailure(
        reason=reason,
        error_code=error_code,
        message=message,
        retryable=retryable,
        purpose=purpose,
    )


# Provider wording for "the prompt is longer than the model's context window" - usually a
# plain 400, so the status alone can't tell it apart from any other bad request.
_INPUT_TOO_LARGE_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "input token count",
    "exceeds the maximum number of tokens",
)


def _input_too_large(exc: BaseException) -> bool:
    text = " ".join(
        str(source) for source in provider_error_details(exc) if source is not None
    ).lower()
    return any(marker in text for marker in _INPUT_TOO_LARGE_MARKERS)


def _by_status(
    status: int, root: BaseException, purpose: str | None, model: str | None
) -> LLMFailure:
    who = _label(purpose, model)
    if status == 401:
        return _failure(
            FailureReason.LLM_AUTH_FAILED,
            ErrorCode.LLM_AUTH_FAILED,
            f"{who}: API key không hợp lệ hoặc đã bị thu hồi (HTTP 401). {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    if status == 403:
        return _failure(
            FailureReason.LLM_AUTH_FAILED,
            ErrorCode.LLM_AUTH_FAILED,
            f"{who}: API key không có quyền dùng mô hình này hoặc dự án chưa bật API "
            f"(HTTP 403). {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    if status == 404:
        return _failure(
            FailureReason.LLM_MODEL_NOT_FOUND,
            ErrorCode.LLM_MODEL_NOT_FOUND,
            f"{who}: nhà cung cấp không tìm thấy mô hình/endpoint (HTTP 404) - kiểm tra "
            f"tên model và base URL. {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    if status == 408:
        return _failure(
            FailureReason.LLM_TIMEOUT,
            ErrorCode.LLM_TIMEOUT,
            f"{who}: nhà cung cấp phản hồi quá lâu (HTTP 408), thử lại sau nhé.",
            retryable=True,
            purpose=purpose,
        )
    if status == 409:
        return _failure(
            FailureReason.LLM_CONFLICT,
            ErrorCode.LLM_REQUEST_REJECTED,
            f"{who}: nhà cung cấp báo xung đột yêu cầu (HTTP 409), thử lại sau ít giây.",
            retryable=True,
            purpose=purpose,
        )
    if status == 413 or (400 <= status < 500 and _input_too_large(root)):
        return _failure(
            FailureReason.LLM_INPUT_TOO_LARGE,
            ErrorCode.LLM_REQUEST_REJECTED,
            f"{who}: nội dung gửi đi vượt giới hạn độ dài/context của mô hình (HTTP {status}).",
            retryable=False,
            purpose=purpose,
        )
    if status == 429:
        rate_limit = google_rate_limit(*provider_error_details(root))
        if rate_limit is not None and rate_limit.window is GoogleQuotaWindow.DAY:
            return _failure(
                FailureReason.LLM_RATE_LIMITED,
                ErrorCode.LLM_RATE_LIMITED,
                f"{who}: API key đã dùng hết lượt gọi trong ngày của model này (HTTP 429), "
                "tự reset lúc 0h giờ Pacific (14h-15h giờ Việt Nam).",
                retryable=True,
                purpose=purpose,
            )
        if is_quota_exhausted(*provider_error_details(root)):
            return _failure(
                FailureReason.LLM_QUOTA_EXHAUSTED,
                ErrorCode.LLM_QUOTA_EXHAUSTED,
                f"{who}: tài khoản nhà cung cấp đã hết hạn mức/credit (HTTP 429). "
                f"Nạp thêm credit hoặc đổi API key. {_CHECK_CONFIG}",
                retryable=False,
                purpose=purpose,
            )
        return _failure(
            FailureReason.LLM_RATE_LIMITED,
            ErrorCode.LLM_RATE_LIMITED,
            f"{who}: nhà cung cấp đang giới hạn tốc độ gọi (HTTP 429), thử lại sau ít phút.",
            retryable=True,
            purpose=purpose,
        )
    if 400 <= status < 500:
        return _failure(
            FailureReason.LLM_REQUEST_REJECTED,
            ErrorCode.LLM_REQUEST_REJECTED,
            f"{who}: nhà cung cấp từ chối yêu cầu (HTTP {status}) - có thể do model không "
            f"hỗ trợ tham số/tính năng này hoặc nội dung quá dài. {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    return _failure(
        FailureReason.LLM_PROVIDER_ERROR,
        ErrorCode.LLM_PROVIDER_ERROR,
        f"{who}: máy chủ nhà cung cấp đang gặp sự cố (HTTP {status}), thử lại sau nhé.",
        retryable=True,
        purpose=purpose,
    )


# Short phrase per stored circuit-breaker reason (see `ModelRouter.record_failure`).
_SUSPENSION_PHRASES: dict[str, str] = {
    FailureReason.LLM_AUTH_FAILED: "API key không hợp lệ hoặc không có quyền",
    FailureReason.LLM_QUOTA_EXHAUSTED: "hết hạn mức/credit",
    FailureReason.LLM_RATE_LIMITED: "bị giới hạn tốc độ gọi",
    FailureReason.LLM_MODEL_NOT_FOUND: "sai tên model hoặc base URL",
    FailureReason.LLM_REQUEST_REJECTED: "nhà cung cấp từ chối yêu cầu",
    FailureReason.LLM_CONFLICT: "nhà cung cấp báo xung đột",
    FailureReason.LLM_PROVIDER_ERROR: "máy chủ nhà cung cấp gặp sự cố",
    FailureReason.LLM_TIMEOUT: "nhà cung cấp phản hồi quá lâu",
    FailureReason.LLM_CONNECTION_ERROR: "không kết nối được tới nhà cung cấp",
    FailureReason.LLM_URL_BLOCKED: "base URL bị chặn vì lý do bảo mật",
    FailureReason.LLM_PROVIDER_UNSUPPORTED: "nhà cung cấp không được hỗ trợ",
    FailureReason.LLM_MALFORMED_RESPONSE: "mô hình trả về sai định dạng",
    FailureReason.LLM_CONTENT_FILTERED: "nội dung bị nhà cung cấp chặn",
}
_TRANSIENT_REASONS: frozenset[str] = frozenset(
    {
        FailureReason.LLM_RATE_LIMITED,
        FailureReason.LLM_PROVIDER_ERROR,
        FailureReason.LLM_TIMEOUT,
        FailureReason.LLM_CONNECTION_ERROR,
        FailureReason.LLM_CONFLICT,
    }
)


def _describe_suspension_reasons(reasons: tuple[str, ...]) -> str:
    """Joins the phrase for each distinct stored reason once, in order (e.g. "API key
    không hợp lệ..., hết hạn mức/credit"); codes without a phrase (unknown or older
    markers) are left out."""

    phrases = dict.fromkeys(_SUSPENSION_PHRASES[r] for r in reasons if r in _SUSPENSION_PHRASES)
    return ", ".join(phrases)


def _budget_failure(reason: str, purpose: str | None, model: str | None) -> LLMFailure:
    who = _label(purpose, model)
    if "THROTTLED" in reason:
        return _failure(
            FailureReason.BUDGET_THROTTLED,
            ErrorCode.LLM_BUDGET_EXCEEDED,
            f"{who}: hệ thống đang xử lý nhiều yêu cầu cùng lúc (giới hạn ngân sách), "
            "thử lại sau ít giây.",
            retryable=True,
            purpose=purpose,
        )
    return _failure(
        FailureReason.BUDGET_EXCEEDED,
        ErrorCode.LLM_BUDGET_EXCEEDED,
        f"{who}: đã đạt giới hạn ngân sách sử dụng, thử lại sau khi ngân sách được làm mới.",
        retryable=False,
        purpose=purpose,
    )


def _stamped_model(exc: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        model = getattr(current, "failed_model", None)
        if model:
            return str(model)
        current = current.__cause__ or getattr(current, "last_error", None)
    return None


def describe_llm_failure(
    exc: BaseException,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
) -> LLMFailure:
    """Classifies one AI-model failure. Never raises. `purpose` is the caller's best guess
    (e.g. "CHAT" for the chat graph); an exception that knows its own purpose
    (`NoAvailableCredentialError`, `EmbeddingProviderError`, ...) overrides it.
    `credential` (or a model stamped on the exception chain by `stamp_failed_model`) puts
    the model's nickname in the message."""

    purpose = _purpose_of(exc, purpose)
    model = credential_nickname(credential) or _stamped_model(exc)
    if isinstance(exc, NoAvailableCredentialError) and exc.last_error is not None:
        cause = describe_llm_failure(exc.last_error, purpose, credential)
        if cause.reason != FailureReason.LLM_UNKNOWN_ERROR:
            return cause
        return _failure(
            FailureReason.LLM_UNAVAILABLE,
            ErrorCode.LLM_ALL_CREDENTIALS_SUSPENDED,
            f"{_label(purpose, model)}: mọi credential đều gọi thất bại (lỗi cuối: "
            f"{type(_root_cause(exc.last_error)).__name__}). {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    root = _root_cause(exc)
    purpose = _purpose_of(root, purpose) if root is not exc else purpose
    model = model or _stamped_model(root)
    who = _label(purpose, model)

    if isinstance(root, ModelRegistryError) or (
        isinstance(root, NoAvailableCredentialError) and not active_credentials_for(root.purpose)
    ):
        return _failure(
            FailureReason.LLM_NOT_CONFIGURED,
            ErrorCode.LLM_NOT_CONFIGURED,
            f"{who}: chưa có credential nào đang hoạt động. Thêm/kích hoạt credential cho "
            f"mục đích {_PURPOSE_LABELS.get(purpose or '', purpose or 'này')} trong trang "
            "Cấu hình AI.",
            retryable=False,
            purpose=purpose,
        )
    if isinstance(root, NoAvailableCredentialError):
        causes = _describe_suspension_reasons(root.suspension_reasons)
        detail = (
            f"do lỗi gần đây: {causes}"
            if causes
            else "do lỗi gần đây (sai API key, hết hạn mức, nhà cung cấp lỗi...)"
        )
        # Only a transient cause (rate limit, provider 5xx, timeout, ...) clears on its own.
        retryable = bool(root.suspension_reasons) and all(
            reason in _TRANSIENT_REASONS for reason in root.suspension_reasons
        )
        return _failure(
            FailureReason.LLM_UNAVAILABLE,
            ErrorCode.LLM_ALL_CREDENTIALS_SUSPENDED,
            f"{who}: mọi credential đang bị tạm ngưng {detail}. "
            + ("Thử lại sau ít phút." if retryable else _CHECK_CONFIG),
            retryable=retryable,
            purpose=purpose,
        )
    if isinstance(root, CredentialRpmSaturatedError):
        return _failure(
            FailureReason.LLM_RATE_LIMITED,
            ErrorCode.LLM_RATE_LIMITED,
            f"{who}: credential đã dùng hết {root.max_rpm} lượt gọi/phút đã cấu hình (maxRpm), "
            "thử lại sau ít giây.",
            retryable=True,
            purpose=purpose,
        )
    if isinstance(root, RequestBudgetRejectedError):
        return _budget_failure(root.reason, purpose, model)
    if isinstance(root, NoBudgetAvailableError):
        return _budget_failure(root.last_deny_reason, purpose, model)
    if isinstance(root, EmbeddingBudgetRejectedError):
        return _budget_failure(root.reason, purpose, model)
    if isinstance(root, EmbeddingIdentityMismatchError):
        return _failure(
            FailureReason.EMBEDDING_IDENTITY_MISMATCH,
            ErrorCode.EMBEDDING_IDENTITY_MISMATCH,
            ErrorCode.EMBEDDING_IDENTITY_MISMATCH.message,
            retryable=False,
            purpose="EMBEDDING",
        )
    if isinstance(root, UnsupportedProviderError):
        return _failure(
            FailureReason.LLM_PROVIDER_UNSUPPORTED,
            ErrorCode.LLM_PROVIDER_UNSUPPORTED,
            f"{who}: nhà cung cấp '{root.provider}' chưa được hệ thống hỗ trợ. {_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )
    if isinstance(root, SsrfBlockedError):
        return _failure(
            FailureReason.LLM_URL_BLOCKED,
            ErrorCode.LLM_PROVIDER_UNSUPPORTED,
            f"{who}: base URL của credential trỏ tới địa chỉ mạng bị chặn vì lý do bảo mật. "
            f"{_CHECK_CONFIG}",
            retryable=False,
            purpose=purpose,
        )

    status = provider_status_code(root)
    if status is not None:
        return _by_status(status, root, purpose, model)

    if isinstance(
        root, openai.APITimeoutError | httpx.TimeoutException | asyncio.TimeoutError | TimeoutError
    ):
        return _failure(
            FailureReason.LLM_TIMEOUT,
            ErrorCode.LLM_TIMEOUT,
            f"{who}: nhà cung cấp phản hồi quá lâu (timeout), thử lại sau nhé.",
            retryable=True,
            purpose=purpose,
        )
    if isinstance(root, openai.APIConnectionError | httpx.TransportError | ModelAPIError):
        return _failure(
            FailureReason.LLM_CONNECTION_ERROR,
            ErrorCode.LLM_CONNECTION_ERROR,
            f"{who}: không kết nối được tới nhà cung cấp - kiểm tra base URL và kết nối mạng "
            "của máy chủ.",
            retryable=True,
            purpose=purpose,
        )
    if isinstance(root, ContentFilterError):
        return _failure(
            FailureReason.LLM_CONTENT_FILTERED,
            ErrorCode.LLM_REQUEST_REJECTED,
            f"{who}: nhà cung cấp chặn nội dung này theo chính sách an toàn, hãy diễn đạt lại.",
            retryable=False,
            purpose=purpose,
        )
    if isinstance(root, MalformedExtractionResponseError | UnexpectedModelBehavior):
        return _failure(
            FailureReason.LLM_MALFORMED_RESPONSE,
            ErrorCode.LLM_PROVIDER_ERROR,
            f"{who}: mô hình trả về phản hồi không đúng định dạng, thử lại hoặc đổi mô hình.",
            retryable=True,
            purpose=purpose,
        )

    if isinstance(root, EmbeddingProviderError):
        # Raised with only a message (no provider exception underneath) - e.g. the identity
        # registration call to backend-java failed.
        return _failure(
            FailureReason.EMBEDDING_PROVIDER_ERROR,
            ErrorCode.EMBEDDING_PROVIDER_ERROR,
            ErrorCode.EMBEDDING_PROVIDER_ERROR.message,
            retryable=True,
            purpose="EMBEDDING",
        )

    return _failure(
        FailureReason.LLM_UNKNOWN_ERROR,
        ErrorCode.LLM_PROVIDER_ERROR,
        f"{who}: gọi mô hình thất bại ({type(root).__name__}). Xem log máy chủ để biết chi tiết.",
        retryable=True,
        purpose=purpose,
    )


_PROVIDER_MODULE_PREFIXES = ("openai", "google", "pydantic_ai", "httpx", "httpcore")


def is_model_failure(exc: BaseException) -> bool:
    """True when `exc` came from an AI-model call (provider SDK, PydanticAI, registry, budget,
    ...) rather than from somewhere else entirely (a bug, Qdrant, the DB) - lets a caller that
    runs more than just model calls (the chat graph) avoid blaming the model for an unrelated
    failure."""

    if isinstance(exc, EmbeddingProviderError | NoAvailableCredentialError):
        return True
    if describe_llm_failure(exc).reason != FailureReason.LLM_UNKNOWN_ERROR:
        return True
    module = type(_root_cause(exc)).__module__ or ""
    return module.startswith(_PROVIDER_MODULE_PREFIXES)


_ADMIN_MESSAGE_MAX_LENGTH = 500


def admin_failure_message(
    exc: BaseException,
    *,
    purpose: str | None,
    api_key: str | None,
    credential: CredentialConfig | None = None,
) -> str:
    """The text stored for admins (credential health `lastErrorMessage`, verification job
    `errorMessage`): the friendly cause first, then the redacted provider detail - so the
    admin page says "API key không hợp lệ (HTTP 401)" up front instead of only a raw English
    SDK string. Capped at 500 chars (Java's `last_error_message` column)."""

    friendly = describe_llm_failure(exc, purpose, credential).message
    detail = safe_error_message(exc, api_key)
    return f"{friendly} Chi tiết: {detail}"[:_ADMIN_MESSAGE_MAX_LENGTH]
