from typing import Any

from app.core.error_codes import ErrorCode


class UniSageException(Exception):
    """Base Exception for UniSage AI Agent Service."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class InvalidQueryException(UniSageException):
    """Exception raised when student query is invalid or empty."""

    def __init__(self, message: str = "Query string cannot be empty."):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_QUERY,
            status_code=400,
        )


class InvalidInternalSecretException(UniSageException):
    """Exception raised when a request is missing or has a wrong `X-Internal-Secret`.

    This gates every client-facing route so it can only be reached through
    the API Gateway (the only holder of the shared secret) rather than
    directly - which is what makes gateway-injected headers like
    `X-User-Department` trustworthy in the first place.
    """

    def __init__(self) -> None:
        super().__init__(
            message="Forbidden: invalid or missing X-Internal-Secret header.",
            error_code=ErrorCode.UNAUTHORIZED,
            status_code=403,
        )


class UnsupportedFileTypeException(UniSageException):
    """Exception raised when a file extension has no ingestion parser."""

    def __init__(self, filename: str):
        super().__init__(
            message=f"File '{filename}' has an unsupported extension for ingestion.",
            error_code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            status_code=415,
        )


class StrategyFileTypeMismatchException(UniSageException):
    """Exception raised when a chunking strategy cannot apply to the object's file type."""

    def __init__(self, strategy: str, filename: str):
        super().__init__(
            message=f"Strategy '{strategy}' cannot be applied to file '{filename}'.",
            error_code=ErrorCode.STRATEGY_FILE_TYPE_MISMATCH,
            status_code=422,
        )


class MissingTrustedContextException(UniSageException):
    """Exception raised when the gateway-injected trusted headers are absent."""

    def __init__(self, header_name: str):
        super().__init__(
            message=f"Required trusted header '{header_name}' is missing.",
            error_code=ErrorCode.MISSING_TRUSTED_CONTEXT,
            status_code=400,
        )


class InvalidTrustedContextException(UniSageException):
    """Exception raised when a gateway-injected trusted header is present but malformed.

    Distinct from `MissingTrustedContextException`: this is "the header is
    there but its JSON is broken or missing a required field", not "the
    header is absent altogether".
    """

    def __init__(self, header_name: str, reason: str):
        super().__init__(
            message=f"Trusted header '{header_name}' is malformed: {reason}",
            error_code=ErrorCode.INVALID_TRUSTED_CONTEXT,
            status_code=400,
        )


class InsufficientDocumentPermissionException(UniSageException):
    """Exception raised when the caller lacks DOCUMENT_ALL/DOCUMENT_CREATE permission."""

    def __init__(self) -> None:
        super().__init__(
            message="Forbidden: requires DOCUMENT_ALL or DOCUMENT_CREATE permission.",
            error_code=ErrorCode.FORBIDDEN_DOCUMENT_PERMISSION,
            status_code=403,
        )


class DepartmentAccessDeniedException(UniSageException):
    """Exception raised when the caller isn't granted access to the requested department.

    Covers both "the department isn't in the caller's department_access at
    all" and "it is, but the requested access_level exceeds the level
    granted for that department".
    """

    def __init__(self, department_id: str) -> None:
        super().__init__(
            message=f"Forbidden: caller is not granted sufficient access to department "
            f"'{department_id}'.",
            error_code=ErrorCode.FORBIDDEN_DEPARTMENT_ACCESS,
            status_code=403,
        )


class LLMProviderException(UniSageException):
    """Exception raised when LLM provider API fails."""

    def __init__(self, message: str = "LLM Provider API encountered an error."):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_PROVIDER_ERROR,
            status_code=502,
        )
