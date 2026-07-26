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


class LLMProviderException(UniSageException):
    """Exception raised when LLM provider API fails."""

    def __init__(self, message: str = "LLM Provider API encountered an error."):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_PROVIDER_ERROR,
            status_code=502,
        )
