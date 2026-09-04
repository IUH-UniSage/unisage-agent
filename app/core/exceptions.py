from app.core.error_codes import ErrorCode


class UniSageException(Exception):
    """Base exception for UniSage AI Agent Service.

    Mirrors backend-java's `AppException`: carries one `ErrorCode` member
    (which supplies the HTTP status and default message) plus an optional
    per-field `errors` map, exactly like Java's `AppException(ErrorCode,
    Map<String, String>)`. `app/main.py`'s exception handler reads
    `error_code.http_status`/`error_code.code` off of it to build the
    response envelope.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str | None = None,
        errors: dict[str, str] | None = None,
    ):
        resolved_message = message or error_code.message
        super().__init__(resolved_message)
        self.message = resolved_message
        self.error_code = error_code
        self.errors = errors or {}


class InvalidQueryException(UniSageException):
    """Exception raised when student query is invalid or empty."""

    def __init__(self, message: str = ErrorCode.INVALID_QUERY.message):
        super().__init__(ErrorCode.INVALID_QUERY, message=message)


class InvalidInternalSecretException(UniSageException):
    """Exception raised when a request is missing or has a wrong `X-Internal-Secret`.

    This gates every client-facing route so it can only be reached through
    the API Gateway (the only holder of the shared secret) rather than
    directly - which is what makes gateway-injected headers like
    `X-User-Department` trustworthy in the first place.
    """

    def __init__(self) -> None:
        super().__init__(ErrorCode.UNAUTHORIZED)


class UnsupportedFileTypeException(UniSageException):
    """Exception raised when a file extension has no ingestion parser."""

    def __init__(self, filename: str):
        super().__init__(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            message=f"Định dạng file '{filename}' chưa được hỗ trợ nạp liệu.",
        )


class StrategyFileTypeMismatchException(UniSageException):
    """Exception raised when a chunking strategy cannot apply to the object's file type."""

    def __init__(self, strategy: str, filename: str):
        super().__init__(
            ErrorCode.STRATEGY_FILE_TYPE_MISMATCH,
            message=f"Chiến lược '{strategy}' không thể áp dụng cho file '{filename}'.",
        )


class MissingTrustedContextException(UniSageException):
    """Exception raised when the gateway-injected trusted headers are absent."""

    def __init__(self, header_name: str):
        super().__init__(
            ErrorCode.MISSING_TRUSTED_CONTEXT,
            message=f"Thiếu header bắt buộc '{header_name}'.",
        )


class InvalidTrustedContextException(UniSageException):
    """Exception raised when a gateway-injected trusted header is present but malformed.

    Distinct from `MissingTrustedContextException`: this is "the header is
    there but its JSON is broken or missing a required field", not "the
    header is absent altogether".
    """

    def __init__(self, header_name: str, reason: str):
        super().__init__(
            ErrorCode.INVALID_TRUSTED_CONTEXT,
            message=f"Header '{header_name}' không hợp lệ: {reason}",
        )


class InsufficientDocumentPermissionException(UniSageException):
    """Exception raised when the caller lacks DOCUMENT_ALL/DOCUMENT_CREATE permission."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.FORBIDDEN_DOCUMENT_PERMISSION)


class DepartmentAccessDeniedException(UniSageException):
    """Exception raised when the caller isn't granted access to the requested department.

    Covers both "the department isn't in the caller's department_access at
    all" and "it is, but the requested access_level exceeds the level
    granted for that department".
    """

    def __init__(self, department_id: str) -> None:
        super().__init__(
            ErrorCode.FORBIDDEN_DEPARTMENT_ACCESS,
            message=f"Bạn không có quyền truy cập phòng ban '{department_id}'.",
        )


class LLMProviderException(UniSageException):
    """Exception raised when LLM provider API fails."""

    def __init__(self, message: str = ErrorCode.LLM_PROVIDER_ERROR.message):
        super().__init__(ErrorCode.LLM_PROVIDER_ERROR, message=message)


class DocumentChunksNotFoundException(UniSageException):
    """Exception raised when a document has no chunking draft yet."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.DOCUMENT_CHUNKS_NOT_FOUND,
            message=f"Tài liệu '{document_id}' chưa được chia đoạn.",
        )


class IngestionJobNotFoundException(UniSageException):
    """Exception raised when no process-log/draft row exists for a document."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.INGESTION_JOB_NOT_FOUND,
            message=f"Không tìm thấy bản nháp nạp liệu cho tài liệu '{document_id}'.",
        )
