from app.core.errors.error_codes import ErrorCode


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


class EmptyDocumentTextException(UniSageException):
    """Exception raised when chunking a document yields no text at all.

    Typical for a scanned PDF (every page is an image, no text layer): the
    pipeline has no OCR step, so without this the endpoint would answer 200
    with an empty chunk list and the client could not tell why.
    """

    def __init__(self) -> None:
        super().__init__(ErrorCode.EMPTY_DOCUMENT_TEXT)


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


class ConversationRejectedException(UniSageException):
    """Java rejected `POST /messages` for this conversation_id.

    Covers both "conversation doesn't exist" (404) and "conversation belongs
    to someone else" (403) - the graph must NOT run and no assistant
    placeholder must be created when this is raised (see tasks/plan.md
    conversation_id ownership invariant).
    """

    def __init__(self, status_code: int) -> None:
        error_code = (
            ErrorCode.CONVERSATION_NOT_FOUND
            if status_code == 404
            else ErrorCode.CONVERSATION_ACCESS_DENIED
        )
        super().__init__(error_code)


class UsageLimitExceededException(UniSageException):
    """Java refused `POST /messages` (USER) with 429: the caller's token quota is used up.

    Carries Java's `errors` (`window`, `resetAt`) through untouched so the web can tell the user
    when they may ask again. The graph must not run and no assistant placeholder is created.
    """

    def __init__(self, errors: dict[str, str] | None = None) -> None:
        super().__init__(ErrorCode.USAGE_LIMIT_EXCEEDED, errors=errors)


class BackendJavaUnavailableException(UniSageException):
    """Network-level failure calling backend-java (not an HTTP error response)."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.BACKEND_JAVA_UNAVAILABLE)


class ChunkingConfigException(UniSageException):
    """Raised when a chunking configuration cannot produce a valid chunk at
    all - e.g. `heading_path`/table header text alone consumes (or exceeds)
    the entire `max_tokens`/`chunk_size` budget, leaving no room (or a
    negative one) for any real content.

    Deliberately fail-fast instead of any of the alternatives considered:
    silently shrinking `overlap`, or silently letting a chunk exceed the
    configured hard cap - both would hide a real configuration problem
    rather than surface it. Raised by the chunkers themselves, before
    `validate_chunks` runs.
    """

    def __init__(self, message: str):
        super().__init__(ErrorCode.CHUNKING_CONFIG_INVALID, message=message)


class IngestionJobNotFoundException(UniSageException):
    """Exception raised when no process-log/draft row exists for a document."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.INGESTION_JOB_NOT_FOUND,
            message=f"Không tìm thấy bản nháp nạp liệu cho tài liệu '{document_id}'.",
        )


class ChunkValidationException(UniSageException):
    """Raised by `validate_chunks` when one or more `Chunk`s have
    internally inconsistent fields - e.g. a TABLE chunk missing
    `source_locator.row_count`, or a PDF TEXT chunk missing `page_start`.

    Distinct from `ChunkingConfigException`: this is for a chunk that has
    ALREADY been built but whose fields don't add up (a bug in a chunker, a
    round-trip/mapping problem in DB or Qdrant, or a client tampering with
    chunk fields before `POST /ingestion/embedding`) - not a configuration
    problem detected before any chunk exists. `errors` carries one entry per
    offending chunk (keyed by `str(chunk_index)`) so the response is
    debuggable without needing a stack trace.
    """

    def __init__(self, errors: dict[str, str]):
        super().__init__(
            ErrorCode.CHUNK_VALIDATION_FAILED,
            errors=errors,
        )


class EmbeddingDraftMismatchException(UniSageException):
    """Raised when `POST /ingestion/embedding`'s `department_id`/`object_key`
    doesn't match the stored chunking draft for `document_id`.

    Deliberately distinct from `IngestionJobNotFoundException` (no draft
    exists at all): here a draft exists, but trusting it for this request
    would silently apply another department/object's canonical metadata -
    so this is a mismatch, not a "not found".
    """

    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.EMBEDDING_DRAFT_MISMATCH,
            message=(
                f"Bản nháp của tài liệu '{document_id}' không khớp department_id/object_key "
                "trong yêu cầu này."
            ),
        )


class EmbeddingChunkSetMismatchException(UniSageException):
    """Raised when `POST /ingestion/embedding`'s `chunks` aren't a valid
    subset (by `chunk_index`) of the canonical draft: a duplicate
    `chunk_index`, or one the draft doesn't have. Sending fewer chunks than
    the draft is allowed (the user deleted some before embedding). Any
    other mismatch makes a partial/best-effort merge unsafe, so this
    always rejects the whole request rather than merging what it can.
    """

    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(
            ErrorCode.EMBEDDING_CHUNK_SET_MISMATCH,
            message=(
                f"Danh sách chunk gửi lên cho tài liệu '{document_id}' không khớp bản nháp "
                f"đã lưu: {reason}"
            ),
        )


class EmbeddingDraftLegacyException(UniSageException):
    """Raised when `POST /ingestion/embedding`'s canonical draft was not
    produced by the current chunking logic: `chunking_version` differs from
    `settings.INGEST_CHUNKING_VERSION` (this covers `"legacy"` and any older
    version), or any canonical chunk has `source_type`/`block_index` still
    `None`.

    Deliberate product decision (plan v5): reject with 409 rather than
    attempting to embed with missing/guessed metadata - heading/page/table
    info cannot be reconstructed accurately from old data. Does not touch
    points already embedded in Qdrant from before this change; it only
    blocks creating NEW embeddings from an old draft.
    """

    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.EMBEDDING_DRAFT_LEGACY,
            message=(
                f"Bản nháp của tài liệu '{document_id}' được tạo bằng phiên bản chia đoạn cũ "
                "hoặc thiếu metadata cấu trúc. Vui lòng chia đoạn (chunk) lại tài liệu "
                "trước khi embed."
            ),
        )


class DocumentUnreadableException(UniSageException):
    """The uploaded file's bytes can't be parsed as its extension says (corrupt, renamed,
    password-protected, not UTF-8 text, ...) - a problem with the file, not the server."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            ErrorCode.DOCUMENT_UNREADABLE,
            message=(
                f"Không đọc được nội dung file '{filename}' ({reason}). File có thể bị hỏng, "
                "sai định dạng so với đuôi file, có mật khẩu hoặc không phải UTF-8."
            ),
        )


class StorageUnavailableException(UniSageException):
    """MinIO failed for a reason other than "object not found" (unreachable, access denied,
    missing bucket, ...)."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            ErrorCode.STORAGE_ERROR,
            message=f"Không truy cập được kho lưu trữ file (MinIO): {reason}. Thử lại sau nhé.",
        )


class ClarificationInvalidException(UniSageException):
    """A panel submit failed validation against the stored panel; `errors` maps
    question_id -> reason. The panel stays open."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__(ErrorCode.CLARIFICATION_INVALID, errors=errors)


class ClarificationStaleException(UniSageException):
    """The panel was already answered/cancelled, is being answered by another request,
    or the panel_id is not the open one."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.CLARIFICATION_STALE)


class ClarificationPendingException(UniSageException):
    """A plain message while a panel is open - it must be answered or cancelled first."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.CLARIFICATION_PENDING)


class ClarificationProcessingException(UniSageException):
    """A plain message while the previous panel answer is still being processed."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.CLARIFICATION_PROCESSING)


class RequestTooLargeException(UniSageException):
    def __init__(self) -> None:
        super().__init__(ErrorCode.REQUEST_TOO_LARGE)
