from enum import Enum


class ErrorCode(Enum):
    """Mirrors backend-java's `ErrorCode.java`: each member carries its own HTTP
    status, a stable numeric code, and a default message - so the FastAPI
    exception handlers in `app/main.py` can build the exact same
    `{code, message, data, errors}` envelope Java's `ApiResponse`/
    `GlobalExceptionHandler` produce, letting the frontend parse both
    backends' responses with one shared utility.

    Numeric codes intentionally stay in the 4xxx/5xxx ranges, which Java's
    own ErrorCode enum never uses (it spans 1xxx/2xxx/9999) - so the two
    services' codes can be merged into one lookup table on the frontend
    without collisions.
    """

    # 400x Client Errors
    INVALID_QUERY = (400, 4001, "Câu hỏi không hợp lệ, kiểm tra lại giúp mình.")
    UNAUTHORIZED = (403, 4002, "Bạn không có quyền dùng chức năng này.")
    MISSING_TRUSTED_CONTEXT = (400, 4003, "Phiên đăng nhập không hợp lệ, vui lòng đăng nhập lại.")
    UNSUPPORTED_FILE_TYPE = (415, 4004, "Định dạng file này chưa được hỗ trợ nạp liệu.")
    STRATEGY_FILE_TYPE_MISMATCH = (
        422,
        4005,
        "Chiến lược chia đoạn này không áp dụng được cho loại file này.",
    )
    INVALID_TRUSTED_CONTEXT = (400, 4006, "Phiên đăng nhập không hợp lệ, vui lòng đăng nhập lại.")
    FORBIDDEN_DOCUMENT_PERMISSION = (403, 4007, "Bạn không có quyền xử lý tài liệu.")
    FORBIDDEN_DEPARTMENT_ACCESS = (
        403,
        4008,
        "Bạn không có quyền truy cập phòng ban của tài liệu này.",
    )
    VALIDATION_ERROR = (400, 4009, "Thông tin nhập chưa hợp lệ, kiểm tra lại giúp mình.")

    # 404x Not Found Errors
    OBJECT_NOT_FOUND = (404, 4041, "Không tìm thấy file gốc của tài liệu này.")
    DOCUMENT_CHUNKS_NOT_FOUND = (404, 4042, "Tài liệu này chưa được chia đoạn.")
    INGESTION_JOB_NOT_FOUND = (404, 4043, "Không tìm thấy bản nháp nạp liệu cho tài liệu này.")

    # 500x Server & LLM Errors
    INTERNAL_ERROR = (500, 5000, "Có lỗi xảy ra, bạn thử lại sau nhé.")
    LLM_TIMEOUT = (504, 5001, "Hệ thống AI phản hồi quá lâu, thử lại sau nhé.")
    LLM_PROVIDER_ERROR = (502, 5002, "Hệ thống AI đang gặp sự cố, thử lại sau nhé.")
    DATABASE_ERROR = (500, 5003, "Có lỗi xảy ra, bạn thử lại sau nhé.")

    def __init__(self, http_status: int, code: int, message: str) -> None:
        self.http_status = http_status
        self.code = code
        self.message = message
