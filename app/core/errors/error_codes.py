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
    EMPTY_DOCUMENT_TEXT = (
        422,
        4221,
        "Không trích xuất được văn bản từ tài liệu này (có thể là bản scan hoặc chỉ có ảnh). "
        "Hãy dùng bản có thể chọn chữ hoặc chạy OCR trước khi nạp.",
    )
    DOCUMENT_UNREADABLE = (
        422,
        4222,
        "Không đọc được nội dung file (file hỏng, sai định dạng so với đuôi file, có mật khẩu "
        "hoặc không phải UTF-8). Hãy kiểm tra lại file rồi tải lên lại.",
    )
    CHUNKING_CONFIG_INVALID = (
        422,
        4010,
        "Cấu hình chia đoạn không hợp lệ với tài liệu này, thử tăng kích thước đoạn.",
    )
    CHUNK_VALIDATION_FAILED = (
        400,
        4011,
        "Dữ liệu chia đoạn không hợp lệ, vui lòng chia đoạn lại.",
    )
    EMBEDDING_DRAFT_MISMATCH = (
        400,
        4012,
        "Bản nháp nạp liệu không khớp với yêu cầu này, vui lòng chia đoạn lại.",
    )
    EMBEDDING_CHUNK_SET_MISMATCH = (
        400,
        4013,
        "Danh sách chunk gửi lên không khớp với bản nháp đã lưu, vui lòng chia đoạn lại.",
    )
    EMBEDDING_DRAFT_LEGACY = (
        409,
        4014,
        "Bản nháp này được tạo bằng phiên bản chia đoạn cũ hoặc thiếu metadata cấu trúc, "
        "vui lòng chia đoạn lại trước khi embed.",
    )
    EMBEDDING_IDENTITY_MISMATCH = (
        409,
        4015,
        "Mô hình Embedding hiện tại không khớp với dữ liệu đã được embedding từ trước trong hệ "
        "thống (có thể do đổi mô hình/nhà cung cấp Embedding mà chưa re-index). Liên hệ quản trị "
        "viên để đăng ký lại danh tính embedding hoặc embedding lại toàn bộ dữ liệu trước khi "
        "tiếp tục.",
    )

    # 404x Not Found Errors
    OBJECT_NOT_FOUND = (404, 4041, "Không tìm thấy file gốc của tài liệu này.")
    DOCUMENT_CHUNKS_NOT_FOUND = (404, 4042, "Tài liệu này chưa được chia đoạn.")
    INGESTION_JOB_NOT_FOUND = (404, 4043, "Không tìm thấy bản nháp nạp liệu cho tài liệu này.")
    CONVERSATION_NOT_FOUND = (404, 4044, "Không tìm thấy cuộc hội thoại này.")

    # 403x Forbidden (chat-specific; distinct code from FORBIDDEN_DEPARTMENT_ACCESS)
    CONVERSATION_ACCESS_DENIED = (403, 4031, "Bạn không có quyền truy cập cuộc hội thoại này.")

    # 429 Usage limit. The code 2130 is backend-java's own USAGE_LIMIT_EXCEEDED, passed through
    # unchanged (this is the one 2xxx code the agent re-emits) so the web sees the same code and
    # message whichever service answered.
    USAGE_LIMIT_EXCEEDED = (
        429,
        2130,
        "Bạn đã dùng hết hạn mức sử dụng. Vui lòng quay lại sau thời điểm được thông báo.",
    )

    # 500x Server & LLM Errors
    INTERNAL_ERROR = (500, 5000, "Có lỗi xảy ra, bạn thử lại sau nhé.")
    LLM_TIMEOUT = (504, 5001, "Hệ thống AI phản hồi quá lâu, thử lại sau nhé.")
    LLM_PROVIDER_ERROR = (502, 5002, "Hệ thống AI đang gặp sự cố, thử lại sau nhé.")
    DATABASE_ERROR = (
        503,
        5003,
        "Không truy cập được cơ sở dữ liệu của dịch vụ AI, thử lại sau nhé.",
    )
    BACKEND_JAVA_UNAVAILABLE = (
        502,
        5004,
        "Không kết nối được hệ thống quản lý hội thoại, thử lại sau nhé.",
    )
    EMBEDDING_JOB_FAILED = (
        502,
        5005,
        "Nạp liệu (embedding) thất bại — kiểm tra Cấu hình AI (mô hình Embedding/Extraction) "
        "rồi thử nạp lại.",
    )
    EMBEDDING_PROVIDER_ERROR = (
        502,
        5006,
        "Nhà cung cấp Embedding đang gặp sự cố (sai cấu hình, hết hạn mức, hoặc lỗi kết nối). "
        "Kiểm tra Cấu hình AI rồi thử lại.",
    )
    # One code per distinguishable AI-model failure cause (see
    # `app.core.errors.llm_failure.describe_llm_failure`, which also builds the
    # purpose-specific message actually sent - these defaults are only a fallback).
    LLM_NOT_CONFIGURED = (
        503,
        5007,
        "Chưa cấu hình mô hình AI cho chức năng này. Thêm credential trong trang Cấu hình AI.",
    )
    LLM_AUTH_FAILED = (
        502,
        5008,
        "API key của mô hình AI không hợp lệ hoặc không có quyền. Kiểm tra trang Cấu hình AI.",
    )
    LLM_QUOTA_EXHAUSTED = (
        502,
        5009,
        "Tài khoản nhà cung cấp mô hình AI đã hết hạn mức/credit. Kiểm tra trang Cấu hình AI.",
    )
    LLM_RATE_LIMITED = (
        503,
        5010,
        "Nhà cung cấp mô hình AI đang giới hạn tốc độ gọi, thử lại sau ít phút.",
    )
    LLM_MODEL_NOT_FOUND = (
        502,
        5011,
        "Nhà cung cấp không tìm thấy mô hình AI đã cấu hình (sai tên model hoặc base URL).",
    )
    LLM_REQUEST_REJECTED = (
        502,
        5012,
        "Nhà cung cấp mô hình AI từ chối yêu cầu (sai tham số, xung đột hoặc nội dung quá dài).",
    )
    LLM_CONNECTION_ERROR = (
        502,
        5013,
        "Không kết nối được tới nhà cung cấp mô hình AI. Kiểm tra base URL và mạng.",
    )
    LLM_BUDGET_EXCEEDED = (
        429,
        5014,
        "Đã đạt giới hạn ngân sách sử dụng mô hình AI. Vui lòng thử lại sau.",
    )
    LLM_PROVIDER_UNSUPPORTED = (
        502,
        5015,
        "Nhà cung cấp/địa chỉ của mô hình AI đã cấu hình không được hệ thống hỗ trợ.",
    )
    LLM_ALL_CREDENTIALS_SUSPENDED = (
        503,
        5016,
        "Mọi credential của mô hình AI đang bị tạm ngưng do lỗi gần đây. "
        "Kiểm tra trang Cấu hình AI.",
    )
    VECTOR_STORE_ERROR = (
        502,
        5017,
        "Không truy cập được kho vector (Qdrant), thử lại sau nhé.",
    )
    STORAGE_ERROR = (
        502,
        5018,
        "Không truy cập được kho lưu trữ file (MinIO), thử lại sau nhé.",
    )
    TASK_QUEUE_UNAVAILABLE = (
        503,
        5019,
        "Không kết nối được hàng đợi xử lý nền (Redis/Celery), thử lại sau nhé.",
    )

    def __init__(self, http_status: int, code: int, message: str) -> None:
        self.http_status = http_status
        self.code = code
        self.message = message
