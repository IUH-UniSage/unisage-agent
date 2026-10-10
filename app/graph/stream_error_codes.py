"""SSE `event: error` codes for chat failures that are NOT an AI-model failure.

An AI-model failure (CHAT, or EMBEDDING during retrieval) is sent with its own
`app.core.errors.llm_failure.FailureReason` as the code. The two codes below cover the
rest. Their messages are fixed sentences rather than anything derived from the
underlying exception, so no provider/key detail can reach the client through them.
"""

from __future__ import annotations

# Anything that isn't a model or vector-store failure (in practice: a bug). The
# exception's class name is appended for an AI admin; students/guests get the plain
# system message (see `app.core.errors.public_errors`).
LLM_STREAM_INTERRUPTED = "LLM_STREAM_INTERRUPTED"

# Retrieval could not reach the vector store (Qdrant).
VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"

MESSAGES: dict[str, str] = {
    LLM_STREAM_INTERRUPTED: "Đã có lỗi xảy ra trong khi tạo câu trả lời. Vui lòng thử lại.",
    VECTOR_STORE_ERROR: (
        "Không truy cập được kho dữ liệu tài liệu (Qdrant) để tìm thông tin. Vui lòng thử lại sau."
    ),
}

RETRYABLE: dict[str, bool] = {
    LLM_STREAM_INTERRUPTED: True,
    VECTOR_STORE_ERROR: True,
}
