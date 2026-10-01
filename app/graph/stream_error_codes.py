"""SSE `event: error` codes for the chat stream.

Each code's message is a fixed, friendly Vietnamese sentence rather than
anything derived from the underlying provider exception - the safest way to
guarantee `safe_error_message`'s whole purpose (never let provider/key
details reach the client) holds even if a future provider SDK's exception
text changes shape. The raw exception itself is still only ever turned into
text via `app.core.security.redaction.safe_error_message` wherever it needs to be
logged or reported to backend-java (see `app.core.registry.model_router.record_failure`)
- never interpolated directly into anything client-facing.
"""

from __future__ import annotations

# Failure happened AFTER at least one chunk of this response already reached
# the client - no retry is possible without mixing two models' output into
# one response, so this is a terminal "tell the client and stop" code.
LLM_STREAM_INTERRUPTED = "LLM_STREAM_INTERRUPTED"

# Every CHAT credential was cooling down/excluded before any chunk streamed
# (`app.core.registry.model_router.NoAvailableCredentialError`) - nothing was sent yet,
# so retrying the whole request later may succeed once a credential recovers.
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

# A BLOCK-action budget (SYSTEM/PURPOSE/PROVIDER) was already at or over its
# limit - retrying immediately would hit the same wall; the budget resets on
# its own period boundary (daily/monthly).
BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

# A THROTTLE-action budget's soft limit was reached and its concurrency cap is
# currently full - unlike BUDGET_EXCEEDED, this can clear itself within
# seconds as in-flight requests finish, so it is safe to retry shortly after.
BUDGET_THROTTLED = "BUDGET_THROTTLED"

# Retrieval could not reach the vector store (Qdrant) - not a model failure.
VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"

# Every AI-model failure (CHAT/EMBEDDING/EXTRACTION) uses its own, more specific
# code and message from `app.core.errors.llm_failure.describe_llm_failure` (its
# `reason`, e.g. "LLM_AUTH_FAILED") - the table below only covers the rest.

MESSAGES: dict[str, str] = {
    LLM_STREAM_INTERRUPTED: ("Đã có lỗi xảy ra trong khi tạo câu trả lời. Vui lòng thử lại."),
    LLM_UNAVAILABLE: (
        "Hệ thống đang tạm thời không thể xử lý yêu cầu này. Vui lòng thử lại sau ít phút."
    ),
    BUDGET_EXCEEDED: ("Hệ thống đã đạt giới hạn sử dụng. Vui lòng thử lại sau."),
    BUDGET_THROTTLED: ("Hệ thống đang xử lý nhiều yêu cầu cùng lúc. Vui lòng thử lại sau ít giây."),
    VECTOR_STORE_ERROR: (
        "Không truy cập được kho dữ liệu tài liệu (Qdrant) để tìm thông tin. Vui lòng thử lại sau."
    ),
}

RETRYABLE: dict[str, bool] = {
    LLM_STREAM_INTERRUPTED: True,
    LLM_UNAVAILABLE: False,
    BUDGET_EXCEEDED: False,
    BUDGET_THROTTLED: True,
    VECTOR_STORE_ERROR: True,
}
