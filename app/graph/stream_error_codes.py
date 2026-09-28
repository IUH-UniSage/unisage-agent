"""SSE `event: error` codes for the chat stream - plan.md "SSE error contract".

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

# Reserved for todo.md Task 20 (running-total budget check) - no code in this
# task raises it; defined here so the wire contract's third code already has
# a home when that task lands.
SYSTEM_BUDGET_EXHAUSTED = "SYSTEM_BUDGET_EXHAUSTED"

MESSAGES: dict[str, str] = {
    LLM_STREAM_INTERRUPTED: ("Đã có lỗi xảy ra trong khi tạo câu trả lời. Vui lòng thử lại."),
    LLM_UNAVAILABLE: (
        "Hệ thống đang tạm thời không thể xử lý yêu cầu này. Vui lòng thử lại sau ít phút."
    ),
    SYSTEM_BUDGET_EXHAUSTED: ("Hệ thống đã đạt giới hạn sử dụng. Vui lòng thử lại sau."),
}

RETRYABLE: dict[str, bool] = {
    LLM_STREAM_INTERRUPTED: True,
    LLM_UNAVAILABLE: False,
    SYSTEM_BUDGET_EXHAUSTED: False,
}
