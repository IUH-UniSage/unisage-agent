"""What a chat user (student, guest) is told when something fails.

`describe_llm_failure` produces a precise, technical explanation (which model, which HTTP
status, "API key không hợp lệ", "Qdrant", ...) - right for the ingest wizard and the AI
admin pages, wrong for the chat window: a student can't act on it, and it discloses the
infrastructure to anonymous guests. Chat therefore shows one of a few plain categories,
chosen by what the USER can do about it, plus a short reference code that matches the
server log line holding the full detail.

A caller holding an AI-configuration permission (an admin testing the chat) still gets
the detailed message - they are the ones who can fix it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from app.core.errors.llm_failure import FailureReason, LLMFailure

# Anyone allowed to see the AI model configuration may see why a model call failed.
AI_ADMIN_PERMISSIONS = frozenset({"CHAT_MODEL_ALL", "CHAT_MODEL_READ"})

# Things the user can fix by changing what they ask.
_USER_FIXABLE_MESSAGES = {
    FailureReason.LLM_CONTENT_FILTERED: (
        "Câu hỏi này bị bộ lọc an toàn của trợ lý AI chặn, bạn thử diễn đạt lại nhé."
    ),
    FailureReason.LLM_INPUT_TOO_LARGE: (
        "Câu hỏi hoặc cuộc hội thoại đã quá dài, bạn thử rút ngắn câu hỏi hoặc mở cuộc "
        "hội thoại mới nhé."
    ),
}
_BUDGET_MESSAGES = {
    FailureReason.BUDGET_EXCEEDED: "Hệ thống đã đạt giới hạn sử dụng. Vui lòng thử lại sau.",
    FailureReason.BUDGET_THROTTLED: (
        "Hệ thống đang xử lý nhiều yêu cầu cùng lúc. Vui lòng thử lại sau ít giây."
    ),
}
TEMPORARY_MESSAGE = "Trợ lý AI đang bận hoặc tạm thời gián đoạn, bạn thử lại sau ít phút nhé."
SYSTEM_MESSAGE = (
    "Trợ lý AI đang tạm ngưng do sự cố hệ thống. Sự cố đã được ghi nhận, bạn vui lòng quay lại sau."
)


def permissions_from_header(raw: str | None) -> list[str]:
    """The gateway's `X-User-Permissions` JSON array, or [] if absent/malformed - only used
    to decide how much detail to show, so a bad header just means "no detail"."""

    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [permission for permission in parsed if isinstance(permission, str)]


def can_see_ai_details(permissions: Iterable[str]) -> bool:
    return not AI_ADMIN_PERMISSIONS.isdisjoint(permissions)


def public_chat_message(failure: LLMFailure | None, *, reference: str | None) -> tuple[str, bool]:
    """`(message, retryable)` to show a chat user. `failure` is None for a failure that
    isn't an AI-model call at all (database, Qdrant, backend-java, a bug) - always a
    system problem from the user's point of view."""

    if failure is None:
        message, retryable = SYSTEM_MESSAGE, True
    elif failure.reason in _USER_FIXABLE_MESSAGES:
        message, retryable = _USER_FIXABLE_MESSAGES[failure.reason], False
    elif failure.reason in _BUDGET_MESSAGES:
        message, retryable = _BUDGET_MESSAGES[failure.reason], failure.retryable
    elif failure.retryable:
        message, retryable = TEMPORARY_MESSAGE, True
    else:
        message, retryable = SYSTEM_MESSAGE, False
    if reference:
        message = f"{message} (Mã tham chiếu: {reference})"
    return message, retryable


def short_reference(request_id: str) -> str:
    """The first 8 hex chars of a request id - short enough to read out, and what every
    log line for that request can be searched by."""

    return request_id.replace("-", "")[:8]
