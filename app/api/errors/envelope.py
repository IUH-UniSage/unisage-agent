"""Building blocks shared by every exception handler: the response envelope, the short
reference code logged with each failure, and the chat-specific "how much detail may this
caller see" decision."""

import uuid
from typing import Any

from fastapi import Request

from app.core.errors.llm_failure import LLMFailure
from app.core.errors.public_errors import (
    can_see_ai_details,
    permissions_from_header,
    public_chat_message,
    short_reference,
)

_CHAT_PATH_PREFIX = "/api/v1/chat"


def error_content(code: int, message: str, errors: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the `{code, message, errors}` envelope - `data` and `errors` are
    omitted when absent, matching Java's `@JsonInclude(NON_NULL)` on
    `ApiResponse`."""

    content: dict[str, Any] = {"code": code, "message": message}
    if errors:
        content["errors"] = errors
    return content


def new_reference() -> str:
    """A fresh short code to put in both the log line and the response, so a failure the
    user reports can be found in the logs."""

    return short_reference(uuid.uuid4().hex)


def public_chat_error(
    request: Request, failure: LLMFailure | None, *, reference: str
) -> tuple[str, dict[str, str]] | None:
    """`(message, errors)` to send instead of the technical detail when the caller is a
    chat user (student/guest) - see `app.core.errors.public_errors`. None when the detail
    should be sent as-is: any non-chat route (ingest, AI admin: their users fix these
    problems), or a chat caller holding an AI-configuration permission."""

    if not request.url.path.startswith(_CHAT_PATH_PREFIX):
        return None
    permissions = permissions_from_header(request.headers.get("x-user-permissions"))
    if can_see_ai_details(permissions):
        return None
    message, _retryable = public_chat_message(failure, reference=reference)
    return message, {"reference": reference}
