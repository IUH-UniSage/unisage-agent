from pydantic import BaseModel, Field, model_validator

from app.schemas.clarification import ClarificationAction

MESSAGE_MAX_CHARS = 2000


class ChatStreamRequest(BaseModel):
    """HTTP request for `POST /chat/stream`.

    `conversation_id` is required - ownership/existence is validated by
    backend-java on the `POST /messages` call this triggers, never by
    Python.

    `message`'s `max_length` is the ONLY length limit on a question: anything
    longer is rejected here (400, code 4009 `VALIDATION_ERROR`, `errors.message`)
    before the handler runs, and nothing downstream truncates it.
    """

    conversation_id: str = Field(min_length=1, max_length=100)
    message: str | None = Field(default=None, min_length=1, max_length=MESSAGE_MAX_CHARS)
    # Answering or cancelling an open clarification panel (contracts/chat-sse.md §1).
    clarification: ClarificationAction | None = None

    @model_validator(mode="after")
    def _message_or_clarification(self) -> "ChatStreamRequest":
        if (self.message is None) == (self.clarification is None):
            raise ValueError("send exactly one of `message` or `clarification`")
        return self
