from pydantic import BaseModel, Field

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
    message: str = Field(min_length=1, max_length=MESSAGE_MAX_CHARS)
