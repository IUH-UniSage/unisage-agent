from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    """HTTP request for `POST /chat/stream`.

    `conversation_id` is required - ownership/existence is validated by
    backend-java on the `POST /messages` call this triggers, never by
    Python.
    """

    conversation_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
