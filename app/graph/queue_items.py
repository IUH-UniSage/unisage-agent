"""Typed items pushed onto the queue between `run_and_persist` (producer) and
`_sse_token_generator` (consumer) - `app/graph/streaming_session.py` /
`app/api/v1/chat.py`.

Replaces the old `str | None` scheme (a token string, or `None` as the
end-of-stream sentinel) with a small closed set of item types, so an error
can be represented on the same queue as a token or the end-of-stream marker
instead of being smuggled through a side channel. Per plan.md "SSE error
contract": `run_and_persist` puts at most one `ErrorItem`, always immediately
before the single `DoneItem` that ends every stream.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenItem:
    """One text delta forwarded from `stream_agent_text()`."""

    text: str


@dataclass(frozen=True)
class ErrorItem:
    """One `event: error` - `code`/`message`/`retryable` map 1:1 onto the SSE
    payload's JSON fields. `message` must already be the redacted, friendly
    Vietnamese text safe to hand to the client - never a raw provider
    exception string."""

    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class DoneItem:
    """End-of-stream sentinel - always the last item on the queue, put in
    `run_and_persist`'s outer `finally` no matter what happened before it."""


QueueItem = TokenItem | ErrorItem | DoneItem
