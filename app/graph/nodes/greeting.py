"""Greeting detection node - fast path for pure greetings.

A first-turn pure greeting gets a static template, zero LLM tokens.
"First turn" is answered by Java's own message history
(`GET /messages/conversation/{id}` returning `[]`), never a Python-side
counter — Python does not own conversation history.
"""

import re

from app.integrations.backend_java_client import BackendJavaClient

GREETING_TEMPLATE = (
    "Xin chào! Mình là Trợ Lý AI Học Vụ của trường. Mình có thể giúp bạn "
    "tra cứu quy chế học vụ, tính GPA/học phí, hoặc hướng dẫn thủ tục giấy tờ. "
    "Bạn cần hỏi gì hôm nay?"
)

_GREETING_PATTERN = re.compile(
    r"^\s*(xin\s+ch[àa]o|ch[àa]o|hello|hi|hey|alo)\b|"
    r"\b(bot\s+[oơ]i|tr[oợơ]\s*l[yý]\s+[oơ]i|cho\s+m[iì]nh\s+h[oỏ]i)\b",
    re.IGNORECASE,
)


def is_pure_greeting(message: str) -> bool:
    return bool(_GREETING_PATTERN.search(message.strip()))


async def is_first_turn(
    java_client: BackendJavaClient,
    *,
    conversation_id: str,
    authorization: str | None,
) -> bool:
    history = await java_client.get_conversation_messages(
        conversation_id=conversation_id, limit=1, authorization=authorization
    )
    return len(history) == 0


def detect_greeting(message: str, *, first_turn: bool) -> bool:
    """Fast Path activates only on the first turn - a later "chào" is a real
    message that must go through the normal flow, not be swallowed by the
    template."""

    return first_turn and is_pure_greeting(message)
