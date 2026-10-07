"""Greeting detection node - fast path for pure greetings.

A first-turn pure greeting gets a static template, zero LLM tokens.
"First turn" is answered by Java (`POST /messages/turn`'s `firstTurn`),
never a Python-side counter — Python does not own conversation history.
"""

import re

GREETING_TEMPLATE = (
    "Xin chào! Mình là Trợ Lý AI Học Vụ của trường. Mình có thể giúp bạn "
    "tra cứu quy chế học vụ, tính GPA/học phí, hoặc hướng dẫn thủ tục giấy tờ. "
    "Bạn cần hỏi gì hôm nay?"
)

# Greeting/lead-in phrases the whole message may consist of. Matched with
# \b boundaries and stripped out (possibly more than once, e.g. "Bot ơi cho
# mình hỏi") - whatever's left over must be nothing but short filler/
# punctuation for the message to count as a PURE greeting. This is what
# tells "Hello" and "Bot ơi cho mình hỏi" (no question attached yet) apart
# from "Hello, cho mình hỏi GPA" or "Cho mình hỏi điều kiện tốt nghiệp" -
# a real question glued onto a greeting must fall through to the normal
# flow, not be swallowed by the static template.
_GREETING_TOKENS = re.compile(
    r"\b(xin\s+ch[àa]o|ch[àa]o|hello|hi|hey|alo|"
    r"bot\s+[oơ]i|tr[oợơ]\s*l[yý]\s+[oơ]i|cho\s+m[iì]nh\s+h[oỏ]i)\b",
    re.IGNORECASE,
)
_FILLER_ONLY_PATTERN = re.compile(
    r"^[\s,.!?~]*(bạn|there|nhé|ơi)?[\s,.!?~]*$",
    re.IGNORECASE,
)


def is_pure_greeting(message: str) -> bool:
    remainder = _GREETING_TOKENS.sub("", message.strip())
    return bool(_FILLER_ONLY_PATTERN.fullmatch(remainder))


def detect_greeting(message: str, *, first_turn: bool) -> bool:
    """Fast Path activates only on the first turn - a later "chào" is a real
    message that must go through the normal flow, not be swallowed by the
    template."""

    return first_turn and is_pure_greeting(message)
