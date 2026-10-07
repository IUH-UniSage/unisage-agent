"""Social chat reply - picked by the message's kind, then at random within it."""

import pytest

from app.graph.nodes.social_chat import (
    GOODBYE_TEMPLATES,
    GREETING_TEMPLATES,
    OTHER_TEMPLATES,
    THANKS_TEMPLATES,
    social_chat_reply,
)


@pytest.mark.parametrize(
    ("message", "templates"),
    [
        ("hi", GREETING_TEMPLATES),
        ("Chào bạn", GREETING_TEMPLATES),
        ("Cảm ơn bạn nhiều", THANKS_TEMPLATES),
        ("ok thanks", THANKS_TEMPLATES),
        ("cảm ơn, tạm biệt nhé", THANKS_TEMPLATES),
        ("chào tạm biệt", GOODBYE_TEMPLATES),
        ("bye bot", GOODBYE_TEMPLATES),
        ("Bạn là ai vậy?", OTHER_TEMPLATES),
    ],
)
def test_reply_matches_the_kind_of_message(message: str, templates: list[str]) -> None:
    assert social_chat_reply(message) in templates


def test_a_later_turn_hi_is_not_answered_as_thanks() -> None:
    assert not {social_chat_reply("hi") for _ in range(50)} & set(THANKS_TEMPLATES)


def test_reply_varies_across_turns() -> None:
    assert len({social_chat_reply("Cảm ơn bạn") for _ in range(200)}) > 1
