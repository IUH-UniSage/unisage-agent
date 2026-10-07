import pytest

from app.graph.nodes.greeting import detect_greeting, is_pure_greeting
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATES, off_topic_reply


@pytest.mark.parametrize(
    "message",
    ["Chào bạn", "xin chào!", "Hello", "hi there", "alo", "Bot ơi cho mình hỏi"],
)
def test_is_pure_greeting_matches_common_greetings(message: str) -> None:
    assert is_pure_greeting(message) is True


def test_is_pure_greeting_rejects_real_question() -> None:
    assert is_pure_greeting("Điều kiện học bổng loại giỏi là gì?") is False


@pytest.mark.parametrize(
    "message",
    [
        "Cho mình hỏi điều kiện tốt nghiệp",
        "Hello, cho mình hỏi GPA",
        "Xin chào, học phí học kỳ này bao nhiêu?",
        "Chào, điều kiện tốt nghiệp là gì?",
        "Alo, học phí bao nhiêu?",
    ],
)
def test_is_pure_greeting_rejects_greeting_glued_to_a_real_question(message: str) -> None:
    assert is_pure_greeting(message) is False


def test_detect_greeting_only_activates_on_first_turn() -> None:
    assert detect_greeting("Chào bạn", first_turn=True) is True
    assert detect_greeting("Chào bạn", first_turn=False) is False


def test_every_off_topic_template_points_back_to_academic_topics() -> None:
    assert len(OFF_TOPIC_TEMPLATES) > 1
    assert all("học bổng" in template for template in OFF_TOPIC_TEMPLATES)


def test_off_topic_reply_varies_across_turns() -> None:
    replies = {off_topic_reply() for _ in range(200)}

    assert replies <= set(OFF_TOPIC_TEMPLATES)
    assert len(replies) > 1
