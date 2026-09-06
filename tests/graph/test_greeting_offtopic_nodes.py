import httpx
import pytest

from app.graph.nodes.greeting import detect_greeting, is_first_turn, is_pure_greeting
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATE
from app.integrations.backend_java_client import BackendJavaClient


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


@pytest.mark.asyncio
async def test_is_first_turn_true_when_java_returns_empty_history() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))

    assert await is_first_turn(client, conversation_id="conv-1", authorization=None) is True


@pytest.mark.asyncio
async def test_is_first_turn_false_when_java_returns_history() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "m1"}])

    client = BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))

    assert await is_first_turn(client, conversation_id="conv-1", authorization=None) is False


def test_off_topic_template_lists_examples() -> None:
    assert "học bổng" in OFF_TOPIC_TEMPLATE
