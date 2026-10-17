"""Social chat - streamed by the LLM; the template fallback is picked by the
message's kind, then at random within it."""

from collections.abc import Awaitable, Callable

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.social_chat import (
    GOODBYE_TEMPLATES,
    GREETING_TEMPLATES,
    IDENTITY_TEMPLATES,
    OTHER_TEMPLATES,
    THANKS_TEMPLATES,
    build_social_chat_agent,
    run_social_chat,
    social_chat_reply,
)
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import make_streaming_llm_model, make_streaming_llm_model_that_fails_after


def _sink(target: list[str]) -> Callable[[str], Awaitable[None]]:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


def _prompt_capturing_model(seen: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        seen.append(getattr(messages[-1].parts[-1], "content", ""))
        yield "Mình biết bạn học ngành CNTT. Bạn muốn hỏi gì về học phí không?"

    return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
async def test_streams_the_model_reply() -> None:
    agent = build_social_chat_agent(make_streaming_llm_model(["Không có gì ", "đâu bạn!"]))
    tokens: list[str] = []

    text = await run_social_chat(
        agent,
        "Cảm ơn bạn",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink(tokens),
    )

    assert text == "Không có gì đâu bạn!"
    assert "".join(tokens) == text


@pytest.mark.asyncio
async def test_prompt_carries_the_profile_the_rules_and_the_message() -> None:
    prompts: list[str] = []

    await run_social_chat(
        build_social_chat_agent(_prompt_capturing_model(prompts)),
        "Bạn biết thông tin gì về tôi?",
        security=AcademicSecurityContext(),
        confirmed_metadata={"nganh": "Công nghệ thông tin", "khoa_nhap_hoc": "2022"},
        token_sink=_sink([]),
    )

    (prompt,) = prompts
    assert "- nganh: Công nghệ thông tin" in prompt
    assert "- khoa_nhap_hoc: 2022 (khóa K18)" in prompt
    assert "Quy Tắc Bảo Mật" in prompt
    assert "câu hỏi dẫn dắt" in prompt
    assert prompt.rstrip().endswith("Tin nhắn của người dùng: Bạn biết thông tin gì về tôi?")
    assert "<academic_context>\n" not in prompt


@pytest.mark.asyncio
async def test_falls_back_to_a_template_when_the_model_fails_before_streaming() -> None:
    model = make_streaming_llm_model_that_fails_after([], RuntimeError("provider down"))
    tokens: list[str] = []

    text = await run_social_chat(
        build_social_chat_agent(model),
        "Cảm ơn bạn",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink(tokens),
    )

    assert text in THANKS_TEMPLATES
    assert tokens == [text]


@pytest.mark.asyncio
async def test_a_failure_after_streaming_started_is_raised() -> None:
    model = make_streaming_llm_model_that_fails_after(["Không có "], RuntimeError("dropped"))

    with pytest.raises(RuntimeError, match="dropped"):
        await run_social_chat(
            build_social_chat_agent(model),
            "Cảm ơn bạn",
            security=AcademicSecurityContext(),
            confirmed_metadata={},
            token_sink=_sink([]),
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
        ("Bạn là ai vậy?", IDENTITY_TEMPLATES),
        ("ban la ai", IDENTITY_TEMPLATES),
        ("Chào bạn, bạn giúp được gì cho mình?", IDENTITY_TEMPLATES),
        ("who are you", IDENTITY_TEMPLATES),
        ("Ok hiểu rồi", OTHER_TEMPLATES),
    ],
)
def test_reply_matches_the_kind_of_message(message: str, templates: list[str]) -> None:
    assert social_chat_reply(message) in templates


def test_a_later_turn_hi_is_not_answered_as_thanks() -> None:
    assert not {social_chat_reply("hi") for _ in range(50)} & set(THANKS_TEMPLATES)


def test_reply_varies_across_turns() -> None:
    assert len({social_chat_reply("Cảm ơn bạn") for _ in range(200)}) > 1


def test_identity_reply_describes_the_assistant_like_the_system_prompt() -> None:
    for reply in IDENTITY_TEMPLATES:
        assert reply.startswith("Mình là Trợ lý AI Học vụ của Nhà trường")
        assert "quy chế đào tạo, học phí, học bổng" in reply
