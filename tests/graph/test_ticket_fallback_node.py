from collections.abc import Awaitable, Callable, Sequence

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.ticket_fallback import build_ticket_fallback_agent, run_ticket_fallback
from app.rag.prompting.citations import source_links_block
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult

_NOTICE = WebSearchResult(
    title="Thông báo v/v nghỉ lễ Quốc khánh năm 2025",
    url="https://qn.iuh.edu.vn/thong-bao/thong-bao-v-v-nghi-le-quoc-khanh-nam-2025",
    content="tb-846-2025 Tải xuống",
    score=0.65,
)


def _sink(target: list[str]) -> Callable[[str], Awaitable[None]]:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


def _prompt_capturing_model(seen: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        last_part = messages[-1].parts[-1]
        seen.append(getattr(last_part, "content", ""))
        yield "Chưa tìm thấy quy định phù hợp."

    return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
async def test_streams_the_model_output_token_by_token(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    agent = build_ticket_fallback_agent(mock_streaming_llm_model(["Chưa tìm thấy ", "quy định."]))
    seen: list[str] = []

    text = await run_ticket_fallback(
        agent,
        "Câu hỏi không tìm thấy trong quy chế nào cả",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink(seen),
    )

    assert text == "Chưa tìm thấy quy định."
    assert "".join(seen) == text


@pytest.mark.asyncio
async def test_prompt_carries_no_retrieved_chunk_and_has_the_security_block() -> None:
    """The model must not be able to cite a regulation it has no source for -
    the prompt should never contain retrieved-chunk data, only the static
    fallback rules, the security block, and the user's question."""

    seen_prompts: list[str] = []
    agent = build_ticket_fallback_agent(_prompt_capturing_model(seen_prompts))

    await run_ticket_fallback(
        agent,
        "Câu hỏi lạ chưa có quy định",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink([]),
    )

    (prompt,) = seen_prompts
    assert "UNIQUE_CHUNK_MARKER_MUST_NOT_APPEAR" not in prompt
    assert "{prepared_context}" not in prompt
    assert "Quy Tắc Bảo Mật" in prompt


@pytest.mark.asyncio
async def test_a_link_in_the_reply_is_exact_and_only_the_other_pages_are_listed_after_it(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """The 14-10-2026 turn: the model retyped the notice's URL as "quoc-khánh" (a 404),
    then the same notice was listed again under the reply."""

    retyped = _NOTICE.url.replace("quoc-khanh", "quoc-khánh")
    notice_2026 = WebSearchResult(
        title="TB-957-2026.pdf",
        url="https://iuh.edu.vn/upload/files/2026/08/TB-957-2026.pdf",
        content="nghỉ lễ Quốc khánh 2026",
        score=0.4,
    )
    agent = build_ticket_fallback_agent(
        mock_streaming_llm_model([f"Bạn xem [Thông báo năm 2025]({retyped}) nhé."])
    )
    seen: list[str] = []

    text = await run_ticket_fallback(
        agent,
        "Lịch nghỉ lễ Quốc khánh 2/9 năm 2025?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        web_sources=[_NOTICE, notice_2026],
        token_sink=_sink(seen),
    )

    assert text == (
        f"Bạn xem [Thông báo năm 2025]({_NOTICE.url}) nhé.\n\n"
        "Văn bản liên quan trên website của Trường:\n"
        f"- [TB-957-2026.pdf]({notice_2026.url})"
    )
    assert "".join(seen).endswith(f"- [TB-957-2026.pdf]({notice_2026.url})")


def test_nothing_is_listed_when_the_reply_links_every_page() -> None:
    assert source_links_block([_NOTICE], f"Xem [thông báo]({_NOTICE.url}).") == ""
    assert source_links_block([], "Mình chưa tìm thấy.") == ""
    assert "(website Phân hiệu Quảng Ngãi)" in source_links_block([_NOTICE])
