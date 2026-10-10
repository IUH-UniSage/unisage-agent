from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.graph.nodes.generation_synthesis import (
    build_generation_agent,
    run_generation_synthesis,
)
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_TRACE = GraphTrace(conversation_id="conv-1", message_id="msg-1", user_id=None, client_ip=None)


@pytest.mark.asyncio
async def test_streams_full_response_and_no_ask_form_when_no_json_block(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    agent = build_generation_agent(
        mock_streaming_llm_model(["Câu trả lời cuối cùng ", "kèm trích dẫn [1]."])
    )
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    result = await run_generation_synthesis(
        agent,
        user_query="Hạn đăng ký học kỳ này là khi nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[RetrievedChunk(chunk_id="c1", content="nội dung", source="s", score=0.9)],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.response_text == "Câu trả lời cuối cùng kèm trích dẫn [1]."
    assert result.ask_forms == ()
    assert "".join(received) == result.response_text


@pytest.mark.asyncio
async def test_captures_the_trailing_ask_form_block(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    response_with_json = (
        "Quy định miễn giảm khác nhau tuỳ hệ đào tạo.\n\n"
        "```json\n"
        '{"type": "ask_user_form", "fields": [{"field": "training_type", '
        '"options": [{"id": "chinh_quy"}, {"id": "lien_thong"}]}]}\n'
        "```"
    )
    agent = build_generation_agent(mock_streaming_llm_model([response_with_json]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="...",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert len(result.ask_forms) == 1
    assert [f["field"] for f in result.ask_forms[0]["fields"]] == ["training_type"]


@pytest.mark.asyncio
async def test_repairs_missing_ask_form_when_prose_asks_for_missing_attribute(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
) -> None:
    """Known model failure mode: a clarification request in prose, without the
    mandatory JSON block. The repair follow-up call (second scripted response)
    supplies it for the graph - it must never reach the client or `response_text`
    (UNISAGE-99: the panel replaces the in-text form)."""

    prose_without_json = "Bạn vui lòng cho biết ngành học của bạn để mình tra học phí nhé!"
    repair_json = (
        '```json\n{"type": "ask_user_form", "fields": [{"field": "nganh", '
        '"options": [{"id": "cntt"}, {"id": "logistics"}]}]}\n```'
    )
    agent = build_generation_agent(
        mock_sequential_streaming_llm_model([[prose_without_json], [repair_json]])
    )
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    result = await run_generation_synthesis(
        agent,
        user_query="học phí ngành tôi học là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert len(result.ask_forms) == 1
    assert [f["field"] for f in result.ask_forms[0]["fields"]] == ["nganh"]
    assert result.response_text == prose_without_json
    assert "ask_user_form" not in "".join(received)
    assert [form["fields"][0]["field"] for form in result.ask_forms] == ["nganh"]


@pytest.mark.asyncio
async def test_model_emitted_form_is_filtered_from_stream_and_captured(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    chunks = [
        "Học phí tuỳ ngành. Bạn học ngành nào?\n\n``",
        '`json\n{"type": "ask_user_form", "fields": [{"field": "nganh", ',
        '"options": [{"id": "cntt", "label": "CNTT"}, {"id": "kt", "label": "Kế toán"}]}]}\n``',
        "`",
    ]
    agent = build_generation_agent(mock_streaming_llm_model(chunks))
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    result = await run_generation_synthesis(
        agent,
        user_query="học phí ngành tôi học là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.response_text == "Học phí tuỳ ngành. Bạn học ngành nào?"
    assert "```" not in "".join(received) and "ask_user_form" not in "".join(received)
    assert len(result.ask_forms) == 1


@pytest.mark.asyncio
async def test_skips_repair_call_when_allow_repair_json_is_false(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CHAT_ALLOW_REPAIR_JSON=false` must skip `_repair_missing_ask_form`
    entirely, not just suppress its effect - only ONE response is scripted
    below (unlike `test_repairs_missing_ask_form_when_prose_asks_for_missing
    _attribute`'s two), so a repair call that fired anyway would exhaust the
    mock and fail the test."""

    monkeypatch.setattr(settings, "CHAT_ALLOW_REPAIR_JSON", False)
    prose_without_json = "Bạn vui lòng cho biết ngành học của bạn để mình tra học phí nhé!"
    agent = build_generation_agent(mock_sequential_streaming_llm_model([[prose_without_json]]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="học phí ngành tôi học là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.response_text == prose_without_json
    assert result.ask_forms == ()


@pytest.mark.asyncio
async def test_repairs_missing_ask_form_with_words_wedged_between_anchor_phrase(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
) -> None:
    """Regression test for a real live miss: the heuristic's first version
    required "cung cấp" to be immediately followed by "thông tin", but the
    actual model output was "cung cấp CHO MÌNH thông tin" - words wedged in
    between made the fixed-phrase regex miss it entirely, so no repair fired
    and the turn genuinely lost the clarification request (see
    tasks/report.md)."""

    prose_without_json = (
        "Về mức học phí, xin vui lòng cung cấp cho mình thông tin về ngành học của bạn."
    )
    repair_json = (
        '```json\n{"type": "ask_user_form", "fields": [{"field": "nganh_hoc", '
        '"options": [{"id": "cntt", "label": "Công nghệ Thông tin"}]}]}\n```'
    )
    agent = build_generation_agent(
        mock_sequential_streaming_llm_model([[prose_without_json], [repair_json]])
    )

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="học phí của ngành tôi học",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert len(result.ask_forms) == 1
    assert [f["field"] for f in result.ask_forms[0]["fields"]] == ["nganh_hoc"]


@pytest.mark.asyncio
async def test_does_not_repair_when_prose_is_not_a_clarification_request(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """A normal answer that happens to contain no JSON block must NOT trigger
    the repair call - only text that heuristically reads like a clarification
    request does."""

    agent = build_generation_agent(
        mock_streaming_llm_model(["Hạn nộp học phí học kỳ này là 15/03 [1]."])
    )

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="hạn nộp học phí học kỳ này là khi nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.ask_forms == ()
    assert result.response_text == "Hạn nộp học phí học kỳ này là 15/03 [1]."


@pytest.mark.asyncio
async def test_does_not_repair_a_conditional_offer_about_an_unrelated_topic(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """Live-observed bug: after fully answering an unrelated question, the
    model appended a generic conditional offer naming a DIFFERENT topic
    ('điều kiện học bổng') the user never asked about this turn. This reads
    like a clarification request to `_CLARIFICATION_PHRASE_PATTERN` alone
    (it contains "cho biết thêm thông tin"), but must NOT trigger repair -
    only ONE response is scripted below, so if repair fired it would run out
    of scripted turns and the test would fail with a mock exhaustion error."""

    prose = (
        "Chương trình đào tạo ngành Logistics gồm 141 tín chỉ, 4 năm [1]. "
        "Nếu bạn cần thêm thông tin chi tiết về điều kiện học bổng hoặc các "
        "thủ tục hành chính cụ thể cho ngành này, vui lòng cho biết thêm "
        "thông tin."
    )
    agent = build_generation_agent(mock_streaming_llm_model([prose]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="chương trình đào tạo ngành Logistics như thế nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.ask_forms == ()
    assert result.response_text == prose


@pytest.mark.asyncio
async def test_does_not_repair_a_trailing_conditional_offer(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """Live-observed bug: the earlier heuristic only recognized "Nếu bạn
    cần..." when it OPENED the sentence, missing the equally common reversed
    phrasing where the "nếu" clause trails the main clause - exactly what
    the model wrote here. Only ONE response is scripted below, so if repair
    (wrongly) fired it would run out of scripted turns and the test would
    fail with a mock exhaustion error."""

    prose = (
        "Mức thu học phí dành cho sinh viên hệ đại học chính quy khóa tuyển sinh "
        "năm học 2025-2026 thuộc khối Công nghệ là 38.350.000 đồng cho một năm "
        "học [1].\n\nBạn vui lòng cho tôi biết thêm thông tin nếu bạn cần hỗ trợ "
        "gì khác."
    )
    agent = build_generation_agent(mock_streaming_llm_model([prose]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="học phí đại học chính quy khối công nghệ khóa 2025-2026 là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.ask_forms == ()
    assert result.response_text == prose


def test_generation_agent_asks_for_the_configured_thinking(
    monkeypatch: pytest.MonkeyPatch,
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    monkeypatch.setattr(settings, "CHAT_GENERATION_THINKING", "low")
    agent = build_generation_agent(mock_streaming_llm_model(["x"]))
    assert agent.model_settings == {"thinking": "low"}


def test_generation_agent_keeps_the_model_default_when_thinking_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    monkeypatch.setattr(settings, "CHAT_GENERATION_THINKING", None)
    agent = build_generation_agent(mock_streaming_llm_model(["x"]))
    assert agent.model_settings == {}
