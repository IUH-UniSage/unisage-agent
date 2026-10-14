"""WebJudgeNode - which pages answer, which are only official sources, and falling back
to the score rule, with model doubles (no network)."""

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.graph.nodes.web_judge import (
    MalformedWebJudgementError,
    build_judge_prompt,
    build_web_judge_agent,
    judge_web_results,
    parse_judgement,
)
from app.schemas.web_search import WebSearchResult
from tests.llm_mocks import make_sync_llm_model

_SCHOLARSHIP = WebSearchResult(
    title="IUH dành 80 tỉ đồng học bổng năm học 2026-2027",
    url="https://iuh.edu.vn/vi/tin-tuc/hoc-bong",
    content="... hỗ trợ 100% học phí ...",
    score=0.9,
)
_FEE_DECISION = WebSearchResult(
    title="1035-QD-DHCN",
    url="https://ptckt.iuh.edu.vn/upload/files/bieumau/1035-QD-DHCN-11-4-2025.pdf",
    content="Quy định mức học phí ... năm học 2025-2026 (chi tiết theo phụ lục đính kèm).",
    score=0.08,
    academic_year="2025-2026",
)
_NOTICE = WebSearchResult(
    title="Thông báo đóng học phí HK1 2026-2027",
    url="https://pdt.iuh.edu.vn/tb.pdf",
    content="Hạn đóng học phí học kỳ I: 15/09/2026.",
    score=0.83,
    academic_year="2026-2027",
)


@pytest.fixture(autouse=True)
def _limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN", 2)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MIN_SCORE", 0.5)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_SOURCES", 3)


@pytest.mark.asyncio
async def test_pages_split_into_answers_and_official_sources() -> None:
    agent = build_web_judge_agent(
        make_sync_llm_model(
            '{"SQ1": {"W2": {"reason": "QĐ mức học phí 2025-2026, đoạn trích chỉ có tiêu đề",'
            ' "answer": false}}, "SQ2": {"W3": {"reason": "hạn đóng 15/09/2026", "answer": true}}}'
        )
    )

    judgement = await judge_web_results(
        agent,
        ["Học phí ngành CNTT", "Hạn đóng học phí HK1"],
        [_SCHOLARSHIP, _FEE_DECISION, _NOTICE],
    )

    assert judgement.failure is None
    assert judgement.answers == [_NOTICE]
    assert judgement.sources == [_FEE_DECISION]


@pytest.mark.asyncio
async def test_a_page_answering_one_sub_query_is_not_also_a_source() -> None:
    agent = build_web_judge_agent(
        make_sync_llm_model(
            '{"SQ1": {"W1": {"reason": "hạn đóng", "answer": true}},'
            ' "SQ2": {"W1": {"reason": "cùng thông báo", "answer": false}}}'
        )
    )

    judgement = await judge_web_results(agent, ["a", "b"], [_NOTICE])

    assert (judgement.answers, judgement.sources) == ([_NOTICE], [])


@pytest.mark.asyncio
async def test_nothing_kept_means_neither_answers_nor_sources() -> None:
    agent = build_web_judge_agent(make_sync_llm_model('{"SQ1": {}}'))

    judgement = await judge_web_results(agent, ["Học phí ngành CNTT"], [_SCHOLARSHIP])

    assert (judgement.answers, judgement.sources, judgement.failure) == ([], [], None)


@pytest.mark.asyncio
async def test_a_failed_judge_falls_back_to_the_score_rule() -> None:
    def broken(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(401, "gpt-x")

    judgement = await judge_web_results(
        build_web_judge_agent(FunctionModel(function=broken)),
        ["Học phí ngành CNTT"],
        [_FEE_DECISION, _SCHOLARSHIP],
    )

    assert judgement.failure is not None
    assert judgement.answers == [_SCHOLARSHIP]


@pytest.mark.asyncio
async def test_malformed_output_falls_back_too() -> None:
    judgement = await judge_web_results(
        build_web_judge_agent(make_sync_llm_model("không phải JSON")), ["q"], [_NOTICE]
    )

    assert judgement.failure is not None
    assert "không đúng định dạng" in judgement.failure


def test_parse_needs_a_reason_and_reads_answer_strictly() -> None:
    kept = parse_judgement(
        '{"SQ1": {"W1": {"reason": "có số", "answer": true}, "W2": {"reason": "", "answer": true},'
        ' "W3": {"reason": "chỉ tiêu đề"}, "W9": {"reason": "ngoài danh sách", "answer": true}}}',
        sub_query_count=1,
        page_count=3,
    )

    assert kept == [{0: ("có số", True), 2: ("chỉ tiêu đề", False)}]


def test_parse_rejects_output_without_sub_query_keys() -> None:
    with pytest.raises(MalformedWebJudgementError):
        parse_judgement('{"W1": true}', sub_query_count=1, page_count=1)


def test_prompt_shows_each_page_with_its_academic_year() -> None:
    prompt = build_judge_prompt(["Học phí ngành CNTT"], [_FEE_DECISION, _SCHOLARSHIP])

    assert "Năm học hiện tại:" in prompt
    assert "SQ1. Học phí ngành CNTT" in prompt
    assert "[W1] (Tiêu đề: 1035-QD-DHCN; URL: https://ptckt.iuh.edu.vn" in prompt
    assert "Năm học: 2025-2026)" in prompt
    assert "Năm học: không ghi)" in prompt


def test_prompt_names_the_campus_whose_site_a_page_is_on() -> None:
    notice = WebSearchResult(
        title="Thông báo v/v nghỉ lễ Quốc khánh năm 2025",
        url="https://qn.iuh.edu.vn/thong-bao/thong-bao-v-v-nghi-le-quoc-khanh-nam-2025",
        content="tb-846-2025 Tải xuống",
        score=0.65,
    )

    prompt = build_judge_prompt(["Lịch nghỉ lễ Quốc khánh 2025"], [notice, _FEE_DECISION])

    assert "Năm học: không ghi; Website của Phân hiệu Quảng Ngãi)" in prompt
    assert prompt.count("Website của") == 1


@pytest.mark.asyncio
async def test_sources_are_capped_by_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_SOURCES", 1)
    agent = build_web_judge_agent(
        make_sync_llm_model(
            '{"SQ1": {"W1": {"reason": "QĐ học phí", "answer": false},'
            ' "W2": {"reason": "thông báo học phí", "answer": false}}}'
        )
    )

    judgement = await judge_web_results(agent, ["Học phí"], [_FEE_DECISION, _NOTICE])

    assert judgement.sources == [_FEE_DECISION]
