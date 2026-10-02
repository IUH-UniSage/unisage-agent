"""WebSearchNode - allocation, filtering and failure handling with a fake
search client (no network)."""

from dataclasses import dataclass, field

import pytest

from app.core.config import settings
from app.graph.nodes.web_search import search_web
from app.integrations.tavily_client import WebSearchUnavailableError
from app.schemas.web_search import WebSearchResult


def _page(url: str, score: float, content: str = "nội dung") -> WebSearchResult:
    return WebSearchResult(title=url, url=f"https://iuh.edu.vn/{url}", content=content, score=score)


@dataclass
class _FakeClient:
    results: dict[str, list[WebSearchResult] | Exception]
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        self.calls.append((query, max_results))
        outcome = self.results[query]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _web_search_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB", 2)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN", 4)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MIN_SCORE", 0.5)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_RESULT_MAX_CHARS", 1500)


@pytest.mark.asyncio
async def test_every_failed_sub_query_gets_its_best_page_before_extra_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN", 3)
    client = _FakeClient(
        {
            "lich thi": [_page("lt1", 0.95), _page("lt2", 0.9)],
            "hoc bong": [_page("hb1", 0.6), _page("hb2", 0.55)],
        }
    )

    results = await search_web(["lich thi", "hoc bong"], client=client)

    # hb1 makes it in despite lt2 outscoring it; the last slot goes to lt2.
    assert [r.title for r in results] == ["lt1", "lt2", "hb1"]
    assert client.calls == [("lich thi", 2), ("hoc bong", 2)]


@pytest.mark.asyncio
async def test_more_failed_sub_queries_than_slots_keeps_the_best_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN", 2)
    client = _FakeClient(
        {"a": [_page("a1", 0.6)], "b": [_page("b1", 0.9)], "c": [_page("c1", 0.8)]}
    )

    results = await search_web(["a", "b", "c"], client=client)

    assert [r.title for r in results] == ["b1", "c1"]


@pytest.mark.asyncio
async def test_low_score_pages_and_duplicate_urls_are_dropped() -> None:
    client = _FakeClient(
        {
            "q1": [_page("shared", 0.9), _page("weak", 0.3)],
            "q2": [_page("shared", 0.8), _page("other", 0.7)],
        }
    )

    results = await search_web(["q1", "q2"], client=client)

    assert [r.title for r in results] == ["shared", "other"]


@pytest.mark.asyncio
async def test_one_failing_search_does_not_drop_the_others() -> None:
    client = _FakeClient({"q1": WebSearchUnavailableError("HTTP 429"), "q2": [_page("ok", 0.8)]})

    results = await search_web(["q1", "q2"], client=client)

    assert [r.title for r in results] == ["ok"]


@pytest.mark.parametrize(
    ("setting", "value"), [("CHAT_WEB_SEARCH_ENABLED", False), ("TAVILY_API_KEY", "")]
)
@pytest.mark.asyncio
async def test_disabled_or_keyless_search_makes_no_call(
    monkeypatch: pytest.MonkeyPatch, setting: str, value: object
) -> None:
    monkeypatch.setattr(settings, setting, value)
    client = _FakeClient({})

    assert await search_web(["q"], client=client) == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_no_failed_sub_query_makes_no_call() -> None:
    client = _FakeClient({})

    assert await search_web([], client=client) == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_long_content_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_RESULT_MAX_CHARS", 100)
    client = _FakeClient({"q": [_page("long", 0.9, content="x" * 500)]})

    (result,) = await search_web(["q"], client=client)

    assert result.content == "x" * 100 + "…"
