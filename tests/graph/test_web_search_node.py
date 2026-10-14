"""WebSearchNode - candidates for the web judge, the result cache and failure handling,
with a fake search client and a fake cache (no network, no Redis)."""

from dataclasses import dataclass, field

import pytest

from app.core.config import settings
from app.graph.nodes.web_search import WebSearchRequest, search_web, select_without_judge
from app.integrations.tavily_client import WebSearchFailureReason, WebSearchUnavailableError
from app.schemas.web_search import WebSearchResult


def _page(
    path: str, score: float, content: str = "nội dung", host: str = "iuh.edu.vn"
) -> WebSearchResult:
    return WebSearchResult(title=path, url=f"https://{host}/{path}", content=content, score=score)


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


@dataclass
class _FakeCache:
    stored: dict[str, list[WebSearchResult]] = field(default_factory=dict)
    ttls: dict[str, int] = field(default_factory=dict)

    async def get(self, query: str, *, max_results: int) -> list[WebSearchResult] | None:
        return self.stored.get(query)

    async def set(
        self, query: str, results: list[WebSearchResult], *, max_results: int, ttl_seconds: int
    ) -> None:
        self.stored[query] = results
        self.ttls[query] = ttl_seconds


@pytest.fixture(autouse=True)
def _web_search_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB", 10)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN", 2)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_QUERIES", 4)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_SCORE_FLOOR", 0.05)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MIN_SCORE", 0.5)
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_RESULT_MAX_CHARS", 1500)


@pytest.mark.asyncio
async def test_low_scored_official_page_still_reaches_the_judge() -> None:
    """The 14-10-2026 case: the fee decision scored 0.08 and the old 0.5 threshold dropped
    it; now only junk under the floor is dropped and the judge decides."""

    client = _FakeClient(
        {
            "q": [
                _page("tin-tuc/hoc-bong", 0.19),
                _page("upload/1035-QD-hoc-phi.pdf", 0.08, host="ptckt.iuh.edu.vn"),
                _page("noise", 0.01),
            ]
        }
    )

    outcome = await search_web(
        [WebSearchRequest(query="q", topic="hoc_phi")], client=client, use_cache=False
    )

    # The fee topic's own site first, then by score; under the floor dropped.
    assert [r.url for r in outcome.results] == [
        "https://ptckt.iuh.edu.vn/upload/1035-QD-hoc-phi.pdf",
        "https://iuh.edu.vn/tin-tuc/hoc-bong",
    ]


@pytest.mark.asyncio
async def test_home_listing_and_other_campus_pages_are_left_out() -> None:
    client = _FakeClient(
        {
            "q": [
                WebSearchResult(title="home", url="https://fit.iuh.edu.vn", content="x", score=0.9),
                _page("news.html@150@Thong-bao", 0.8, host="ctsv.iuh.edu.vn"),
                _page("thong-bao/hoc-phi-qn", 0.7, host="qn.iuh.edu.vn"),
                _page("vi/thong-bao/hoc-phi-2026-2027", 0.3),
            ]
        }
    )

    outcome = await search_web(
        [WebSearchRequest(query="q", topic="hoc_phi")], client=client, use_cache=False
    )

    assert [r.url for r in outcome.results] == ["https://iuh.edu.vn/vi/thong-bao/hoc-phi-2026-2027"]
    assert outcome.results[0].academic_year == "2026-2027"


@pytest.mark.asyncio
async def test_campus_page_is_kept_when_the_question_names_that_campus() -> None:
    client = _FakeClient({"q": [_page("thong-bao/hoc-phi", 0.7, host="qn.iuh.edu.vn")]})

    outcome = await search_web(
        [WebSearchRequest(query="q", topic="hoc_phi", co_so="Quảng Ngãi")],
        client=client,
        use_cache=False,
    )

    assert len(outcome.results) == 1


@pytest.mark.asyncio
async def test_newer_academic_year_comes_first_within_the_same_tier() -> None:
    client = _FakeClient(
        {
            "q": [
                _page("hoc-phi-2025-2026", 0.9, host="pdt.iuh.edu.vn"),
                _page("hoc-phi-2026-2027", 0.4, host="pdt.iuh.edu.vn"),
            ]
        }
    )

    outcome = await search_web(
        [WebSearchRequest(query="q", topic="hoc_phi")], client=client, use_cache=False
    )

    assert [r.academic_year for r in outcome.results] == ["2026-2027", "2025-2026"]


@pytest.mark.asyncio
async def test_candidates_are_shared_round_robin_without_duplicate_urls() -> None:
    client = _FakeClient(
        {
            "q1": [_page("shared", 0.9), _page("a2", 0.8)],
            "q2": [_page("shared", 0.8), _page("b1", 0.7)],
        }
    )

    outcome = await search_web(["q1", "q2"], client=client, use_cache=False)

    assert [r.title for r in outcome.results] == ["shared", "b1", "a2"]
    assert client.calls == [("q1", 10), ("q2", 10)]


@pytest.mark.asyncio
async def test_every_page_left_after_the_filters_reaches_the_judge() -> None:
    client = _FakeClient({"q": [_page(f"p{i}", 0.9 - i / 20) for i in range(10)]})

    outcome = await search_web(["q"], client=client, use_cache=False)

    assert [r.title for r in outcome.results] == [f"p{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_cache_hit_skips_the_search() -> None:
    cache = _FakeCache(stored={"q": [_page("cached", 0.7)]})
    client = _FakeClient({})

    outcome = await search_web(["q"], client=client, cache=cache)

    assert [r.title for r in outcome.results] == ["cached"]
    assert outcome.cache_hits == 1
    assert client.calls == []


@pytest.mark.asyncio
async def test_cache_miss_stores_raw_results_with_the_topic_ttl() -> None:
    cache = _FakeCache()
    raw = [_page("fee", 0.7), _page("noise", 0.01)]
    client = _FakeClient({"q": raw, "empty": []})

    await search_web(
        [
            WebSearchRequest(query="q", topic="hoc_phi"),
            WebSearchRequest(query="empty", topic="hoc_phi"),
        ],
        client=client,
        cache=cache,
    )

    # Raw results, before the floor: what is a candidate depends on the policy, not the cache.
    assert cache.stored["q"] == raw
    assert cache.ttls["q"] == 24 * 3600
    assert cache.ttls["empty"] == 30 * 60


@pytest.mark.asyncio
async def test_a_failed_search_is_not_cached() -> None:
    cache = _FakeCache()
    rate_limited = WebSearchUnavailableError(WebSearchFailureReason.RATE_LIMITED, "HTTP 429")
    client = _FakeClient({"q": rate_limited})

    outcome = await search_web(["q"], client=client, cache=cache)

    assert outcome.failure is rate_limited
    assert cache.stored == {}


@pytest.mark.asyncio
async def test_one_failing_search_does_not_drop_the_others() -> None:
    rate_limited = WebSearchUnavailableError(WebSearchFailureReason.RATE_LIMITED, "HTTP 429")
    client = _FakeClient({"q1": rate_limited, "q2": [_page("ok", 0.8)]})

    outcome = await search_web(["q1", "q2"], client=client, use_cache=False)

    assert [r.title for r in outcome.results] == ["ok"]
    assert outcome.failure is rate_limited


@pytest.mark.asyncio
async def test_disabled_search_makes_no_call_and_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_ENABLED", False)
    client = _FakeClient({})

    outcome = await search_web(["q"], client=client)

    assert (outcome.results, outcome.failure) == ([], None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_enabled_without_a_key_reports_it_as_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")
    client = _FakeClient({})

    outcome = await search_web(["q"], client=client)

    assert outcome.results == []
    assert outcome.failure is not None
    assert outcome.failure.reason is WebSearchFailureReason.NOT_CONFIGURED
    assert client.calls == []


@pytest.mark.asyncio
async def test_no_request_makes_no_call() -> None:
    client = _FakeClient({})

    assert (await search_web([], client=client)).results == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_long_content_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_RESULT_MAX_CHARS", 100)
    client = _FakeClient({"q": [_page("long", 0.9, content="x" * 500)]})

    (result,) = (await search_web(["q"], client=client, use_cache=False)).results

    assert result.content == "x" * 100 + "…"


@pytest.mark.asyncio
async def test_only_the_first_max_queries_requests_are_searched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_WEB_SEARCH_MAX_QUERIES", 2)
    client = _FakeClient(
        {"a": [_page("a1", 0.6)], "b": [_page("b1", 0.9)], "c": [_page("c1", 0.8)]}
    )

    await search_web(["a", "b", "c"], client=client, use_cache=False)

    assert [query for query, _ in client.calls] == ["a", "b"]


@pytest.mark.asyncio
async def test_page_scores_are_logged_without_debug(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "APP_DEBUG", False)
    client = _FakeClient({"q": [_page("good", 0.8), _page("weak", 0.01, content="secret body")]})

    with caplog.at_level("INFO", logger="app.graph.nodes.web_search"):
        await search_web(["q"], client=client, use_cache=False)

    assert "query='q' topic=None cache=off: 2 page(s) found, 1 candidate(s)" in caplog.text
    assert "0.01 https://iuh.edu.vn/weak" in caplog.text
    # Page bodies only with APP_DEBUG.
    assert "secret body" not in caplog.text


def test_without_a_judge_pages_are_kept_by_the_old_score_rule() -> None:
    pages = [_page("a", 0.3), _page("b", 0.9), _page("c", 0.6), _page("d", 0.55)]

    assert [r.title for r in select_without_judge(pages)] == ["b", "c"]
