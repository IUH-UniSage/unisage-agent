"""Fallback web search scope from the evidence levels (app/graph/web_policy.py)."""

from app.graph.web_policy import decide_fallback


def test_disabled_web_search_never_searches() -> None:
    decision = decide_fallback(["none", "weak"], emptied_by_rerank=[0], enabled=False)

    assert decision.sub_queries == [] and decision.any is False


def test_empty_sub_queries_and_rerank_emptied_ones_are_searched_weak_only_when_asked() -> None:
    levels = ["ok", "weak", "none", "weak"]

    default = decide_fallback(levels, emptied_by_rerank=[3], enabled=True, on_weak=False)
    on_weak = decide_fallback(levels, emptied_by_rerank=[], enabled=True, on_weak=True)

    assert default.sub_queries == [2, 3]
    assert default.reasons == {3: "rerank_emptied", 2: "no_candidates"}
    # Empty ones first, then the weak ones, each in sub-query order.
    assert on_weak.sub_queries == [2, 1, 3]
    assert on_weak.reasons[1] == "weak_evidence"


def test_strong_evidence_is_never_searched() -> None:
    assert decide_fallback(["ok", "ok"], enabled=True, on_weak=True).sub_queries == []
