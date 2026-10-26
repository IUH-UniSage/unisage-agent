"""Evidence sufficiency signals and levels (app/graph/evidence.py)."""

import pytest

from app.core.config import settings
from app.graph.evidence import levels_for, signals_for
from app.schemas.retrieval import RetrievedChunk


def _chunk(
    score: float,
    rrf: float | None = None,
    dense: int | None = None,
    bm25: int | None = None,
    *,
    lexical_only: bool = False,
    ce: float | None = None,
) -> RetrievedChunk:
    meta: dict[str, object] = {
        "lexical_only": lexical_only,
        "rrf_score": rrf,
        "dense_rank": dense,
        "bm25_rank": bm25,
    }
    if ce is not None:
        meta["ce_score"] = ce
    return RetrievedChunk(
        chunk_id=f"c{rrf}", content="x", source="s.pdf", score=score, metadata=meta
    )


def test_signals_read_the_fusion_metadata() -> None:
    chunks = [
        _chunk(0.72, rrf=0.9, dense=1, bm25=2, ce=4.2),
        _chunk(0.70, rrf=0.6, dense=2, bm25=None),
        _chunk(0.70, rrf=0.5, dense=None, bm25=1, lexical_only=True),
    ]

    s = signals_for(chunks)

    assert s.candidates == 3 and s.top1_cosine == 0.72 and s.top1_rrf == 0.9
    assert s.rrf_gap == pytest.approx(0.3) and s.agreement_top3 == 1
    assert s.ce_top1 == 4.2 and s.lexical_only_top1 is False
    assert "cos=0.720" in s.describe() and "agree3=1" in s.describe()


def test_levels_none_weak_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_EVIDENCE_MIN_COSINE", 0.60)
    monkeypatch.setattr(settings, "CHAT_EVIDENCE_MIN_CE", 1.0)
    weak = [_chunk(0.52, rrf=0.5, dense=1), _chunk(0.50, rrf=0.33, dense=2)]
    agreed = [_chunk(0.52, rrf=0.9, dense=1, bm25=1)]  # low cosine but both lists found it
    confident = [_chunk(0.81, rrf=0.5, dense=1)]
    rescued_by_ce = [_chunk(0.52, rrf=0.5, dense=1, ce=3.5)]
    lexical_only = [_chunk(0.70, rrf=0.5, bm25=1, lexical_only=True)]

    assert signals_for([]).level() == "none"
    assert signals_for(weak).level() == "weak"
    assert signals_for(agreed).level() == "ok"
    assert signals_for(confident).level() == "ok"
    assert signals_for(rescued_by_ce).level() == "ok"
    assert signals_for(lexical_only).level() == "weak"  # no cosine, no agreement, no CE
    assert [lvl for _, lvl in levels_for([weak, confident, []])] == ["weak", "ok", "none"]


def test_thresholds_can_be_overridden_per_call() -> None:
    chunks = [_chunk(0.65, rrf=0.5, dense=1)]

    assert signals_for(chunks).level(min_cosine=0.60) == "ok"
    assert signals_for(chunks).level(min_cosine=0.70) == "weak"
