"""evals.benchmark - candidates, metrics, the three runners and the report, with variant
doubles (no Qdrant, no service, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from evals.benchmark import metrics
from evals.benchmark.candidates import Candidate, load_candidates
from evals.benchmark.report import write_reports
from evals.benchmark.runner import (
    pick_fixed_candidate,
    run_accuracy,
    run_concurrency,
    run_performance,
)
from evals.benchmark.variants import Selection, Variant, VariantError


def _chunk(doc: str, n: int, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc}:{n}",
        content=f"{doc} {n}",
        source=f"{doc}.pdf",
        score=score,
        metadata={"document_id": doc},
    )


def _candidate(qid: str = "q1", category: str = "normal") -> Candidate:
    return Candidate(
        question_id=qid,
        category=category,
        question="học phí?",
        chunks=[_chunk("x", 1), _chunk("good", 1), _chunk("y", 1), _chunk("good", 2)],
        expected_document_ids=frozenset({"good"}),
    )


class FakeRetrieval:
    def __init__(self, missing: set[str] = frozenset()) -> None:
        self.missing = missing
        self.calls: list[tuple[list[str], AcademicSecurityContext]] = []

    def fetch(self, chunk_ids, *, security):
        self.calls.append((list(chunk_ids), security))
        return [
            _chunk(cid.split(":")[0], int(cid.split(":")[1]), 1.0)
            for cid in chunk_ids
            if cid not in self.missing
        ]


def _write_run(tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return run


def test_candidates_come_from_the_first_retrieval_with_recorded_scores(tmp_path: Path) -> None:
    rows = [
        {
            "id": "d1",
            "category": "normal",
            "persona": "guest",
            "access": [],
            "expected_document_ids": ["good"],
            "retrievals": [
                {
                    "query": "Học phí CNTT?",
                    "chunks": [
                        {"chunk_id": "good:1", "score": 0.91},
                        {"chunk_id": "x:1", "score": 0.7},
                    ],
                },
                {"query": "second", "chunks": [{"chunk_id": "y:1", "score": 0.5}]},
            ],
        },
        {
            "id": "d2",
            "category": "normal",
            "expected_document_ids": [],
            "retrievals": [{"query": "q", "chunks": [{"chunk_id": "x:1"}]}],
        },
        {"id": "d3", "category": "calculation", "expected_document_ids": ["z"], "retrievals": []},
        {
            "id": "d4",
            "category": "calculation",
            "expected_document_ids": ["gone"],
            "retrievals": [{"query": "q", "chunks": [{"chunk_id": "gone:1", "score": 0.8}]}],
        },
    ]
    retrieval = FakeRetrieval(missing={"gone:1"})

    candidates, stats = load_candidates(_write_run(tmp_path, rows), retrieval)

    assert [c.question_id for c in candidates] == ["d1"]
    assert candidates[0].question == "Học phí CNTT?"
    assert [(c.chunk_id, c.score) for c in candidates[0].chunks] == [("good:1", 0.91), ("x:1", 0.7)]
    assert candidates[0].relevant_count == 1
    assert dict(stats.skipped) == {"no_expected_document": 1, "no_retrieval": 1, "chunks_gone": 1}
    assert stats.chunks_missing == 1 and stats.kept == 1 and stats.rows == 4
    assert retrieval.calls[0][1].is_guest


def test_candidate_filters_limit_category_and_ids(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"d{i}",
            "category": "normal" if i % 2 else "calculation",
            "expected_document_ids": ["good"],
            "retrievals": [{"query": "q", "chunks": [{"chunk_id": "good:1", "score": 0.9}]}],
        }
        for i in range(6)
    ]
    run = _write_run(tmp_path, rows)

    by_category, _ = load_candidates(run, FakeRetrieval(), categories={"normal"})
    limited, _ = load_candidates(run, FakeRetrieval(), limit=2)
    by_id, _ = load_candidates(run, FakeRetrieval(), ids={"d4"})

    assert [c.question_id for c in by_category] == ["d1", "d3", "d5"]
    assert [c.question_id for c in limited] == ["d0", "d1"]
    assert [c.question_id for c in by_id] == ["d4"]


def test_metrics_binary_at_document_level() -> None:
    candidate = _candidate()
    kept = [_chunk("x", 1), _chunk("good", 1), _chunk("good", 2)]

    scores = metrics.score_question(candidate, kept)

    assert scores.recall_at_8 == 1.0 and scores.recall_at_5 == 1.0
    assert scores.mrr == 0.5
    assert scores.relevant_kept == 2 and scores.kept == 3 and scores.empty is False
    # DCG: ranks 2 and 3 relevant; IDCG: 2 relevant candidates at ranks 1 and 2.
    expected = (1 / metrics.math.log2(3) + 1 / metrics.math.log2(4)) / (
        1 + 1 / metrics.math.log2(3)
    )
    assert scores.ndcg_at_16 == pytest.approx(expected)
    assert metrics.score_question(candidate, []).empty is True
    assert metrics.score_question(candidate, [_chunk("x", 1)]).recall_at_8 == 0.0
    assert metrics.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert (
        metrics.percentile([10, 20, 30, 40], 95) == 40
        and metrics.percentile([10, 20, 30, 40], 50) == 20
    )


class ScriptedVariant(Variant):
    """Returns the kept lists in `script` order, one per call; a None entry fails."""

    uses_llm = False

    def __init__(
        self, name: str, script: list[list[RetrievedChunk] | None], *, uses_llm: bool = False
    ):
        self.name = name
        self.script = list(script)
        self.uses_llm = uses_llm
        self.calls = 0

    async def select(self, candidate: Candidate) -> Selection:
        self.calls += 1
        kept = self.script.pop(0) if self.script else [candidate.chunks[1]]
        if kept is None:
            raise VariantError("service down")
        return Selection(kept=kept, took_ms=12.0)


@pytest.mark.asyncio
async def test_accuracy_aggregates_per_category_and_reports_stability() -> None:
    candidates = [_candidate("q1", "normal"), _candidate("q2", "calculation")]
    good, x = _chunk("good", 1), _chunk("x", 1)
    # repeat 1: both right; repeat 2: q1 right, q2 wrong set; repeat 3: q1 errors, q2 right.
    llm = ScriptedVariant("llm_rerank", [[good], [good], [good], [x], None, [good]], uses_llm=True)
    ce = ScriptedVariant("cross_encoder", [[good, x], [good, x], [good, x], [good, x]])

    results = await run_accuracy(candidates, [llm, ce], llm_repeats=3, ce_repeats=2)

    llm_result, ce_result = results
    assert llm.calls == 6 and ce.calls == 4
    overall = llm_result.aggregate()
    assert overall.questions == 2 and overall.runs == 6 and overall.errors == 1
    assert overall.recall_at_8 == pytest.approx(4 / 5)
    assert llm_result.aggregate("calculation").recall_at_8 == pytest.approx(2 / 3)
    stability = llm_result.stability()
    assert stability is not None and stability.repeats == 3
    assert stability.identical_rate == 0.5  # q1 identical (2 ok runs), q2 differed
    assert stability.recall_at_8_spread == pytest.approx(0.5)
    ce_stability = ce_result.stability()
    assert (
        ce_stability is not None
        and ce_stability.identical_rate == 1.0
        and ce_stability.mean_jaccard == 1.0
    )


@pytest.mark.asyncio
async def test_performance_and_concurrency_count_requests_and_errors() -> None:
    candidates = [_candidate("q1"), _candidate("q2")]
    good = _chunk("good", 1)
    perf_variant = ScriptedVariant("cross_encoder", [[good]] * 20)
    perf = await run_performance(candidates, [perf_variant], warmup=1, iterations=2)
    assert perf_variant.calls == 1 + 4
    assert len(perf[0].latencies_ms) == 4 and perf[0].summary()["p50_ms"] == 12.0
    assert perf[0].resources.samples >= 0

    conc_variant = ScriptedVariant("cross_encoder", [[good]] * 3 + [None] + [[good]] * 20)
    points = await run_concurrency(candidates, [conc_variant], levels=(1, 2), requests=4, warmup=1)
    assert [p.concurrency for p in points] == [1, 2]
    assert points[0].errors == 1 and len(points[0].latencies_ms) == 3
    assert points[1].errors == 0 and len(points[1].latencies_ms) == 4
    assert pick_fixed_candidate(candidates).question_id == "q1"


@pytest.mark.asyncio
async def test_reports_are_written_as_markdown_and_json(tmp_path: Path) -> None:
    candidates = [_candidate()]
    good = _chunk("good", 1)
    variant = ScriptedVariant("cross_encoder", [[good]] * 30)
    accuracy = await run_accuracy(candidates, [variant], ce_repeats=2)
    performance = await run_performance(candidates, [variant], warmup=0, iterations=1)
    concurrency = await run_concurrency(candidates, [variant], levels=(1,), requests=2, warmup=0)
    from evals.benchmark.candidates import CandidateStats

    md_path, json_path = write_reports(
        tmp_path / "out",
        environment={"GPU / driver": "test"},
        candidate_stats=CandidateStats(rows=1, kept=1),
        accuracy=accuracy,
        performance=performance,
        concurrency=concurrency,
        vram={"trước khi benchmark": 100, "sau khi tắt service": 100, "đã nhả VRAM": "có"},
    )

    text = md_path.read_text(encoding="utf-8")
    assert (
        "Bảng 1 — Accuracy" in text
        and "Bảng 2 — Performance" in text
        and "Bảng 3 — Concurrency" in text
    )
    assert "| cross_encoder | tất cả | 1 | 2 | 1.000 |" in text
    assert "đã nhả VRAM | có" in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["accuracy"][0]["aggregate"]["recall_at_8"] == 1.0
    assert payload["accuracy"][0]["runs"][0]["kept_ids"] == ["good:1"]
    assert payload["concurrency"][0]["concurrency"] == 1


@pytest.mark.asyncio
async def test_llm_variant_retries_a_rate_limit_and_times_only_the_last_call(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.graph.nodes.llm_rerank import LLMRerankOutcome
    from app.graph.nodes.post_retrieval_rerank import turn_result
    from evals.benchmark import variants as module

    calls: list[int] = []

    async def fake_llm_rerank(agent, questions, result, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return LLMRerankOutcome(
                result=result, failure="Mô hình Rerank: credential đã dùng hết 15 lượt gọi/phút"
            )
        return LLMRerankOutcome(
            result=turn_result([[result.per_query[0].chunks[1]]], best_scores=[0.9])
        )

    monkeypatch.setattr(module, "llm_rerank", fake_llm_rerank)
    monkeypatch.setattr(module, "build_llm_rerank_agent", lambda model: object())
    models = SimpleNamespace(
        rerank=object(),
        rerank_unavailable=None,
        rerank_credential=None,
        rerank_purpose="RERANK",
        snapshot_version=None,
    )
    variant = module.LLMRerankVariant(models, rpm=0, retry_wait_seconds=0, retries=3)  # type: ignore[arg-type]

    selection = await variant.select(_candidate())

    assert len(calls) == 3
    assert [c.chunk_id for c in selection.kept] == ["good:1"]
    assert selection.took_ms < 1000

    variant_no_retry = module.LLMRerankVariant(models, rpm=0, retry_wait_seconds=0, retries=0)  # type: ignore[arg-type]
    calls.clear()
    with pytest.raises(VariantError):
        await variant_no_retry.select(_candidate())
    assert len(calls) == 1
    assert module.is_rate_limited("HTTP 429 Too Many Requests") is True
    assert module.is_rate_limited("kết quả lọc không đúng định dạng") is False
