"""evals.benchmark.retrieval_pool - job selection, the floor, CE ordering, doc-level metrics
and offline scoring from written files (no Qdrant, no service, no embedder)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evals.benchmark import retrieval_pool as rp


def _entry(doc: str, n: int, rank: int, score: float = 0.6, lexical: bool = False) -> dict:
    return {
        "chunk_id": f"{doc}:{n}",
        "document_id": doc,
        "rrf_rank": rank,
        "score": score,
        "lexical_only": lexical,
    }


def test_floor_keeps_lexical_only_hits():
    pool = [_entry("a", 1, 1, 0.40), _entry("b", 1, 2, 0.70, lexical=True), _entry("c", 1, 3, 0.5)]
    assert [e["chunk_id"] for e in rp.floored(pool, 0.45)] == ["b:1", "c:1"]
    assert [e["chunk_id"] for e in rp.ce_candidates(pool, 2, 0.45)] == ["b:1"]


def test_ce_order_ties_keep_retrieval_order():
    assert rp.ce_order(["x", "y", "z"], [1.0, 2.0, 1.0]) == ["y", "x", "z"]


def test_doc_level_metrics():
    expected = frozenset({"good", "other"})
    docs = ["x", "good", "good", "y", "other"]
    m = rp.question_metrics(docs, expected, keep=8)
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["recall@5"] == 1.0
    assert m["hit@5"] == 1.0
    assert m["mrr"] == 0.5
    assert m["first_rank"] == 2
    # a document counts once: the second "good" chunk adds nothing
    assert 0 < m["ndcg@8"] < 1
    assert rp.ndcg_at(["good", "other"], expected, 8) == pytest.approx(1.0)


def test_load_jobs_maps_ids_and_personas(tmp_path: Path):
    dataset = tmp_path
    rows = [
        {
            "id": "q1",
            "category": "access",
            "question": "Câu?",
            "question_user": "cau?",
            "expected_doc_ids": ["f1"],
            "expect_visible": True,
            "persona": "same_department",
            "asker": {"department_access": [{"department_id": "KHOA_A", "access_level": 2}]},
        },
        {
            "id": "q2",
            "category": "access",
            "question": "Ẩn?",
            "question_user": "",
            "expected_doc_ids": ["f1"],
            "expect_visible": False,
            "persona": "guest",
            "asker": {},
        },
        {"id": "q3", "category": "social", "question": "Chào", "expected_doc_ids": []},
    ]
    (dataset / "questions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "document_id", "title"])
        writer.writeheader()
        writer.writerow({"file_id": "f1", "document_id": "uuid-1", "title": "Tài liệu 1"})
    (dataset / "departments.json").write_text(json.dumps({"KHOA_A": "dep-uuid"}))

    jobs = rp.load_jobs(dataset)
    assert [j.key for j in jobs] == ["q1:exact", "q1:user", "q2:exact"]
    assert jobs[0].expected_document_ids == ("uuid-1",)
    assert jobs[0].expected_titles == ("Tài liệu 1",)
    security = jobs[0].security()
    assert [e.department_id for e in security.department_access] == ["dep-uuid"]
    assert not jobs[2].expect_visible
    assert [j.key for j in rp.load_jobs(dataset, include_hidden=False)] == ["q1:exact", "q1:user"]


def test_score_all_offline(tmp_path: Path):
    pool = [
        _entry("x", 1, 1, 0.7),
        _entry("y", 1, 2, 0.40),  # under the floor: no variant sees it
        _entry("x", 2, 3, 0.6),
        _entry("good", 1, 4, 0.7, lexical=True),
    ]
    pool_row = {
        "key": "q1:exact",
        "id": "q1",
        "variant": "exact",
        "category": "normal",
        "persona": "guest",
        "expect_visible": True,
        "text": "?",
        "expected_document_ids": ["good"],
        "expected_titles": ["G"],
        "pool": pool,
        "graph_pool_ids": ["x:1", "x:2"],
    }
    hidden = {**pool_row, "key": "q2:exact", "id": "q2", "expect_visible": False}
    (tmp_path / "pools.jsonl").write_text(
        json.dumps(pool_row) + "\n" + json.dumps(hidden) + "\n", encoding="utf-8"
    )
    ce = {
        "key": "q1:exact",
        "sizes": {
            "4": {
                "ids": ["x:1", "x:2", "good:1"],
                "scores": [0.1, -1.0, 3.0],
                "took_ms": 12.0,
                "truncated": 0,
            }
        },
    }
    (tmp_path / "ce.jsonl").write_text(json.dumps(ce) + "\n", encoding="utf-8")

    summary = rp.score_all(tmp_path, keep=2, floor=0.45, none_depth=4)
    assert summary["queries"] == 2
    rows = [json.loads(line) for line in (tmp_path / "per_query.jsonl").read_text().splitlines()]
    q1 = rows[0]
    assert q1["variants"]["none"]["kept_ids"] == ["x:1", "x:2"]
    assert q1["variants"]["none"]["recall@8"] == 0.0
    assert q1["variants"]["ce4"]["kept_ids"] == ["good:1", "x:1"]
    assert q1["variants"]["ce4"]["first_rank"] == 1
    assert q1["first_rank_rrf"] == 4
    assert q1["in_pool"]["8"] == 1.0
    assert q1["graph_pool_same_order"] is False
    assert q1["leak"] is None
    assert rows[1]["leak"] is True
