"""The benchmark's input: for each eval question, the standalone question and the chunks
retrieval handed the rerank, re-read from Qdrant by id.

`runs/<run>/results.jsonl` (written by `evals.run`) records each turn's retrievals: the
query text and the chunk ids in retrieval order (RRF for hybrid). The chunk text is not
recorded, so it is fetched again with the same permission filter the turn had; a chunk
since re-indexed is simply missing. Every variant then judges exactly the same list.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from evals.run.runner import security_of

logger = logging.getLogger("evals.benchmark")


@dataclass(frozen=True)
class Candidate:
    question_id: str
    category: str
    question: str
    chunks: list[RetrievedChunk]  # retrieval order, scores as retrieved
    expected_document_ids: frozenset[str]

    @property
    def relevant_count(self) -> int:
        return sum(1 for chunk in self.chunks if is_relevant(chunk, self.expected_document_ids))


@dataclass
class CandidateStats:
    rows: int = 0
    kept: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    chunks_missing: int = 0

    def describe(self) -> str:
        skipped = ", ".join(f"{why}={n}" for why, n in sorted(self.skipped.items())) or "none"
        return (
            f"{self.kept}/{self.rows} questions usable (skipped: {skipped}; "
            f"{self.chunks_missing} recorded chunk(s) no longer in the collection)"
        )


def document_id_of(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("document_id") or chunk.chunk_id.split(":")[0])


def is_relevant(chunk: RetrievedChunk, expected: frozenset[str]) -> bool:
    return document_id_of(chunk) in expected


def run_dir(name_or_path: str, *, runs_root: Path = Path("runs")) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    candidate = runs_root / name_or_path
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"run not found: {name_or_path} (looked in ./ and {runs_root}/)")


def read_results(run: Path) -> list[dict[str, Any]]:
    path = run / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing - run `task eval:run ... --out {run}` first")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_candidates(
    run: Path,
    retrieval: RetrievalServiceProtocol,
    *,
    limit: int | None = None,
    categories: set[str] | None = None,
    ids: set[str] | None = None,
) -> tuple[list[Candidate], CandidateStats]:
    """Questions of `run` that have expected documents and a recorded first retrieval
    whose chunks can still be fetched. `limit` keeps the first N usable ones."""

    stats = CandidateStats()
    candidates: list[Candidate] = []
    for row in read_results(run):
        stats.rows += 1
        if categories and row.get("category") not in categories:
            stats.skipped["category"] += 1
            continue
        if ids and row.get("id") not in ids:
            stats.skipped["id"] += 1
            continue
        expected = frozenset(row.get("expected_document_ids") or [])
        if not expected:
            stats.skipped["no_expected_document"] += 1
            continue
        retrievals = row.get("retrievals") or []
        if not retrievals or not retrievals[0].get("chunks"):
            stats.skipped["no_retrieval"] += 1
            continue
        first = retrievals[0]
        recorded = {str(c["chunk_id"]): float(c.get("score") or 0.0) for c in first["chunks"]}
        security = security_of(row, row.get("access") or [])
        fetched = retrieval.fetch(list(recorded), security=security)
        stats.chunks_missing += len(recorded) - len(fetched)
        if not fetched:
            stats.skipped["chunks_gone"] += 1
            continue
        chunks = [
            chunk.model_copy(update={"score": recorded.get(chunk.chunk_id, chunk.score)})
            for chunk in fetched
        ]
        candidates.append(
            Candidate(
                question_id=str(row["id"]),
                category=str(row.get("category") or "?"),
                question=str(first["query"]),
                chunks=chunks,
                expected_document_ids=expected,
            )
        )
        stats.kept += 1
        if limit is not None and stats.kept >= limit:
            break
    logger.info("candidates from %s: %s", run, stats.describe())
    return candidates, stats
