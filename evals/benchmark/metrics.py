"""Ranking metrics on a variant's kept list, binary relevance at document level: a
chunk is relevant when its document is one of the question's expected documents."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.schemas.retrieval import RetrievedChunk
from evals.benchmark.candidates import Candidate, document_id_of, is_relevant


@dataclass(frozen=True)
class QuestionScores:
    recall_at_8: float
    recall_at_5: float
    ndcg_at_16: float
    mrr: float
    empty: bool
    kept: int
    relevant_kept: int


def recall_at(kept: Sequence[RetrievedChunk], expected: frozenset[str], k: int) -> float:
    """Share of the expected documents present among the first k kept chunks."""

    if not expected:
        return 0.0
    found = {document_id_of(chunk) for chunk in kept[:k]} & expected
    return len(found) / len(expected)


def mrr(kept: Sequence[RetrievedChunk], expected: frozenset[str]) -> float:
    for rank, chunk in enumerate(kept, 1):
        if is_relevant(chunk, expected):
            return 1.0 / rank
    return 0.0


def ndcg_at(
    kept: Sequence[RetrievedChunk], expected: frozenset[str], k: int, *, ideal_relevant: int
) -> float:
    """Binary NDCG of the kept order; the ideal list is every relevant candidate first
    (`ideal_relevant` = relevant chunks among the candidates), so a variant that keeps
    8 of 16 is judged on where it put the relevant ones AND how many it kept."""

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk in enumerate(kept[:k], 1)
        if is_relevant(chunk, expected)
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(ideal_relevant, k) + 1))
    return dcg / idcg if idcg else 0.0


def score_question(candidate: Candidate, kept: Sequence[RetrievedChunk]) -> QuestionScores:
    expected = candidate.expected_document_ids
    return QuestionScores(
        recall_at_8=recall_at(kept, expected, 8),
        recall_at_5=recall_at(kept, expected, 5),
        ndcg_at_16=ndcg_at(kept, expected, 16, ideal_relevant=candidate.relevant_count),
        mrr=mrr(kept, expected),
        empty=not kept,
        kept=len(kept),
        relevant_kept=sum(1 for chunk in kept if is_relevant(chunk, expected)),
    )


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]
