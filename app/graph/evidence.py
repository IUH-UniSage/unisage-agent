"""Evidence sufficiency: how confident retrieval is about a sub-query, from signals the
fusion already produced, before any model is asked.

Per sub-query, from the chunks the rerank stage kept (retrieval order):

- `top1_cosine`: best dense cosine (a BM25-only chunk has none);
- `top1_rrf` / `rrf_gap`: fused score of the first chunk and its lead over the second;
- `agreement_top3`: chunks among the first three that both dense and BM25 found;
- `ce_top1`: the cross-encoder's score of the first chunk when that stage ran;
- `lexical_only_top1`: the first chunk was found by BM25 alone.

`level()` folds them into `ok | weak | none`: `weak` when the best cosine is under
`CHAT_EVIDENCE_MIN_COSINE` and, if a cross-encoder scored the top chunk, that score is
under `CHAT_EVIDENCE_MIN_CE`. It is a *relative* signal (RRF scores and gaps are not
probabilities), so every turn logs all signals next to the level and the thresholds come
from data (UNISAGE-106 Phase 6 study: `task benchmark:evidence`), never by hand. Used by
`CHAT_RERANK_MODE=auto` (LLM rerank only when a sub-query is weak) and by the web search
policy (`app/graph/web_policy.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from app.core.config import settings
from app.schemas.retrieval import RetrievedChunk

Level = Literal["ok", "weak", "none"]


@dataclass(frozen=True)
class EvidenceSignals:
    candidates: int
    top1_cosine: float | None
    top1_rrf: float | None
    rrf_gap: float | None
    agreement_top3: int
    ce_top1: float | None
    lexical_only_top1: bool

    def level(self, *, min_cosine: float | None = None, min_ce: float | None = None) -> Level:
        if self.candidates == 0:
            return "none"
        min_cosine = settings.CHAT_EVIDENCE_MIN_COSINE if min_cosine is None else min_cosine
        min_ce = settings.CHAT_EVIDENCE_MIN_CE if min_ce is None else min_ce
        # Threshold study (139 câu demo, report Phase 6): the 5 questions whose document is
        # not in the top-8 all had dense/BM25 agreement 1-3, so agreement does not mark a
        # miss; cosine < 0,60 flags 10 % of turns and catches 3/5. Agreement stays logged.
        cosine_weak = self.top1_cosine is None or self.top1_cosine < min_cosine
        ce_weak = self.ce_top1 is None or self.ce_top1 < min_ce
        return "weak" if cosine_weak and ce_weak else "ok"

    def describe(self) -> str:
        return (
            f"n={self.candidates} cos={_f(self.top1_cosine)} rrf={_f(self.top1_rrf)} "
            f"gap={_f(self.rrf_gap)} agree3={self.agreement_top3} ce={_f(self.ce_top1)} "
            f"lex1={int(self.lexical_only_top1)}"
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def signals_for(chunks: Sequence[RetrievedChunk]) -> EvidenceSignals:
    """Signals of one sub-query from its kept chunks, in retrieval order."""

    if not chunks:
        return EvidenceSignals(0, None, None, None, 0, None, True)
    dense = [c.score for c in chunks if not c.metadata.get("lexical_only")]
    rrf = [_num(c.metadata.get("rrf_score")) for c in chunks]
    rrf_known = [r for r in rrf if r is not None]
    top1_rrf = rrf[0]
    gap = (rrf[0] - rrf[1]) if len(rrf) > 1 and rrf[0] is not None and rrf[1] is not None else None
    agreement = sum(
        1
        for c in chunks[:3]
        if c.metadata.get("dense_rank") is not None and c.metadata.get("bm25_rank") is not None
    )
    return EvidenceSignals(
        candidates=len(chunks),
        top1_cosine=max(dense) if dense else None,
        top1_rrf=top1_rrf if rrf_known else None,
        rrf_gap=gap,
        agreement_top3=agreement,
        ce_top1=_num(chunks[0].metadata.get("ce_score")),
        lexical_only_top1=bool(chunks[0].metadata.get("lexical_only")),
    )


def levels_for(
    per_query: Sequence[Sequence[RetrievedChunk]],
) -> list[tuple[EvidenceSignals, Level]]:
    """One (signals, level) per sub-query."""

    out = []
    for chunks in per_query:
        signals = signals_for(chunks)
        out.append((signals, signals.level()))
    return out
