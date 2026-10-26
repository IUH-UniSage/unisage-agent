"""Which sub-queries go to web search, decided apart from the rerank stage.

Before UNISAGE-106 the only trigger of the fallback web search was "the LLM rerank left
this sub-query with no chunk". Without an LLM rerank (`none`, `cross_encoder`, `auto`
with strong evidence) a sub-query is never left empty, so that trigger never fires. The
policy here reads the evidence level instead and keeps the two questions separate:

- *Is the internal evidence enough?*  `app/graph/evidence.py` (`ok | weak | none`).
- *Should we look outside for this sub-query?*  this module: `none` always qualifies for
  the fallback search; `weak` only when `CHAT_WEB_SEARCH_ON_WEAK_EVIDENCE` says so; the
  speculative mode (fast-changing topics searched alongside retrieval) is untouched; and
  nothing is searched when web search is off.

The decision is logged with its reasons so the fallback can be evaluated on its own.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import settings
from app.graph.evidence import Level

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebDecision:
    # 0-based sub-query indexes the fallback search should cover, in the order to spend
    # the limited searches (empty ones first, then weak ones).
    sub_queries: list[int]
    reasons: dict[int, str] = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return bool(self.sub_queries)


def decide_fallback(
    levels: Sequence[Level],
    *,
    emptied_by_rerank: Sequence[int] = (),
    enabled: bool | None = None,
    on_weak: bool | None = None,
) -> WebDecision:
    """`levels[i]` is sub-query i's evidence level; `emptied_by_rerank` the sub-queries an
    LLM rerank left without chunks (the pre-existing trigger, still honoured)."""

    enabled = settings.CHAT_WEB_SEARCH_ENABLED if enabled is None else enabled
    on_weak = settings.CHAT_WEB_SEARCH_ON_WEAK_EVIDENCE if on_weak is None else on_weak
    if not enabled:
        return WebDecision(sub_queries=[])
    reasons: dict[int, str] = {}
    for index in emptied_by_rerank:
        reasons[index] = "rerank_emptied"
    for index, level in enumerate(levels):
        if level == "none":
            reasons.setdefault(index, "no_candidates")
        elif level == "weak" and on_weak:
            reasons.setdefault(index, "weak_evidence")
    order = sorted(reasons, key=lambda i: (reasons[i] == "weak_evidence", i))
    decision = WebDecision(sub_queries=order, reasons=reasons)
    logger.info(
        "web_policy fallback=%s reasons=%s levels=%s on_weak=%s",
        decision.sub_queries,
        decision.reasons,
        list(levels),
        on_weak,
    )
    return decision
