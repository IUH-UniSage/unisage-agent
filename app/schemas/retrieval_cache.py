"""What the retrieval cache stores (docs/specs/SPEC-retrieval-cache.md).

Only chunk ids and scores, never chunk content: on a hit the chunks are fetched again
from Qdrant with the caller's permission filter, so an edited document reads back fresh
and a deleted or no-longer-visible chunk turns the hit into a miss.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CachedChunkRef(BaseModel):
    """One chunk a turn's rerank kept: its payload `chunk_id` ("{document_id}:{index}")
    and the score it had, which `turn_result` needs back to rebuild the rerank result."""

    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)

    @property
    def document_id(self) -> str:
        return self.chunk_id.rsplit(":", 1)[0]


class CachedRetrievalBundle(BaseModel):
    """L2 entry: the retrieval + rerank outcome of one list of standalone questions."""

    schema_version: Literal[1] = 1
    created_at: datetime
    # The standalone questions, raw (the key holds their normalized form) - for debugging.
    questions: list[str]
    # `TurnRerankResult.per_query` after the LLM rerank, one list per question.
    per_query: list[list[CachedChunkRef]]
    # `TurnRerankResult.best_scores` - what web search uses to rank the failed queries.
    best_scores: list[float]
    # `TurnRerankResult.chunks` as chunk ids, in order (the generation context).
    merged: list[str]

    def chunk_ids(self) -> list[str]:
        """Every id the bundle references, each once, merged order first."""

        seen: dict[str, None] = dict.fromkeys(self.merged)
        for refs in self.per_query:
            for ref in refs:
                seen.setdefault(ref.chunk_id, None)
        return list(seen)

    def document_ids(self) -> list[str]:
        return list(dict.fromkeys(chunk_id.rsplit(":", 1)[0] for chunk_id in self.chunk_ids()))


class CachedSubQuery(BaseModel):
    question: str
    retrieval_text: str
    task_index: int = 0


class CachedTurnBundle(CachedRetrievalBundle):
    """L1 entry: L2's outcome plus what the classifier and HyDE produced, so a hit can
    skip them too. `tasks` holds every `ClassifiedTask` field, including the ones the
    task model excludes from its own dump (hyde_text, sub_queries, web, sub_query_web),
    because web search and generation read them."""

    tasks: list[dict[str, Any]]
    confidence: float | None = None
    sub_queries: list[CachedSubQuery]
