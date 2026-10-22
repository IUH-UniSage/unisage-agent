"""Cross-encoder rerank service (docs/specs/SPEC-cross-encoder-rerank.md).

One process, one model loaded in the lifespan, two routes:

    POST /rerank  {"query": str, "passages": [str], "top_k"?: int}
                  -> {"scores": [float], "ranking": [int], "truncated": int,
                      "model": str, "took_ms": int}
    GET  /health  -> {"status", "model", "device", "loaded", "vram_used_mb"}

`scores` follow the passage order; `ranking` lists passage indexes best first (cut to
`top_k` when given). Stopping the process (SIGTERM / Ctrl+C) unloads the model so the
VRAM is returned - the agent never imports torch.

    uv run --project reranker uvicorn reranker.app:app --port 8403
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reranker.scorer import Scorer, scorer_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reranker")

MAX_PASSAGES = int(os.environ.get("RERANKER_MAX_PASSAGES", "64"))


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=MAX_PASSAGES)
    top_k: int | None = Field(default=None, ge=1)


class RerankResponse(BaseModel):
    scores: list[float]
    ranking: list[int]
    truncated: int
    model: str
    took_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str | None
    device: str | None
    loaded: bool
    vram_used_mb: int | None


def create_app(scorer_factory: Callable[[], Scorer] = scorer_from_env) -> FastAPI:
    """`scorer_factory` is swapped in tests so no model is downloaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.scorer = scorer_factory()
        try:
            yield
        finally:
            scorer = getattr(app.state, "scorer", None)
            app.state.scorer = None
            if scorer is not None:
                scorer.close()

    app = FastAPI(title="UniSage reranker", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        scorer: Scorer | None = getattr(app.state, "scorer", None)
        vram = getattr(scorer, "vram_used_mb", None)
        return HealthResponse(
            status="ok" if scorer is not None else "loading",
            model=getattr(scorer, "model_name", None),
            device=getattr(scorer, "device", None),
            loaded=scorer is not None,
            vram_used_mb=vram() if callable(vram) else None,
        )

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest) -> RerankResponse:
        scorer: Scorer | None = getattr(app.state, "scorer", None)
        if scorer is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        result = scorer.score(request.query, request.passages)
        if len(result.scores) != len(request.passages):
            raise HTTPException(status_code=500, detail="scorer returned a wrong number of scores")
        ranking = sorted(range(len(result.scores)), key=lambda i: (-result.scores[i], i))
        if request.top_k is not None:
            ranking = ranking[: request.top_k]
        logger.info(
            "rerank passages=%d truncated=%d took_ms=%d",
            len(request.passages),
            result.truncated,
            result.took_ms,
        )
        return RerankResponse(
            scores=result.scores,
            ranking=ranking,
            truncated=result.truncated,
            model=scorer.model_name,
            took_ms=result.took_ms,
        )

    return app


app = create_app()
