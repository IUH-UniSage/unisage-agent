"""Cross-encoder rerank service (docs/specs/SPEC-cross-encoder-rerank.md).

One process, one model loaded in the lifespan, two routes:

    POST /rerank  {"query": str, "passages": [str], "top_k"?: int}
                  -> {"scores": [float], "ranking": [int], "truncated": int,
                      "model": str, "took_ms": int}
    GET  /health  -> {"status", "model", "device", "loaded", "vram_used_mb"}

`scores` follow the passage order; `ranking` lists passage indexes best first (cut to
`top_k` when given). Stopping the process (SIGTERM / Ctrl+C) unloads the model so the
VRAM is returned - the agent never imports torch.

Dynamic batching: requests queued at the same moment (and, with `RERANKER_BATCH_WINDOW_MS`
> 0, those arriving within that window; up to `RERANKER_MAX_BATCH_PAIRS` pairs) share one
forward pass, run in a worker thread so the event loop keeps answering `/health` under
load; `batched` in the response says how many requests shared it. Measured 26/10 on a
4080: the GPU is compute-bound at these passage lengths (pairs/s constant from batch 16
to 64), so batching does not raise throughput - the window defaults to 0 and the lever is
`RERANKER_MAX_LENGTH`.

    uv run --project reranker uvicorn reranker.app:app --port 8403
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reranker.scorer import Scorer, ScoreResult, scorer_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reranker")

MAX_PASSAGES = int(os.environ.get("RERANKER_MAX_PASSAGES", "64"))
BATCH_WINDOW_MS = float(os.environ.get("RERANKER_BATCH_WINDOW_MS", "0"))
MAX_BATCH_PAIRS = int(os.environ.get("RERANKER_MAX_BATCH_PAIRS", "64"))


@dataclass
class _Job:
    query: str
    passages: list[str]
    future: asyncio.Future[ScoreResult] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class Batcher:
    """Collects concurrent requests into one `score_pairs` call.

    The first job waits up to `window_ms` for company (or until `max_pairs`), then the
    batch runs in a thread; each job gets back only its own scores. One worker, so the GPU
    is never asked to run two forward passes at once."""

    def __init__(
        self,
        scorer: Scorer,
        *,
        window_ms: float = BATCH_WINDOW_MS,
        max_pairs: int = MAX_BATCH_PAIRS,
    ) -> None:
        self._scorer = scorer
        self._window = max(0.0, window_ms) / 1000
        self._max_pairs = max(1, max_pairs)
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.batches = 0
        self.requests = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="reranker-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, query: str, passages: list[str]) -> ScoreResult:
        job = _Job(query, passages)
        await self._queue.put(job)
        return await job.future

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = [await self._queue.get()]
            pairs = len(jobs[0].passages)
            deadline = loop.time() + self._window
            while pairs < self._max_pairs:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    job = await asyncio.wait_for(self._queue.get(), remaining)
                except TimeoutError:
                    break
                jobs.append(job)
                pairs += len(job.passages)
            await self._run(jobs)

    async def _run(self, jobs: list[_Job]) -> None:
        flat = [(job.query, passage) for job in jobs for passage in job.passages]
        try:
            result = await asyncio.to_thread(self._scorer.score_pairs, flat)
        except Exception as exc:  # every waiting request learns about it
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
            return
        self.batches += 1
        self.requests += len(jobs)
        offset = 0
        for job in jobs:
            end = offset + len(job.passages)
            if not job.future.done():
                job.future.set_result(
                    ScoreResult(
                        scores=list(result.scores[offset:end]),
                        truncated=sum(result.truncated[offset:end]),
                        took_ms=result.took_ms,
                        batched=len(jobs),
                    )
                )
            offset = end


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
    batched: int = 1


class HealthResponse(BaseModel):
    status: str
    model: str | None
    device: str | None
    loaded: bool
    vram_used_mb: int | None
    batches: int = 0
    requests: int = 0


def create_app(
    scorer_factory: Callable[[], Scorer] = scorer_from_env,
    *,
    batch_window_ms: float = BATCH_WINDOW_MS,
    max_batch_pairs: int = MAX_BATCH_PAIRS,
) -> FastAPI:
    """`scorer_factory` is swapped in tests so no model is downloaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        scorer = scorer_factory()
        app.state.scorer = scorer
        batcher = Batcher(scorer, window_ms=batch_window_ms, max_pairs=max_batch_pairs)
        batcher.start()
        app.state.batcher = batcher
        try:
            yield
        finally:
            await batcher.stop()
            app.state.batcher = None
            scorer = getattr(app.state, "scorer", None)
            app.state.scorer = None
            if scorer is not None:
                scorer.close()

    app = FastAPI(title="UniSage reranker", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        scorer: Scorer | None = getattr(app.state, "scorer", None)
        vram = getattr(scorer, "vram_used_mb", None)
        batcher: Batcher | None = getattr(app.state, "batcher", None)
        return HealthResponse(
            status="ok" if scorer is not None else "loading",
            model=getattr(scorer, "model_name", None),
            device=getattr(scorer, "device", None),
            loaded=scorer is not None,
            vram_used_mb=vram() if callable(vram) else None,
            batches=batcher.batches if batcher else 0,
            requests=batcher.requests if batcher else 0,
        )

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest) -> RerankResponse:
        scorer: Scorer | None = getattr(app.state, "scorer", None)
        batcher: Batcher | None = getattr(app.state, "batcher", None)
        if scorer is None or batcher is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        try:
            result = await batcher.submit(request.query, request.passages)
        except Exception as exc:
            logger.exception("scoring failed")
            raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc
        if len(result.scores) != len(request.passages):
            raise HTTPException(status_code=500, detail="scorer returned a wrong number of scores")
        ranking = sorted(range(len(result.scores)), key=lambda i: (-result.scores[i], i))
        if request.top_k is not None:
            ranking = ranking[: request.top_k]
        logger.info(
            "rerank passages=%d truncated=%d took_ms=%d batched=%d",
            len(request.passages),
            result.truncated,
            result.took_ms,
            result.batched,
        )
        return RerankResponse(
            scores=result.scores,
            ranking=ranking,
            truncated=result.truncated,
            model=scorer.model_name,
            took_ms=result.took_ms,
            batched=result.batched,
        )

    return app


app = create_app()
