"""The three measurements: -a accuracy, -p performance, -c concurrency.

Every variant sees the same candidates. `-a` runs each variant over every question
(`repeats` times for the non-deterministic LLM variants, twice for the cross-encoder to
confirm it is deterministic) and scores the kept list. `-p` warms up, then loops the
whole set `iterations` times at concurrency 1 with the resource sampler running. `-c`
fires `conc_requests` requests of one fixed question per concurrency level through an
`asyncio.Semaphore`, like vya-ocr's benchmark.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from evals.benchmark.candidates import Candidate
from evals.benchmark.metrics import QuestionScores, jaccard, mean, percentile, score_question
from evals.benchmark.resource_sampler import ResourceSampler, ResourceStats
from evals.benchmark.variants import Variant, VariantError

logger = logging.getLogger("evals.benchmark")

DEFAULT_WARMUP = 3
DEFAULT_PERF_ITERATIONS = 5
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 4, 8)
DEFAULT_CONC_REQUESTS = 30
DEFAULT_LLM_REPEATS = 3
CE_REPEATS = 2  # determinism check, not stability


# ── Accuracy ─────────────────────────────────────────────────────────────────


@dataclass
class QuestionRun:
    question_id: str
    category: str
    repeat: int
    kept_ids: list[str]
    scores: QuestionScores | None
    took_ms: float
    error: str | None = None
    detail: dict[str, Any] | None = None


@dataclass
class AccuracyAgg:
    questions: int = 0
    runs: int = 0
    errors: int = 0
    recall_at_8: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_16: float = 0.0
    mrr: float = 0.0
    empty_rate: float = 0.0
    avg_kept: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0


@dataclass
class StabilityAgg:
    """Across repeats of one variant: how alike the kept sets were."""

    repeats: int
    mean_jaccard: float
    identical_rate: float  # questions whose kept ORDER was the same every repeat
    recall_at_8_spread: float  # mean over questions of (max - min recall@8 across repeats)


@dataclass
class AccuracyResult:
    variant: str
    repeats: int
    runs: list[QuestionRun] = field(default_factory=list)

    def aggregate(self, category: str | None = None) -> AccuracyAgg:
        runs = [r for r in self.runs if category is None or r.category == category]
        scored = [r.scores for r in runs if r.scores is not None]
        agg = AccuracyAgg(
            questions=len({r.question_id for r in runs}),
            runs=len(runs),
            errors=sum(1 for r in runs if r.error),
        )
        if scored:
            agg.recall_at_8 = mean([s.recall_at_8 for s in scored])
            agg.recall_at_5 = mean([s.recall_at_5 for s in scored])
            agg.ndcg_at_16 = mean([s.ndcg_at_16 for s in scored])
            agg.mrr = mean([s.mrr for s in scored])
            agg.empty_rate = mean([1.0 if s.empty else 0.0 for s in scored])
            agg.avg_kept = mean([float(s.kept) for s in scored])
        took = [r.took_ms for r in runs if not r.error]
        agg.p50_ms, agg.p95_ms = percentile(took, 50), percentile(took, 95)
        return agg

    def categories(self) -> list[str]:
        return sorted({r.category for r in self.runs})

    def stability(self) -> StabilityAgg | None:
        if self.repeats < 2:
            return None
        by_question: dict[str, list[QuestionRun]] = defaultdict(list)
        for run in self.runs:
            if not run.error:
                by_question[run.question_id].append(run)
        jaccards: list[float] = []
        identical: list[float] = []
        spreads: list[float] = []
        for runs in by_question.values():
            if len(runs) < 2:
                continue
            pairs = [
                jaccard(runs[i].kept_ids, runs[j].kept_ids)
                for i in range(len(runs))
                for j in range(i + 1, len(runs))
            ]
            jaccards.append(mean(pairs))
            identical.append(1.0 if all(r.kept_ids == runs[0].kept_ids for r in runs) else 0.0)
            recalls = [r.scores.recall_at_8 for r in runs if r.scores is not None]
            spreads.append(max(recalls) - min(recalls) if recalls else 0.0)
        return StabilityAgg(
            repeats=self.repeats,
            mean_jaccard=mean(jaccards),
            identical_rate=mean(identical),
            recall_at_8_spread=mean(spreads),
        )


async def run_accuracy(
    candidates: list[Candidate],
    variants: list[Variant],
    *,
    llm_repeats: int = DEFAULT_LLM_REPEATS,
    ce_repeats: int = CE_REPEATS,
    llm_concurrency: int = 4,
) -> list[AccuracyResult]:
    results: list[AccuracyResult] = []
    for variant in variants:
        repeats = llm_repeats if variant.uses_llm else ce_repeats
        result = AccuracyResult(variant=variant.name, repeats=repeats)
        # LLM calls are network-bound: a few in flight; the service is scored one at a time
        # so -a's timings stay comparable with -p.
        semaphore = asyncio.Semaphore(llm_concurrency if variant.uses_llm else 1)

        async def one(
            candidate: Candidate,
            repeat: int,
            variant: Variant = variant,
            semaphore: asyncio.Semaphore = semaphore,
        ) -> QuestionRun:
            async with semaphore:
                started = time.perf_counter()
                try:
                    selection = await variant.select(candidate)
                except VariantError as exc:
                    return QuestionRun(
                        candidate.question_id,
                        candidate.category,
                        repeat,
                        [],
                        None,
                        (time.perf_counter() - started) * 1000,
                        error=str(exc)[:200],
                    )
                return QuestionRun(
                    candidate.question_id,
                    candidate.category,
                    repeat,
                    [chunk.chunk_id for chunk in selection.kept],
                    score_question(candidate, selection.kept),
                    selection.took_ms,
                    detail=selection.detail,
                )

        for repeat in range(1, repeats + 1):
            logger.info(
                "-a %s repeat %d/%d over %d questions",
                variant.name,
                repeat,
                repeats,
                len(candidates),
            )
            result.runs.extend(
                await asyncio.gather(*(one(candidate, repeat) for candidate in candidates))
            )
        agg = result.aggregate()
        logger.info(
            "-a %s: recall@8=%.3f recall@5=%.3f ndcg@16=%.3f mrr=%.3f empty=%.1f%% "
            "errors=%d p50=%.0fms",
            variant.name,
            agg.recall_at_8,
            agg.recall_at_5,
            agg.ndcg_at_16,
            agg.mrr,
            agg.empty_rate * 100,
            agg.errors,
            agg.p50_ms,
        )
        results.append(result)
    return results


# ── Performance ──────────────────────────────────────────────────────────────


@dataclass
class PerfResult:
    variant: str
    warmup: int
    iterations: int
    questions: int
    latencies_ms: list[float] = field(default_factory=list)
    service_ms: list[float] = field(default_factory=list)
    errors: int = 0
    wall_s: float = 0.0
    resources: ResourceStats = field(default_factory=ResourceStats)

    @property
    def throughput_rps(self) -> float:
        return len(self.latencies_ms) / self.wall_s if self.wall_s else 0.0

    def summary(self) -> dict[str, float]:
        lat = self.latencies_ms
        return {
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "avg_ms": mean(lat),
            "max_ms": max(lat) if lat else 0.0,
            "stdev_ms": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
            "throughput_rps": self.throughput_rps,
        }


async def run_performance(
    candidates: list[Candidate],
    variants: list[Variant],
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_PERF_ITERATIONS,
) -> list[PerfResult]:
    results: list[PerfResult] = []
    for variant in variants:
        result = PerfResult(variant.name, warmup, iterations, len(candidates))
        for i in range(min(warmup, len(candidates))):
            try:
                await variant.select(candidates[i % len(candidates)])
            except VariantError as exc:
                logger.warning("-p %s warm-up #%d failed: %s", variant.name, i + 1, exc)
        with ResourceSampler() as sampler:
            started = time.perf_counter()
            for round_index in range(iterations):
                for candidate in candidates:
                    try:
                        selection = await variant.select(candidate)
                    except VariantError:
                        result.errors += 1
                        continue
                    result.latencies_ms.append(selection.took_ms)
                    if selection.service_ms is not None:
                        result.service_ms.append(selection.service_ms)
                logger.info("-p %s round %d/%d done", variant.name, round_index + 1, iterations)
            result.wall_s = time.perf_counter() - started
        result.resources = sampler.stats
        summary = result.summary()
        logger.info(
            "-p %s: p50=%.0fms p95=%.0fms avg=%.0fms %.2f req/s errors=%d peak_vram=%s MiB",
            variant.name,
            summary["p50_ms"],
            summary["p95_ms"],
            summary["avg_ms"],
            summary["throughput_rps"],
            result.errors,
            result.resources.peak_vram_mb,
        )
        results.append(result)
    return results


# ── Concurrency ──────────────────────────────────────────────────────────────


@dataclass
class ConcurrencyPoint:
    variant: str
    concurrency: int
    requests: int
    question_id: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    wall_s: float = 0.0
    resources: ResourceStats = field(default_factory=ResourceStats)

    @property
    def throughput_rps(self) -> float:
        return len(self.latencies_ms) / self.wall_s if self.wall_s else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


def pick_fixed_candidate(candidates: list[Candidate]) -> Candidate:
    """The first question with the most candidates (16 at the default settings)."""

    return max(candidates, key=lambda c: len(c.chunks))


async def run_concurrency(
    candidates: list[Candidate],
    variants: list[Variant],
    *,
    levels: tuple[int, ...] = DEFAULT_CONCURRENCY_LEVELS,
    requests: int = DEFAULT_CONC_REQUESTS,
    warmup: int = DEFAULT_WARMUP,
) -> list[ConcurrencyPoint]:
    fixed = pick_fixed_candidate(candidates)
    points: list[ConcurrencyPoint] = []
    for variant in variants:
        for i in range(warmup):
            try:
                await variant.select(fixed)
            except VariantError as exc:
                logger.warning("-c %s warm-up #%d failed: %s", variant.name, i + 1, exc)
        for level in levels:
            point = ConcurrencyPoint(variant.name, level, requests, fixed.question_id)
            semaphore = asyncio.Semaphore(level)

            async def one(
                variant: Variant = variant, semaphore: asyncio.Semaphore = semaphore
            ) -> tuple[float, str | None]:
                async with semaphore:
                    try:
                        selection = await variant.select(fixed)
                    except VariantError as exc:
                        return 0.0, type(exc).__name__
                    # The call itself (the LLM variants exclude their rate-limit waits, which
                    # the wall time / throughput still reflects).
                    return selection.took_ms, None

            with ResourceSampler() as sampler:
                started = time.perf_counter()
                outcomes = await asyncio.gather(*(one() for _ in range(requests)))
                point.wall_s = time.perf_counter() - started
            point.resources = sampler.stats
            for latency, error in outcomes:
                if error:
                    point.errors += 1
                else:
                    point.latencies_ms.append(latency)
            logger.info(
                "-c %s concurrency=%d: p50=%.0fms p95=%.0fms %.2f req/s errors=%d",
                variant.name,
                level,
                percentile(point.latencies_ms, 50),
                percentile(point.latencies_ms, 95),
                point.throughput_rps,
                point.errors,
            )
            points.append(point)
    return points
