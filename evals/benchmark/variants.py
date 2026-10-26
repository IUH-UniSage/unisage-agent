"""The three rerank strategies behind one signature: `select(candidate) -> Selection`.

Each variant sees the same candidate list and returns the chunks it would hand the
generation prompt, best first. The LLM variants call the real `llm_rerank` with the
RERANK model the eval environment resolves (`get_graph_models`, so the backend must be
up); the cross-encoder variants call the running service. A variant failure (service
down, model call failed, malformed output) is raised, not failed open, so the report
counts it as an error rather than crediting the unreranked list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.graph.nodes.cross_encoder_rerank import cross_encoder_rerank
from app.graph.nodes.llm_rerank import build_llm_rerank_agent, llm_rerank
from app.graph.nodes.post_retrieval_rerank import TurnRerankResult, turn_result
from app.graph.streaming_state import GraphModels
from app.integrations.cross_encoder_client import CrossEncoderClient
from app.schemas.retrieval import RetrievedChunk
from evals.benchmark.candidates import Candidate

logger = logging.getLogger("evals.benchmark")

VARIANT_NAMES = ("llm_rerank", "cross_encoder", "cross_encoder_then_llm")
LLM_VARIANTS = frozenset({"llm_rerank", "cross_encoder_then_llm"})
CE_VARIANTS = frozenset({"cross_encoder", "cross_encoder_then_llm"})


class VariantError(RuntimeError):
    """The variant could not produce a selection for this question."""


DEFAULT_LLM_RPM = 14  # the eval RERANK credential is capped at maxRpm 15
DEFAULT_LLM_RETRY_WAIT_SECONDS = 20.0
DEFAULT_LLM_RETRIES = 6
_RATE_LIMIT_MARKERS = ("lượt gọi/phút", "maxrpm", "quota", "429", "rate limit", "giới hạn tốc độ")


def is_rate_limited(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


class RpmPacer:
    """Spaces call starts 60/rpm apart across every concurrent caller (rpm 0 = no pacing)."""

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self._interval


@dataclass(frozen=True)
class Selection:
    kept: list[RetrievedChunk]
    took_ms: float  # the rerank step alone, as this process measured it
    service_ms: float | None = None  # cross-encoder: the service's own scoring time
    detail: dict[str, Any] | None = None


ModelsLoader = Callable[[], Awaitable[GraphModels]]


def _single_query(candidate: Candidate) -> TurnRerankResult:
    best = max((chunk.score for chunk in candidate.chunks), default=0.0)
    return turn_result([list(candidate.chunks)], best_scores=[best])


class Variant:
    name: str
    uses_llm = False
    uses_service = False

    async def select(self, candidate: Candidate) -> Selection:
        raise NotImplementedError


class CrossEncoderVariant(Variant):
    name = "cross_encoder"
    uses_service = True

    def __init__(self, client: CrossEncoderClient, *, keep: int) -> None:
        self._client = client
        self._keep = keep

    async def select(self, candidate: Candidate) -> Selection:
        started = time.perf_counter()
        outcome = await cross_encoder_rerank(
            self._client, [candidate.question], _single_query(candidate), keep=self._keep
        )
        if outcome.failure is not None:
            raise VariantError(outcome.failure)
        kept = outcome.result.per_query[0].chunks
        return Selection(
            kept=kept,
            took_ms=(time.perf_counter() - started) * 1000,
            detail={"ce_scores": [chunk.metadata.get("ce_score") for chunk in kept]},
        )


class LLMRerankVariant(Variant):
    name = "llm_rerank"
    uses_llm = True

    def __init__(
        self,
        models: GraphModels,
        *,
        rpm: float = DEFAULT_LLM_RPM,
        retry_wait_seconds: float = DEFAULT_LLM_RETRY_WAIT_SECONDS,
        retries: int = DEFAULT_LLM_RETRIES,
    ) -> None:
        if models.rerank is None:
            raise VariantError(
                "no RERANK/EXTRACTION model in the registry snapshot: "
                f"{models.rerank_unavailable or 'unknown reason'}"
            )
        self._models = models
        self._agent = build_llm_rerank_agent(models.rerank)
        self._pacer = RpmPacer(rpm)
        self._retry_wait = retry_wait_seconds
        self._retries = retries
        self._last_call_ms = 0.0

    async def judge(self, question: str, result: TurnRerankResult) -> list[RetrievedChunk]:
        """One `llm_rerank` call, paced to the credential's rpm; a rate-limit failure waits
        and retries (the wait is not part of the measured time)."""

        for attempt in range(1, self._retries + 2):
            await self._pacer.wait()
            started = time.perf_counter()
            outcome = await llm_rerank(
                self._agent,
                [question],
                result,
                credential=self._models.rerank_credential,
                purpose=self._models.rerank_purpose,
                snapshot_version=self._models.snapshot_version,
            )
            self._last_call_ms = (time.perf_counter() - started) * 1000
            if outcome.failure is None:
                return outcome.result.per_query[0].chunks[: settings.CHAT_CONTEXT_MAX_CHUNKS]
            if attempt <= self._retries and is_rate_limited(outcome.failure):
                logger.warning(
                    "LLM rerank rate-limited (attempt %d/%d), waiting %.0fs: %s",
                    attempt,
                    self._retries + 1,
                    self._retry_wait,
                    outcome.failure[:120],
                )
                await asyncio.sleep(self._retry_wait)
                continue
            raise VariantError(outcome.failure)
        raise VariantError("unreachable")

    async def select(self, candidate: Candidate) -> Selection:
        kept, took_ms = await self.timed_judge(candidate.question, _single_query(candidate))
        return Selection(kept=kept, took_ms=took_ms)

    async def timed_judge(
        self, question: str, result: TurnRerankResult
    ) -> tuple[list[RetrievedChunk], float]:
        """`judge` plus the time of its last attempt only (pacing and retry waits excluded)."""

        self._last_call_ms = 0.0
        kept = await self.judge(question, result)
        return kept, self._last_call_ms


class CrossEncoderThenLLMVariant(Variant):
    name = "cross_encoder_then_llm"
    uses_llm = True
    uses_service = True

    def __init__(self, cross_encoder: CrossEncoderVariant, llm: LLMRerankVariant) -> None:
        self._ce = cross_encoder
        self._llm = llm

    async def select(self, candidate: Candidate) -> Selection:
        first = await self._ce.select(candidate)
        narrowed = Candidate(
            question_id=candidate.question_id,
            category=candidate.category,
            question=candidate.question,
            chunks=first.kept,
            expected_document_ids=candidate.expected_document_ids,
        )
        kept, llm_ms = await self._llm.timed_judge(candidate.question, _single_query(narrowed))
        return Selection(
            kept=kept,
            took_ms=first.took_ms + llm_ms,
            detail={
                "ce_ms": round(first.took_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "ce_kept": len(first.kept),
            },
        )


async def build_variants(
    names: list[str],
    *,
    keep: int,
    client: CrossEncoderClient | None = None,
    models_loader: ModelsLoader | None = None,
    llm_rpm: float = DEFAULT_LLM_RPM,
    llm_retry_wait_seconds: float = DEFAULT_LLM_RETRY_WAIT_SECONDS,
    llm_retries: int = DEFAULT_LLM_RETRIES,
) -> list[Variant]:
    """Variants in `names` order. Graph models are only resolved when an LLM variant is
    asked for (that needs the backend's model registry)."""

    unknown = [name for name in names if name not in VARIANT_NAMES]
    if unknown:
        raise VariantError(f"unknown variant(s) {unknown}; choose from {list(VARIANT_NAMES)}")
    client = client or CrossEncoderClient()
    cross_encoder = CrossEncoderVariant(client, keep=keep)
    llm: LLMRerankVariant | None = None
    if any(name in LLM_VARIANTS for name in names):
        if models_loader is None:
            from app.api.deps import get_graph_models

            models_loader = get_graph_models
        llm = LLMRerankVariant(
            await models_loader(),
            rpm=llm_rpm,
            retry_wait_seconds=llm_retry_wait_seconds,
            retries=llm_retries,
        )
    built: list[Variant] = []
    for name in names:
        if name == "cross_encoder":
            built.append(cross_encoder)
        elif name == "llm_rerank":
            assert llm is not None
            built.append(llm)
        else:
            assert llm is not None
            built.append(CrossEncoderThenLLMVariant(cross_encoder, llm))
    return built
