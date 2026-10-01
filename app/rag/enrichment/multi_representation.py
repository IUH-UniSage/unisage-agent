from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker, RequestBudgetRejectedError
from app.core.config import settings
from app.core.errors.provider_errors import MalformedExtractionResponseError
from app.core.llm.provider_models import build_model
from app.core.registry import model_router
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import (
    ModelRegistryError,
    get_current_snapshot,
    require_top_priority_credential,
)
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.streaming import AttemptOutcome, BudgetContext, run_agent_text_with_failover
from app.rag.prompting.loader import get_templates
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class EnrichedChunk:
    """A chunk enriched with a summary and hypothetical questions for multi-representation
    search."""

    chunk: Chunk
    summary: str
    questions: list[str]


def _extract_json_object(raw_output: str) -> dict[str, Any] | None:
    """The output as-is, else its first `{...}` block (e.g. inside a code fence) - same
    tolerance `message_classification._load_json_object` gives CHAT's JSON output, needed
    here too now that nothing forces strict JSON mode the way OpenAI's `response_format=
    json_object` used to (a native, provider-agnostic call has no equivalent knob)."""

    candidate = raw_output.strip().strip("`").strip()
    match = _JSON_OBJECT_PATTERN.search(candidate)
    for text in (candidate, match.group(0) if match else None):
        if text is None:
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _parse_enrichment_output(
    raw_output: str, chunk: Chunk, question_count: int
) -> EnrichedChunk | None:
    """`None` for anything that doesn't parse into a usable result - malformed JSON, missing
    keys, or the wrong number of questions - never raises."""

    data = _extract_json_object(raw_output)
    if data is None:
        return None
    try:
        summary = str(data["summary"])
        questions = [str(question) for question in data["questions"]]
    except (KeyError, TypeError):
        return None
    if not summary or len(questions) != question_count:
        return None
    return EnrichedChunk(chunk=chunk, summary=summary, questions=questions)


@dataclass(frozen=True)
class MultiRepresentationEnricher:
    """Add a summary + hypothetical questions to a chunk via one LLM call.

    Model name, API key and base URL come from the model registry's ACTIVE EXTRACTION
    credential - never `.env`. Resolved lazily on first `enrich()`/`enrich_tracked()` call, not
    at construction time.

    Provider-agnostic: every attempt builds its `pydantic_ai.Agent` through
    `app.core.llm.provider_models.build_model()` - the SAME factory `get_graph_models()` (CHAT)
    and the embedding identity guard use - so an EXTRACTION credential on ANY provider that
    factory supports (`openai`, `google`, or a `SELF_HOSTED` OpenAI-compatible server) just
    works, with no provider-specific base-URL configuration needed. This replaced an earlier
    implementation that always spoke the OpenAI wire format directly regardless of
    `credential.provider` - which 404'd for a Google credential unless its `api_base_url` was
    pointed at Google's separate OpenAI-compatibility shim path instead of the native Gemini API.

    Credential selection and failover go through `app.graph.streaming.run_agent_text_with_failover`
    - the same shared machinery `MessageClassificationNode`/`QueryTransformationNode` use - not a
    second, home-grown loop. `model_router` is still consulted directly for the initial pick
    (`require_top_priority_credential`) and for the "malformed response from a fallback
    credential" escalation below, exactly like CHAT's own top-priority pick in
    `app.api.deps.get_graph_models()`.
    """

    model: Model | str | None = None
    question_count: int = field(default_factory=lambda: settings.INGEST_MULTI_REP_QUESTION_COUNT)

    def _build_agent(self, model: Model | str) -> Agent[None, str]:
        return Agent(
            model=model,
            system_prompt=get_templates().agent_multi_representation_enricher.format(
                question_count=self.question_count
            ),
        )

    def enrich(self, chunk: Chunk) -> EnrichedChunk:
        """Enrich one chunk.

        Test-injected `model` (a `pydantic_ai` `Model`/model-name string): skips the
        registry/router entirely, one call, no failover, no usage line - a malformed/short
        response logs a warning and falls back to an empty result rather than raising, so one
        bad chunk doesn't kill the whole embedding batch.

        Registry-resolved (the production path, `self.model is None`): one `enrich()` call is
        one purpose=EXTRACTION business request, self-contained - builds and closes its own
        `UsageRecorder`, same as `OpenAIEmbedder.embed()`. `app.worker.celery_app.embed_chunks`
        instead calls `enrich_tracked()` directly with a document-wide shared recorder.
        """

        if self.model is not None:
            agent = self._build_agent(self.model)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(agent.run(chunk.content))
            else:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, agent.run(chunk.content)).result()
            parsed = _parse_enrichment_output(result.output or "", chunk, self.question_count)
            if parsed is not None:
                return parsed
            return EnrichedChunk(chunk=chunk, summary="", questions=[])

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._enrich_recorded(chunk))
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._enrich_recorded(chunk)).result()

    async def _enrich_recorded(self, chunk: Chunk) -> EnrichedChunk:
        # One event loop and one Redis client for the whole call - an asyncio Redis
        # connection is bound to the loop that opened it (see OpenAIEmbedder.embed()).
        redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
        budget_tracker = BudgetTracker(redis_client=redis_client)
        recorder = UsageRecorder(
            request_id=str(uuid.uuid4()), purpose="EXTRACTION", budget_tracker=budget_tracker
        )
        status = "ERROR"
        try:
            reserve_result = await budget_tracker.reserve_request(
                request_id=recorder.request_id,
                purpose="EXTRACTION",
                estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)),
            )
            if reserve_result != "OK":
                raise RequestBudgetRejectedError("EXTRACTION", reserve_result)
            result = await self.enrich_tracked(chunk, recorder, budget_tracker)
            status = "SUCCESS"
            return result
        finally:
            # `NoAvailableCredentialError`/`NoBudgetAvailableError` (every
            # EXTRACTION credential exhausted) can propagate out of
            # `enrich_tracked` uncaught - `status` stays "ERROR" in that case.
            try:
                await recorder.close(status=status)
            finally:
                try:
                    await redis_client.aclose()
                except Exception:
                    logger.debug("enricher: closing the Redis client failed", exc_info=True)

    async def enrich_tracked(
        self,
        chunk: Chunk,
        usage_recorder: UsageRecorder,
        budget_tracker: BudgetTracker,
    ) -> EnrichedChunk:
        """Enrich within a CALLER-managed `UsageRecorder`/`BudgetTracker` - the ingestion
        pipeline's shared per-document recorder (`app.worker.celery_app.embed_chunks`), which
        reserves/settles the request-level budget once for the whole document rather than once
        per `enrich()` call. Registry-resolved only - there is no test-injected `model` bypass
        here, unlike `enrich()`.

        `NoAvailableCredentialError`/`NoBudgetAvailableError` (every EXTRACTION
        credential exhausted) propagates uncaught - deliberately NOT folded into the
        "log + return empty" path used for a single malformed-looking response. The caller
        (`app.worker.celery_app.embed_chunks`) already treats any exception from this method as a
        per-chunk `FAILED` result without aborting the rest of the batch.
        """

        try:
            credential = require_top_priority_credential("EXTRACTION")
        except ModelRegistryError as exc:
            raise NoAvailableCredentialError("EXTRACTION") from exc

        snapshot = get_current_snapshot()
        snapshot_version = snapshot.version if snapshot is not None else 0
        base_on_attempt = usage_recorder.bind("multi_representation_enrich")

        while True:
            model = build_model(credential)
            agent = self._build_agent(model)
            # Tracks which credential/attempt actually produced a response (success or not) -
            # `run_agent_text_with_failover` doesn't return this itself, and it's needed below to
            # tell "malformed on the PRIMARY credential" (fall back to empty) apart from
            # "malformed on a FALLBACK credential" (report + try yet another one).
            last_success: dict[str, Any] = {}

            def _on_attempt(
                outcome: AttemptOutcome, _store: dict[str, Any] = last_success
            ) -> Decimal:
                if outcome.status == "SUCCESS":
                    _store["credential"] = outcome.credential
                    _store["attempt"] = outcome.attempt
                return base_on_attempt(outcome)

            output = await run_agent_text_with_failover(
                agent,
                chunk.content,
                purpose="EXTRACTION",
                credential=credential,
                snapshot_version=snapshot_version,
                agent_factory=self._build_agent,
                on_attempt=_on_attempt,
                budget=BudgetContext(
                    tracker=budget_tracker,
                    request_id=usage_recorder.request_id,
                    reserve_seq=usage_recorder.reserve_budget_seq,
                    per_attempt_estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)),
                ),
            )
            parsed = _parse_enrichment_output(output, chunk, self.question_count)
            if parsed is not None:
                return parsed

            succeeded_credential = last_success.get("credential")
            succeeded_attempt = last_success.get("attempt", 0)
            if succeeded_credential is None or succeeded_attempt == 0:
                logger.warning(
                    "Malformed multi-representation response for chunk %s on primary "
                    "credential %s; falling back to empty.",
                    chunk.chunk_index,
                    credential.id,
                )
                return EnrichedChunk(chunk=chunk, summary="", questions=[])

            logger.warning(
                "Malformed multi-representation response for chunk %s on fallback "
                "credential %s; reporting as a permanent failure for that credential.",
                chunk.chunk_index,
                succeeded_credential.id,
            )
            malformed = MalformedExtractionResponseError(
                f"fallback credential {succeeded_credential.id!r} returned a malformed "
                f"multi-representation response for chunk {chunk.chunk_index}"
            )
            await model_router.record_failure(
                succeeded_credential,
                malformed,
                snapshot_version=snapshot_version,
                purpose="EXTRACTION",
            )
            try:
                credential = await model_router.get_next_credential("EXTRACTION")
            except NoAvailableCredentialError:
                raise NoAvailableCredentialError("EXTRACTION", last_error=malformed) from malformed
