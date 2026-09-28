import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from openai import OpenAI

from app.core import model_router
from app.core.config import settings
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.llm_error_classifier import MalformedExtractionResponseError
from app.core.model_registry import CredentialConfig, get_current_snapshot
from app.core.usage_recorder import UsageRecorder
from app.rag.prompting.loader import get_templates
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichedChunk:
    """A chunk enriched with a summary and hypothetical questions for multi-representation
    search."""

    chunk: Chunk
    summary: str
    questions: list[str]


@dataclass(frozen=True)
class MultiRepresentationEnricher:
    """Add a summary + hypothetical questions to a chunk via one LLM call.

    Model name, API key and base URL come from the model registry's ACTIVE EXTRACTION
    credential (plan.md "Cutover khỏi cấu hình `.env` tĩnh") - never `.env`. Both are resolved
    lazily on first `enrich()` call, not at construction time - mirrors `OpenAIEmbedder`; tests
    inject `model`/`client` directly to skip the registry (and `model_router`) entirely (see
    tests/test_multi_representation.py).

    When resolving from the registry, credential selection and failover go through
    `app.core.model_router` (todo.md Task 10) - the same circuit breaker Task 11 wired CHAT
    streaming into - never a second, home-grown cooldown/exclusion mechanism. Every attempt
    (primary and every fallback) builds its `OpenAI` client through the same SSRF-guarded
    `build_provider_http_client_sync()` factory.
    """

    model: str | None = None
    question_count: int = field(default_factory=lambda: settings.INGEST_MULTI_REP_QUESTION_COUNT)
    client: OpenAI | None = None

    def enrich(self, chunk: Chunk) -> EnrichedChunk:
        """Enrich one chunk.

        Test-injected `model`/`client` (both set): skips the registry/router entirely, one
        call, no failover - a malformed/short response logs a warning and falls back to an
        empty result rather than raising, so one bad chunk doesn't kill the whole embedding
        batch.

        Registry-resolved (the production path): each attempt picks the next EXTRACTION
        credential via `model_router.get_next_credential()`. A provider-call failure at any
        point reports it via `model_router.record_failure()` (with the credential + snapshot
        version captured at the moment THAT credential was selected, never re-read after the
        fact) and moves to the next credential - this is a plain call-then-parse-JSON flow,
        not a stream, so there is no "before vs after first chunk" boundary to preserve, unlike
        `stream_agent_text()`'s CHAT failover.

        A malformed/short response on the FIRST (primary) attempt keeps the original
        "log + return empty" behavior. The SAME malformed response on a FALLBACK attempt is
        different: silently handing a second bad-looking result to the ingest pipeline would
        hide that the fallback credential itself might be broken, so it's reported to
        `model_router.record_failure()` as `MalformedExtractionResponseError` (see that class's
        docstring for the one open gap here) and the loop moves to yet another credential.

        `model_router.NoAvailableCredentialError` (every EXTRACTION credential exhausted)
        propagates uncaught - deliberately NOT folded into the "log + return empty" path.
        That path exists for a single chunk's response happening to be malformed; total
        credential exhaustion is a different kind of problem entirely (a real provider
        outage, not this chunk's bad luck), and it needs to surface loudly rather than let
        every remaining chunk in the batch silently degrade to placeholder embeddings. The
        caller (`app.worker.celery_app`'s `embed_chunks_task`) already treats any exception
        from `enrich()` as a per-chunk `FAILED` result without aborting the rest of the
        batch, so propagating here is safe at the job level too.
        """

        model = self.model
        client = self.client
        if model is not None and client is not None:
            # Test-injected model/client, no registry credential to snapshot - matches
            # OpenAIEmbedder's identical precedent, no usage line at all here.
            result, _usage = self._call_and_parse(chunk, model, client)
            if result is not None:
                return result
            logger.warning(
                "Malformed multi-representation response for chunk %s; falling back to empty.",
                chunk.chunk_index,
            )
            return EnrichedChunk(chunk=chunk, summary="", questions=[])

        # Cost Tracking plan.md Task 8: one enrich() call is one purpose=EXTRACTION
        # business request, one line per attempt (including a failed attempt before
        # failover) - self-contained like OpenAIEmbedder.embed(), so
        # app.worker.celery_app's caller needs no changes.
        recorder = UsageRecorder(request_id=str(uuid.uuid4()), purpose="EXTRACTION")
        status = "ERROR"
        try:
            result = self._enrich_with_failover(chunk, recorder)
            status = "SUCCESS"
            return result
        finally:
            # `model_router.NoAvailableCredentialError` (every EXTRACTION credential
            # exhausted) can propagate out of `_enrich_with_failover` uncaught (see this
            # method's own docstring) - `status` stays "ERROR" in that case, same as it
            # would for any other exception escaping this method.
            asyncio.run(recorder.close(status=status))

    def _enrich_with_failover(self, chunk: Chunk, recorder: UsageRecorder) -> EnrichedChunk:
        is_fallback = False
        attempt = 0
        while True:
            credential = asyncio.run(model_router.get_next_credential("EXTRACTION"))
            snapshot = get_current_snapshot()
            snapshot_version = snapshot.version if snapshot is not None else 0
            resolved_model = credential.model_name or ""
            resolved_client = self._build_client(credential)

            started_at = time.monotonic()
            try:
                result, usage = self._call_and_parse(chunk, resolved_model, resolved_client)
            except Exception as exc:
                recorder.record(
                    node_name="multi_representation_enrich",
                    attempt=attempt,
                    credential=credential,
                    status="ERROR",
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                    error_code=type(exc).__name__,
                )
                asyncio.run(
                    model_router.record_failure(
                        credential, exc, snapshot_version=snapshot_version, purpose="EXTRACTION"
                    )
                )
                is_fallback = True
                attempt += 1
                continue

            latency_ms = int((time.monotonic() - started_at) * 1000)
            # The provider call itself succeeded (tokens were spent) even when the
            # response body turns out malformed below - that's a data-quality
            # problem, not a call failure, so this line is always SUCCESS.
            recorder.record(
                node_name="multi_representation_enrich",
                attempt=attempt,
                credential=credential,
                status="SUCCESS",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
            )

            if result is not None:
                return result

            if not is_fallback:
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
                credential.id,
            )
            asyncio.run(
                model_router.record_failure(
                    credential,
                    MalformedExtractionResponseError(
                        f"fallback credential {credential.id!r} returned a malformed "
                        f"multi-representation response for chunk {chunk.chunk_index}"
                    ),
                    snapshot_version=snapshot_version,
                    purpose="EXTRACTION",
                )
            )
            is_fallback = True
            attempt += 1

    def _build_client(self, credential: CredentialConfig) -> OpenAI:
        """Builds the `OpenAI` client for one attempt (primary or fallback) - always through
        the SSRF-guarded factory, so the Task 0.6 architecture test
        (`tests/core/test_no_raw_provider_clients.py`) stays green for every fallback attempt
        too, not just the first one."""

        return OpenAI(
            api_key=credential.api_key,
            base_url=credential.api_base_url or None,
            http_client=build_provider_http_client_sync(
                ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
            ),
        )

    def _call_and_parse(
        self, chunk: Chunk, model: str, client: OpenAI
    ) -> tuple[EnrichedChunk | None, object | None]:
        """One provider call + response parse. Returns `(None, usage)` for a
        malformed/short response (never raises for that - only a JSON/shape
        problem, not a call failure - but `usage` is still real, tokens were
        still spent); a provider-call failure (network, auth, rate limit, ...)
        propagates as whatever exception the SDK/PydanticAI raised, uncaught
        here, for the caller to classify and report via `model_router`."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": get_templates().agent_multi_representation_enricher.format(
                        question_count=self.question_count
                    ),
                },
                {"role": "user", "content": chunk.content},
            ],
            response_format={"type": "json_object"},
        )
        usage = response.usage
        raw_content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(raw_content)
            summary = str(data["summary"])
            questions = [str(question) for question in data["questions"]]
            if not summary or len(questions) != self.question_count:
                raise ValueError("incomplete multi-representation response")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None, usage

        return EnrichedChunk(chunk=chunk, summary=summary, questions=questions), usage
