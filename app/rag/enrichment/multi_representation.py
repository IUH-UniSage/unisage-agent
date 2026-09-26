import asyncio
import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from app.core import model_router
from app.core.config import settings
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.model_registry import CredentialConfig, get_current_snapshot
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


class MalformedExtractionResponseError(Exception):
    """Raised locally (never by a provider SDK) when a FALLBACK EXTRACTION credential's
    response fails the same summary/questions shape check as the primary credential -
    reported to `model_router.record_failure()` (todo.md Task 12, point 4) so it counts
    against that credential's circuit-breaker state instead of `enrich()` silently handing
    a malformed result to the ingest pipeline.

    Never raised for the PRIMARY (first) attempt - that case keeps the pre-existing
    "log + return empty" behavior (see `enrich()`'s docstring for why), since a malformed
    response from the one credential everyone is normally using is "this one chunk had a
    bad day", not evidence that credential is broken.

    Known gap, flagged rather than silently patched (todo.md Task 12 explicitly puts
    `app.core.llm_error_classifier` out of this task's scope - "call them, don't modify
    their logic"): `classify_llm_error()` has no `isinstance` branch for this exception
    type, so it falls through to its own documented "unrecognized -> TRANSIENT" default
    rather than todo.md's literal "PERMANENT" wording. The failure is still reported
    (Java gets a health ping, and the credential is put in cooldown) and - regardless of
    how it's classified - `enrich()`'s loop never retries the SAME credential again within
    one call, so the practical effect (move to a different credential) holds either way;
    only the *durability* of the exclusion (TTL length) is affected.
    """


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
            result = self._call_and_parse(chunk, model, client)
            if result is not None:
                return result
            logger.warning(
                "Malformed multi-representation response for chunk %s; falling back to empty.",
                chunk.chunk_index,
            )
            return EnrichedChunk(chunk=chunk, summary="", questions=[])

        is_fallback = False
        while True:
            credential = asyncio.run(model_router.get_next_credential("EXTRACTION"))
            snapshot = get_current_snapshot()
            snapshot_version = snapshot.version if snapshot is not None else 0
            resolved_model = credential.model_name or ""
            resolved_client = self._build_client(credential)

            try:
                result = self._call_and_parse(chunk, resolved_model, resolved_client)
            except Exception as exc:
                asyncio.run(
                    model_router.record_failure(credential, exc, snapshot_version=snapshot_version)
                )
                is_fallback = True
                continue

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
                )
            )
            is_fallback = True

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

    def _call_and_parse(self, chunk: Chunk, model: str, client: OpenAI) -> EnrichedChunk | None:
        """One provider call + response parse. Returns `None` for a malformed/short
        response (never raises for that - only a JSON/shape problem, not a call failure);
        a provider-call failure (network, auth, rate limit, ...) propagates as whatever
        exception the SDK/PydanticAI raised, uncaught here, for the caller to classify and
        report via `model_router`."""

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
        raw_content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(raw_content)
            summary = str(data["summary"])
            questions = [str(question) for question in data["questions"]]
            if not summary or len(questions) != self.question_count:
                raise ValueError("incomplete multi-representation response")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

        return EnrichedChunk(chunk=chunk, summary=summary, questions=questions)
