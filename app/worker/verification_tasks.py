"""Verify-before-active claim loop (plan.md "Verification lifecycle") — claims queued
verification jobs from backend-java, tries each candidate credential exactly once, and
reports the outcome back. Pull-based: there is no Java -> Python push, Python is always the
caller.

Everything in this module is plain async functions with no Celery dependency, so the whole
claim -> verify -> report cycle is testable without a broker or a running worker.
`app.worker.celery_app` wires `run_verification_batch()` into a Celery task (Beat schedule +
an immediate wake-up on a pub/sub message) — see that module for the task wrapper and the
distributed lock that keeps only one trigger running the loop at a time.

Secret hygiene: nothing here ever holds a claimed credential's API key longer than the
single job it belongs to, and it never crosses a function boundary that could turn it into a
Celery task argument or result. `safe_error_message()` is the only way a provider failure's
text reaches Java.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import redis
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.core.config import settings
from app.core.errors.llm_error_classifier import ErrorType, classify_llm_error
from app.core.errors.llm_failure import admin_failure_message
from app.core.llm.embedding_probe import EmbeddingFingerprint, measure_fingerprint
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.llm.provider_models import build_model
from app.core.observability.alerting import alert_credential_failure
from app.core.registry.model_registry import CredentialConfig
from app.integrations.backend_java_client import (
    BackendJavaClient,
    BackendJavaConnectionError,
    BackendJavaHTTPError,
)

logger = logging.getLogger(__name__)

_EMBEDDING_PURPOSE = "EMBEDDING"
# One minimal call, no internal retry - a claimed job that fails gets retried by Java
# re-queuing it (TRANSIENT) or ends FAILED (PERMANENT), never by looping in this function.
_PROVIDER_CALL_TIMEOUT_SECONDS = 15.0
_MINIMAL_COMPLETION_PROMPT = "Reply with a single short word."
_MINIMAL_COMPLETION_MAX_TOKENS = 512

# Submitting a result to Java is idempotent on Java's side (duplicate submissions of the same
# leaseToken are a no-op) - safe to retry a network failure with the exact same token.
_RESULT_SUBMIT_MAX_ATTEMPTS = 3
_RESULT_SUBMIT_RETRY_DELAY_SECONDS = 1.0

# Distributed lock so only one trigger (a Beat tick or a verification-requested pub/sub
# wake-up, across however many worker processes are subscribed) actually runs the claim loop
# at a time. Short TTL: Java's own `FOR UPDATE SKIP LOCKED` already makes concurrent claiming
# safe on its own, so this lock exists purely to avoid wasted duplicate `claim` calls, not for
# correctness - a lock that outlives a crashed holder self-heals within one TTL either way.
VERIFICATION_LOCK_KEY = "mr:verify:lock"
VERIFICATION_LOCK_TTL_SECONDS = 20


def try_acquire_verification_lock(redis_url: str | None = None) -> bool:
    """Best-effort `SET NX EX` lock. A Redis failure here degrades to "proceed without the
    lock" (returns `True`) rather than blocking verification entirely - correctness doesn't
    depend on this lock, only on Java's own fencing (see module docstring)."""

    try:
        conn = redis.Redis.from_url(redis_url or settings.REDIS_URL)
        try:
            return bool(
                conn.set(VERIFICATION_LOCK_KEY, "1", nx=True, ex=VERIFICATION_LOCK_TTL_SECONDS)
            )
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "verification loop: Redis lock unavailable, proceeding without it", exc_info=True
        )
        return True


def _credential_from_candidate(candidate: dict[str, Any]) -> CredentialConfig:
    """Builds a `CredentialConfig` from one claim response entry's `credential` object -
    same field names `app.core.registry.model_registry._parse_credential` reads off the snapshot,
    minus `revision` (a candidate under verification has none yet meaningful to attach to a
    health report, and nothing here sends one)."""

    return CredentialConfig(
        id=str(candidate["chatModelId"]),
        revision=0,
        source_type=str(candidate.get("sourceType") or ""),
        provider=candidate.get("provider"),
        model_name=candidate.get("modelName"),
        api_base_url=candidate.get("apiBaseUrl"),
        priority=None,
        max_rpm=candidate.get("maxRpm"),
        api_key=candidate.get("apiKey") or "",
    )


def _error_code_for(exc: Exception) -> str:
    """A short, non-secret-bearing code for the result's `errorCode` field - same shape
    `app.core.registry.model_router._error_code_for` uses for health reports."""

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return f"{type(exc).__name__}:{status_code}"
    return type(exc).__name__


async def _run_minimal_completion(credential: CredentialConfig) -> None:
    """One minimal CHAT/EXTRACTION completion call through the SSRF-pinned factory
    (`build_model` -> `build_provider_http_client`, never a raw client). Raises the
    provider's own exception (or `asyncio.TimeoutError`) on failure - callers classify it,
    this function doesn't."""

    model = build_model(credential)
    agent: Agent[None, str] = Agent(model=model)
    await asyncio.wait_for(
        agent.run(
            _MINIMAL_COMPLETION_PROMPT,
            model_settings=ModelSettings(max_tokens=_MINIMAL_COMPLETION_MAX_TOKENS, thinking=False),
        ),
        timeout=_PROVIDER_CALL_TIMEOUT_SECONDS,
    )


def _measure_embedding_fingerprint_sync(credential: CredentialConfig) -> EmbeddingFingerprint:
    """Embeds the 3 fixed probe sentences with the candidate EMBEDDING credential. Blocking -
    run this off the event loop (`asyncio.to_thread`).

    Dispatches on `credential.provider` the same way
    `app.rag.embeddings.provider.build_embedder()` does for an ACTIVE credential - a Google
    credential speaks Gemini's native `embedContent` API (not the OpenAI wire format every
    other provider/`SELF_HOSTED` server speaks). This can't just call `build_embedder()` itself:
    a candidate under verification isn't in the registry snapshot yet - there is nothing to
    resolve, the credential to probe is the one handed in directly.
    """

    if credential.provider == "google":
        return _measure_embedding_fingerprint_google_sync(credential)
    return _measure_embedding_fingerprint_openai_sync(credential)


def _measure_embedding_fingerprint_openai_sync(
    credential: CredentialConfig,
) -> EmbeddingFingerprint:
    """Through the same SSRF-pinned sync factory `app.rag.embeddings.openai_embedder.OpenAIEmbedder`
    uses."""

    client = OpenAI(
        api_key=credential.api_key,
        base_url=credential.api_base_url or None,
        timeout=_PROVIDER_CALL_TIMEOUT_SECONDS,
        http_client=build_provider_http_client_sync(
            ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
        ),
    )
    model_name = credential.model_name or ""

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model_name, input=texts)
        return [item.embedding for item in response.data]

    return measure_fingerprint(_embed)


def _measure_embedding_fingerprint_google_sync(
    credential: CredentialConfig,
) -> EmbeddingFingerprint:
    """Through the same SSRF-pinned sync factory + `embedContent` batching
    `app.rag.embeddings.google_embedder.GoogleEmbedder` uses."""

    from google.genai import Client
    from google.genai.types import HttpOptions

    from app.core.llm.http_client import build_provider_http_client
    from app.rag.embeddings.google_embedder import request_embeddings_sync

    connection_info = ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
    client = Client(
        vertexai=False,
        api_key=credential.api_key,
        http_options=HttpOptions(
            base_url=credential.api_base_url or None,
            httpx_client=build_provider_http_client_sync(connection_info),
            # Only the sync path is ever used here (this function is sync top to bottom), but
            # `google-genai`'s `Client.__init__` eagerly falls back to auto-building its own
            # aiohttp session for `.aio` whenever `httpx_async_client` is left `None` (see
            # `_api_client.py`'s `_use_aiohttp()`) - that fallback breaks in this environment
            # (`AttributeError: module 'aiohttp' has no attribute 'ClientSession'`, likely a
            # broken/shadowed aiohttp install). Supplying our own async client too (never called)
            # disables that fallback entirely - same as `GoogleEmbedder._resolve_from_registry()`
            # already does for the exact same reason.
            httpx_async_client=build_provider_http_client(connection_info),
        ),
    )
    model_name = credential.model_name or ""

    def _embed(texts: list[str]) -> list[list[float]]:
        vectors, _ = request_embeddings_sync(client, model_name, texts)
        return vectors

    return measure_fingerprint(_embed)


async def _submit_result(
    client: BackendJavaClient,
    *,
    job_id: str,
    lease_token: str,
    result_type: str,
    error_code: str | None = None,
    message: str | None = None,
    embedding_dimension: int | None = None,
    embedding_fingerprint: list[float] | None = None,
) -> None:
    """Submits one job's result, retrying a network failure up to
    `_RESULT_SUBMIT_MAX_ATTEMPTS` times with the exact same `lease_token` (safe: Java's result
    endpoint is idempotent on that token). A `409` (lease lost - reclaimed by another worker,
    or the job was superseded/cancelled meanwhile) is logged at info level and this returns
    without retrying, per plan.md "Verification lifecycle"."""

    last_exc: Exception | None = None
    for attempt in range(1, _RESULT_SUBMIT_MAX_ATTEMPTS + 1):
        try:
            await client.post_verification_result(
                job_id=job_id,
                lease_token=lease_token,
                result_type=result_type,  # type: ignore[arg-type]
                error_code=error_code,
                message=message,
                embedding_dimension=embedding_dimension,
                embedding_fingerprint=embedding_fingerprint,
            )
            return
        except BackendJavaHTTPError as exc:
            if exc.status_code == 409:
                logger.info(
                    "verification job %s: lease lost submitting %s result - lease was "
                    "reclaimed or the job was superseded/cancelled, moving on",
                    job_id,
                    result_type,
                )
                return
            logger.warning(
                "verification job %s: result submission rejected with HTTP %s",
                job_id,
                exc.status_code,
            )
            return
        except BackendJavaConnectionError as exc:
            last_exc = exc
            if attempt < _RESULT_SUBMIT_MAX_ATTEMPTS:
                logger.warning(
                    "verification job %s: network error submitting result (attempt %d/%d), "
                    "retrying with the same lease token",
                    job_id,
                    attempt,
                    _RESULT_SUBMIT_MAX_ATTEMPTS,
                )
                await asyncio.sleep(_RESULT_SUBMIT_RETRY_DELAY_SECONDS)
                continue

    logger.error(
        "verification job %s: giving up submitting %s result after %d attempts",
        job_id,
        result_type,
        _RESULT_SUBMIT_MAX_ATTEMPTS,
        exc_info=last_exc,
    )


async def _verify_one_job(job: dict[str, Any], *, client: BackendJavaClient) -> None:
    job_id = str(job["jobId"])
    lease_token = str(job["leaseToken"])
    attempt = job.get("attempt")
    candidate = job["credential"]
    credential = _credential_from_candidate(candidate)
    purpose = candidate.get("modelPurpose")

    logger.info(
        "verification job %s: verifying chat_model %s (%s, provider=%s, model=%s, attempt %s)",
        job_id,
        credential.id,
        purpose,
        credential.provider,
        credential.model_name,
        attempt,
    )

    try:
        if purpose == _EMBEDDING_PURPOSE:
            fingerprint = await asyncio.to_thread(_measure_embedding_fingerprint_sync, credential)
            await _submit_result(
                client,
                job_id=job_id,
                lease_token=lease_token,
                result_type="OK",
                embedding_dimension=fingerprint.dimension,
                embedding_fingerprint=fingerprint.flattened(),
            )
        else:
            await _run_minimal_completion(credential)
            await _submit_result(client, job_id=job_id, lease_token=lease_token, result_type="OK")
        logger.info(
            "verification job %s: chat_model %s attempt %s succeeded",
            job_id,
            credential.id,
            attempt,
        )
    except Exception as exc:
        error_type = classify_llm_error(exc)
        result_type = "TRANSIENT" if error_type is ErrorType.TRANSIENT else "PERMANENT"
        message = admin_failure_message(
            exc, purpose=purpose, api_key=credential.api_key, credential=credential
        )
        logger.warning(
            "verification job %s: chat_model %s attempt %s failed (%s) - %s: %s",
            job_id,
            credential.id,
            attempt,
            result_type,
            _error_code_for(exc),
            message,
        )
        if result_type == "PERMANENT":
            # PERMANENT always ends this job FAILED on the Java side (plan.md's
            # verification lifecycle table), so Python knows for certain at this point
            # that the job is about to fail - alert now rather than polling Java after
            # the fact.
            #
            # A TRANSIENT result is deliberately NOT alerted here even when this was the
            # job's last attempt (which would also end it FAILED): this claim response
            # carries `attempt` but not `maxAttempts` (see
            # `InternalVerificationClaimResponse` on the Java side), so Python cannot
            # tell "TRANSIENT, Java will re-queue it" apart from "TRANSIENT, Java is
            # about to mark it FAILED" without guessing. Guessing wrong in the alerting
            # direction would flood Slack for a routine, still-retrying transient
            # failure - exactly what this feature exists to avoid - so this case is left
            # unalerted. Closing this gap needs either `maxAttempts` added to the claim
            # response, or Java alerting itself at the point it makes that FAILED
            # transition.
            await alert_credential_failure(
                credential, "VERIFICATION_FAILED_PERMANENT", message, purpose=purpose
            )
        await _submit_result(
            client,
            job_id=job_id,
            lease_token=lease_token,
            result_type=result_type,
            error_code=_error_code_for(exc),
            message=message,
        )


async def run_verification_batch(
    *, client: BackendJavaClient | None = None, limit: int | None = None
) -> int:
    """Claims up to `limit` queued/lease-expired verification jobs and verifies each one in
    turn. Returns how many jobs were claimed (0 when there was nothing to do).

    One job's failure to verify (or to submit its result) never aborts the batch - each job
    is fully independent, so `_verify_one_job` swallows what it needs to and this loop just
    moves to the next entry.
    """

    resolved_client = client if client is not None else BackendJavaClient()
    resolved_limit = (
        limit if limit is not None else settings.MODEL_REGISTRY_VERIFICATION_CLAIM_LIMIT
    )

    jobs = await resolved_client.claim_verifications(limit=resolved_limit)
    if jobs:
        logger.info(
            "verification batch: claimed %d job(s): %s",
            len(jobs),
            ", ".join(
                f"{job.get('jobId')}(chat_model={job.get('credential', {}).get('chatModelId')}, "
                f"attempt={job.get('attempt')})"
                for job in jobs
            ),
        )
    for job in jobs:
        try:
            await _verify_one_job(job, client=resolved_client)
        except Exception:
            logger.exception(
                "verification job %s: unhandled failure verifying candidate credential",
                job.get("jobId"),
            )
    return len(jobs)
