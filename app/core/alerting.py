"""Slack alerting for the model registry's failure conditions (plan.md's alerting
story) — the single place `send_slack_alert()` gets called from.

Trigger conditions (each wired at the module that already knows the failure
happened, never by adding new polling elsewhere):

- Every provider-call failure recorded by `model_router.record_failure()` — both
  PERMANENT (the credential just got excluded by the circuit breaker) and TRANSIENT
  (a 503/high-demand/rate-limit blip that the circuit breaker will retry after a
  cooldown). This used to be PERMANENT-only; changed by product decision (spec.md
  "Slack alerting") because a human should hear about a provider outage the first
  time it happens, not only after the circuit breaker gives up on the credential
  entirely — the 15-minute debounce below is what keeps this from flooding the
  channel during a sustained blip, not withholding the alert in the first place.
- `model_router.NoAvailableCredentialError` — every credential for a purpose is
  cooling down or excluded.
- An embedding job aborted by `EmbeddingProviderError`/`EmbeddingIdentityMismatchError`
  escaping `embed_chunks`'s per-chunk loop — that failure mode never auto-fails-over,
  so the job ends FAILED unconditionally.
- A verify-before-active job whose result is PERMANENT — that combination always ends
  the job FAILED on the Java side (see `app.worker.verification_tasks` for the
  documented gap: a TRANSIENT result cannot be alerted from here, because Java's claim
  response never tells Python `maxAttempts`, only the current `attempt`).

**Debounce**: same alert scope (a specific credential, or a purpose when there is no
single credential to blame — the exhaustion case) + same incident type → at most one
alert every 15 minutes, via a Redis `SET NX EX` marker key (`mr:alert:<scope>:<incident
type>`). Many concurrent callers all hitting the same failure at once collapse to one
`SET NX` winner exactly like `model_router`'s circuit breaker does.

**Redis-down policy**: alert anyway. Every other Redis-dependent piece of this codebase
(`model_router`'s circuit breaker, `registry_subscriber`'s hot-reload, the verification
loop's distributed lock) degrades by *proceeding without the guard* rather than
blocking/failing the caller — the same "degrade, don't crash" posture applies here, but
the failure mode of skipping debounce is "maybe a duplicate Slack message", whereas the
failure mode of skipping the alert entirely is total silence during exactly the kind of
outage a human most needs to hear about. Between those two, a duplicate message is the
better failure.

Every message is redacted (`app.core.redaction.redact`) right before it is handed to
`send_slack_alert()`, passing the credential's own API key as the known secret — defense
in depth even though `reason` should already have gone through `safe_error_message()`
at the call site, matching how the Java side re-redacts `error_message` on its own end
despite Python already having redacted it once.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.core.model_registry import CredentialConfig
from app.core.redaction import redact
from app.integrations.slack_notifier import send_slack_alert

logger = logging.getLogger(__name__)

_KEY_PREFIX = "mr:alert"
_DEBOUNCE_SECONDS = 15 * 60


class _RedisLike(Protocol):
    """Structural subset of `redis.asyncio.Redis` this module actually calls — same
    trick `model_router._RedisLike` uses to let tests hand in a bare fake."""

    async def set(
        self, name: str, value: Any, *, nx: bool = False, ex: int | None = None
    ) -> Any: ...

    async def aclose(self) -> Any: ...


def _debounce_key(scope_id: str, incident_type: str) -> str:
    return f"{_KEY_PREFIX}:{scope_id}:{incident_type}"


def _credential_label(credential: CredentialConfig) -> str:
    """Nickname + id when there's a nickname, else just the id - a bare UUID
    in a Slack message forces whoever reads it to go cross-reference the
    admin UI just to know which credential failed."""

    return f"{credential.display_name} ({credential.id})" if credential.display_name else credential.id


def _scope_id(credential: CredentialConfig | None, purpose: str | None) -> str:
    """A credential's own id when there is one; otherwise the purpose (the
    `NoAvailableCredentialError` case has no single credential to blame)."""

    if credential is not None:
        return credential.id
    return f"purpose:{purpose or 'unknown'}"


async def _should_alert(
    scope_id: str, incident_type: str, *, redis_client: _RedisLike | None
) -> bool:
    """`True` iff this call should actually send the alert — either the debounce
    `SET NX` won (key was absent), or Redis was unreachable and this degrades to
    "alert anyway" (see module docstring for why)."""

    key = _debounce_key(scope_id, incident_type)

    if redis_client is not None:
        try:
            acquired = await redis_client.set(key, "1", nx=True, ex=_DEBOUNCE_SECONDS)
            return bool(acquired)
        except Exception:
            logger.warning(
                "alerting: Redis unavailable checking debounce key %s - alerting anyway",
                key,
                exc_info=True,
            )
            return True

    try:
        conn = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    except Exception:
        logger.warning(
            "alerting: Redis unavailable checking debounce key %s - alerting anyway",
            key,
            exc_info=True,
        )
        return True

    try:
        acquired = await conn.set(key, "1", nx=True, ex=_DEBOUNCE_SECONDS)
        return bool(acquired)
    except Exception:
        logger.warning(
            "alerting: Redis unavailable checking debounce key %s - alerting anyway",
            key,
            exc_info=True,
        )
        return True
    finally:
        try:
            await conn.aclose()
        except Exception:  # pragma: no cover - best-effort cleanup only
            pass


_ALERT_COLOR = "#E01E5A"  # Slack's own "danger" red - every incident type this
# module sends is a failure condition worth a red card, so this isn't varied
# per incident_type (see the module docstring's "Trigger conditions" list).


def _format_message(
    credential: CredentialConfig | None,
    incident_type: str,
    reason: str,
    purpose: str | None,
) -> str:
    """Plain-text fallback for notification previews/screen readers - Slack
    requires a top-level `text` even when `blocks`/`attachments` carry the
    actual rendered card (see `_build_slack_payload`)."""

    lines = [f"[Model Registry] {incident_type}"]
    if purpose:
        lines.append(f"Purpose: {purpose}")
    if credential is not None:
        lines.append(f"Provider: {credential.provider or 'unknown'}")
        lines.append(f"Model: {credential.model_name or 'unknown'}")
        lines.append(f"Credential: {_credential_label(credential)}")
    lines.append(f"Reason: {reason}")
    lines.append(f"Time: {datetime.now(UTC).isoformat()}")
    return "\n".join(lines)


def _build_slack_payload(
    credential: CredentialConfig | None,
    incident_type: str,
    reason: str,
    purpose: str | None,
) -> dict[str, Any]:
    """A Block Kit card wrapped in a colored `attachments` entry (the classic
    attachment's `color` bar is still the only way to get a colored left
    border on a Block Kit card) - each field on its own line instead of one
    run-on paragraph, matching the plan's "khó phân từng message dính nhau"
    fix request."""

    fields = []
    if purpose:
        fields.append({"type": "mrkdwn", "text": f"*Purpose:*\n{purpose}"})
    if credential is not None:
        fields.append({"type": "mrkdwn", "text": f"*Provider:*\n{credential.provider or 'unknown'}"})
        fields.append({"type": "mrkdwn", "text": f"*Model:*\n{credential.model_name or 'unknown'}"})
        fields.append({"type": "mrkdwn", "text": f"*Credential:*\n{_credential_label(credential)}"})
    fields.append(
        {"type": "mrkdwn", "text": f"*Time:*\n{datetime.now(UTC).isoformat()}"}
    )

    return {
        "text": _format_message(credential, incident_type, reason, purpose),
        "attachments": [
            {
                "color": _ALERT_COLOR,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"⚠️ Model Registry — {incident_type}",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": f"`{reason}`"}],
                    },
                    {"type": "divider"},
                    {"type": "section", "fields": fields},
                ],
            }
        ],
    }


async def alert_credential_failure(
    credential: CredentialConfig | None,
    incident_type: str,
    reason: str,
    *,
    purpose: str | None = None,
    redis_client: _RedisLike | None = None,
) -> None:
    """Sends (at most one, per 15-minute debounce window) a Slack alert for one
    credential/purpose incident. Never raises — a failure to alert must never take
    down whatever caught the actual failure and called this.

    Args:
        credential: the credential the incident happened to, or `None` for the
            "every credential for a purpose is exhausted" case, which has no single
            credential to blame.
        incident_type: a short, stable string identifying the kind of incident
            (e.g. `"PERMANENT"`, `"TRANSIENT"`, `"NO_AVAILABLE_CREDENTIAL"`,
            `"EMBEDDING_PROVIDER_FAILURE"`, `"VERIFICATION_FAILED_PERMANENT"`) — this
            is half of the debounce key, so the same credential failing two
            different ways alerts independently for each.
        reason: human-readable cause, already redacted by the caller
            (`safe_error_message()`); redacted again here regardless.
        purpose: `CHAT`/`EMBEDDING`/`EXTRACTION` when known — included in the
            message and used as the debounce scope when `credential` is `None`.
        redis_client: test seam — anything satisfying `_RedisLike` in place of a
            real Redis connection.
    """

    try:
        scope_id = _scope_id(credential, purpose)
        if not await _should_alert(scope_id, incident_type, redis_client=redis_client):
            return

        known_secret = credential.api_key if credential is not None else None
        redacted_reason = redact(reason, known_secret)
        payload = _build_slack_payload(credential, incident_type, redacted_reason, purpose)
        await send_slack_alert(payload)
    except Exception:
        logger.warning(
            "alert_credential_failure: failed to send alert for incident_type=%s",
            incident_type,
            exc_info=True,
        )
