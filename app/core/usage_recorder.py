"""Per-request usage measurement + outbox handoff - Cost Tracking plan.md Task 6.

One instance per Chat request, created in `chat.py` and threaded through the
graph as an explicit parameter (same pattern as `GraphTrace`/`trace`) - never
global mutable state, so concurrent requests never share or clobber each
other's line list. `bind(node_name)` is what each graph node call site passes
as `on_attempt` to `classify_intent`/`transform_tasks`/`run_ticket_fallback`/
`run_generation_synthesis` (see `streaming_graph.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.core.cost_calculator import (
    COST_STATUS_FREE,
    COST_STATUS_PRICED,
    COST_STATUS_UNPRICED,
    calculate_actual,
)
from app.graph.streaming import AttemptOutcome, AttemptRecorder

logger = logging.getLogger(__name__)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _decimal_str(value: Decimal) -> str:
    return format(value, "f")


@dataclass
class UsageRecorder:
    request_id: str
    purpose: str
    conversation_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    user_id: str | None = None
    guest_ip: str | None = None

    _lines: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _started_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def bind(self, node_name: str) -> AttemptRecorder:
        """An `on_attempt` callback pre-bound with this node's display name - the
        `nodeName` field on every line it produces (e.g. `GenerationSynthesisNode`,
        not the numbered trace label `10_GenerationSynthesisNode` - see
        docs/product/DECISIONS.md's node-naming rule)."""

        def _on_attempt(outcome: AttemptOutcome) -> None:
            self._record_attempt(node_name, outcome)

        return _on_attempt

    def _record_attempt(self, node_name: str, outcome: AttemptOutcome) -> None:
        # Called synchronously from inside streaming.py's own try/except, which
        # already wraps this in a catch-all - but a second layer here means a bug
        # in THIS method can never propagate into the failover/streaming logic
        # that called it, only lose the one line it was recording.
        try:
            self._lines.append(self._build_line(node_name, outcome))
        except Exception:
            logger.exception(
                "UsageRecorder failed to record an attempt for node=%s - this line is LOST, "
                "not retried (the provider call itself already happened)",
                node_name,
            )

    def _build_line(self, node_name: str, outcome: AttemptOutcome) -> dict[str, Any]:
        credential = outcome.credential
        occurred_at = _iso_z(datetime.now(UTC))

        if credential is None:
            # No credential known at all (e.g. failed before any was resolved) -
            # nothing to price, nothing to snapshot.
            return {
                "seq": len(self._lines),
                "nodeName": node_name,
                "attempt": outcome.attempt,
                "chatModelId": None,
                "provider": None,
                "modelName": None,
                "sourceType": None,
                "inputTokens": 0,
                "outputTokens": 0,
                "cachedTokens": 0,
                "costUsd": None,
                "estimatedCostUsd": "0",
                "costStatus": COST_STATUS_UNPRICED,
                "latencyMs": outcome.latency_ms,
                "status": outcome.status,
                "errorCode": outcome.error_code,
                "occurredAt": occurred_at,
            }

        usage = outcome.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        cached_tokens = (usage.cache_read_tokens or 0) if usage else 0

        if outcome.status == "SUCCESS" and credential.model_name:
            result = calculate_actual(
                model_name=credential.model_name,
                source_type=credential.source_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
            estimated_cost_usd = result.estimated_cost_usd
            cost_status = result.cost_status
            # The DB CHECK on the Java side requires costUsd IS NULL for every
            # status except PRICED - cost_calculator's FREE result carries a
            # real Decimal("0") (a true statement about the dollar amount), but
            # the wire payload must null it out for anything that isn't PRICED.
            cost_usd = _decimal_str(result.cost_usd) if cost_status == COST_STATUS_PRICED else None
        else:
            # A failed attempt spent no priceable tokens (or there is no model
            # name to look up) - FREE for a self-hosted credential, UNPRICED
            # otherwise, matching cost_calculator's own SELF_HOSTED short-circuit.
            cost_status = (
                COST_STATUS_FREE
                if credential.source_type == "SELF_HOSTED"
                else COST_STATUS_UNPRICED
            )
            cost_usd = None
            estimated_cost_usd = Decimal("0")

        return {
            "seq": len(self._lines),
            "nodeName": node_name,
            "attempt": outcome.attempt,
            "chatModelId": credential.id,
            "provider": credential.provider,
            "modelName": credential.model_name,
            "sourceType": credential.source_type,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cachedTokens": cached_tokens,
            "costUsd": cost_usd,
            "estimatedCostUsd": _decimal_str(estimated_cost_usd),
            "costStatus": cost_status,
            "latencyMs": outcome.latency_ms,
            "status": outcome.status,
            "errorCode": outcome.error_code,
            "occurredAt": occurred_at,
        }

    async def close(self, *, status: str) -> None:
        """Call exactly once when the request ends - success, error, or client
        disconnect (plan.md Task 6). Idempotent: a second call is a no-op, so a
        caller that closes defensively in more than one place can never double-send.

        No lines recorded (no provider call was ever made this request) -> settle
        only, nothing enqueued, matching plan.md's "Request không có LLM call"
        rule. `status` is the graph's own outcome (SUCCESS/ERROR); it is
        downgraded to PARTIAL here when at least one line failed but the graph
        still produced SUCCESS overall (a failover recovered from it).
        """

        if self._closed:
            return
        self._closed = True

        if not self._lines:
            return

        has_error_line = any(line["status"] == "ERROR" for line in self._lines)
        final_status = "PARTIAL" if status == "SUCCESS" and has_error_line else status

        finished_at = datetime.now(UTC)
        payload = {
            "requestId": self.request_id,
            "purpose": self.purpose,
            "conversationId": self.conversation_id,
            "userMessageId": self.user_message_id,
            "assistantMessageId": self.assistant_message_id,
            "userId": self.user_id,
            "guestIp": self.guest_ip,
            "status": final_status,
            "startedAt": _iso_z(self._started_at),
            "finishedAt": _iso_z(finished_at),
            "lines": self._lines,
        }

        from app.core.usage_outbox import (
            enqueue_usage_payload,  # local import: avoids a cycle at module load
        )

        await enqueue_usage_payload(payload)
