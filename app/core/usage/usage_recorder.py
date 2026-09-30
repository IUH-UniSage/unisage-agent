"""Per-request usage measurement + outbox handoff.

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
from typing import TYPE_CHECKING, Any

from app.core.registry.model_registry import CredentialConfig
from app.core.usage.cost_calculator import (
    COST_STATUS_FREE,
    COST_STATUS_PRICED,
    COST_STATUS_UNPRICED,
    calculate_actual,
)
from app.graph.streaming import AttemptOutcome, AttemptRecorder

if TYPE_CHECKING:
    from app.core.budget.tracker import BudgetTracker

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
    document_id: str | None = None
    budget_tracker: BudgetTracker | None = None

    _lines: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _started_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _next_budget_seq: int = field(default=0, init=False, repr=False)

    def reserve_budget_seq(self) -> int:
        """A monotonic counter independent of `_lines`' own `seq` numbering - used as
        the Redis hash field disambiguator for one provider-budget acquire/release
        pair. Must be called once per attempt, before that attempt's
        `acquire_provider` call, so every attempt across every node in this request
        gets its own field even though each node's `attempt_index` restarts at 0."""

        seq = self._next_budget_seq
        self._next_budget_seq += 1
        return seq

    def bind(self, node_name: str) -> AttemptRecorder:
        """An `on_attempt` callback pre-bound with this node's display name - the
        `nodeName` field on every line it produces (e.g. `GenerationSynthesisNode`,
        not the numbered trace label `10_GenerationSynthesisNode` - see
        docs/product/DECISIONS.md's node-naming rule). Returns the line's committed
        dollar amount (PRICED cost, else estimated, else 0) so a caller wiring budget
        reservation can release the matching PROVIDER-scope reservation with it."""

        def _on_attempt(outcome: AttemptOutcome) -> Decimal:
            return self._record_attempt(node_name, outcome)

        return _on_attempt

    def _record_attempt(self, node_name: str, outcome: AttemptOutcome) -> Decimal:
        # Called synchronously from inside streaming.py's own try/except, which
        # already wraps this in a catch-all - but a second layer here means a bug
        # in THIS method (including reading `outcome.usage`'s fields, which is why
        # this whole method has its own try/except rather than relying on
        # `record()`'s - that one starts too late to protect this extraction step)
        # can never propagate into the failover/streaming logic that called it,
        # only lose the one line it was recording.
        try:
            usage = outcome.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            cached_tokens = (usage.cache_read_tokens or 0) if usage else 0
        except Exception:
            logger.exception(
                "UsageRecorder failed to read usage fields for node=%s - this line is LOST",
                node_name,
            )
            return Decimal("0")

        return self.record(
            node_name=node_name,
            attempt=outcome.attempt,
            credential=outcome.credential,
            status=outcome.status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            latency_ms=outcome.latency_ms,
            error_code=outcome.error_code,
        )

    def record(
        self,
        *,
        node_name: str,
        attempt: int,
        credential: CredentialConfig | None,
        status: str,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        error_code: str | None = None,
    ) -> Decimal:
        """Records one line directly from plain token counts - used by Embedding/
        Extraction, which call the OpenAI SDK directly rather than through
        `streaming.py`'s `on_attempt`/`AttemptOutcome` (PydanticAI-only machinery
        `bind()` wraps). `_record_attempt()` above is the PydanticAI path; this is
        the raw path both it and Embedding/Extraction's call sites end up funneling
        into.

        Never raises - a bug here must not take down the enrichment/embedding job
        that called it, only lose the one line it was recording. Returns the line's
        committed dollar amount (`Decimal("0")` if recording itself failed), for a
        caller releasing a matching PROVIDER-scope budget reservation.
        """

        try:
            line, committed_usd = self._build_line(
                node_name=node_name,
                attempt=attempt,
                credential=credential,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                latency_ms=latency_ms,
                error_code=error_code,
            )
            self._lines.append(line)
            return committed_usd
        except Exception:
            logger.exception(
                "UsageRecorder failed to record an attempt for node=%s - this line is LOST, "
                "not retried (the provider call itself already happened)",
                node_name,
            )
            return Decimal("0")

    def _build_line(
        self,
        *,
        node_name: str,
        attempt: int,
        credential: CredentialConfig | None,
        status: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        latency_ms: int,
        error_code: str | None,
    ) -> tuple[dict[str, Any], Decimal]:
        occurred_at = _iso_z(datetime.now(UTC))

        if credential is None:
            # No credential known at all (e.g. failed before any was resolved) -
            # nothing to price, nothing to snapshot.
            line = {
                "seq": len(self._lines),
                "nodeName": node_name,
                "attempt": attempt,
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
                "latencyMs": latency_ms,
                "status": status,
                "errorCode": error_code,
                "occurredAt": occurred_at,
            }
            return line, Decimal("0")

        if status == "SUCCESS" and credential.model_name:
            result = calculate_actual(
                provider=credential.provider,
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
            # Budget commits PRICED cost when known, else the estimate (UNPRICED),
            # else nothing (FREE) - same rule Java's period-totals query uses.
            committed_usd = (
                result.cost_usd
                if cost_status == COST_STATUS_PRICED
                else (estimated_cost_usd if cost_status == COST_STATUS_UNPRICED else Decimal("0"))
            )
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
            committed_usd = Decimal("0")

        line = {
            "seq": len(self._lines),
            "nodeName": node_name,
            "attempt": attempt,
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
            "latencyMs": latency_ms,
            "status": status,
            "errorCode": error_code,
            "occurredAt": occurred_at,
        }
        return line, committed_usd

    @staticmethod
    def _line_committed_usd(line: dict[str, Any]) -> Decimal:
        """Same PRICED→costUsd / UNPRICED→estimatedCostUsd / FREE→0 rule
        `_build_line` used when it first computed each line - recomputed here from
        the stored wire-format strings since `_lines` keeps the built dicts, not the
        original `Decimal`s."""

        if line["costStatus"] == COST_STATUS_PRICED:
            return Decimal(line["costUsd"])
        if line["costStatus"] == COST_STATUS_UNPRICED:
            return Decimal(line["estimatedCostUsd"])
        return Decimal("0")

    async def close(self, *, status: str) -> None:
        """Call exactly once when the request ends - success, error, or client
        disconnect. Idempotent: a second call is a no-op, so a caller that closes
        defensively in more than one place can never double-send.

        Always settles the request-level budget reservation (if `budget_tracker`
        was given), even with zero lines (no provider call was ever made this
        request, e.g. a fast-path greeting) - the reservation still exists and must
        be released. Only the usage-log enqueue is skipped when there are no lines.
        `status` is the graph's own outcome (SUCCESS/ERROR); it is downgraded to
        PARTIAL here when at least one line failed but the graph still produced
        SUCCESS overall (a failover recovered from it).
        """

        if self._closed:
            return
        self._closed = True

        actual_total_usd = sum(
            (self._line_committed_usd(line) for line in self._lines), start=Decimal("0")
        )
        if self.budget_tracker is not None:
            await self.budget_tracker.settle_request(
                request_id=self.request_id, actual_total_usd=actual_total_usd
            )

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
            "documentId": self.document_id,
            "status": final_status,
            "startedAt": _iso_z(self._started_at),
            "finishedAt": _iso_z(finished_at),
            "lines": self._lines,
        }

        from app.core.usage.usage_outbox import (
            enqueue_usage_payload,  # local import: avoids a cycle at module load
        )

        try:
            await enqueue_usage_payload(payload)
        except Exception:
            # enqueue_usage_payload() itself already never raises (see its own
            # docstring) - this guards the payload-building above instead (e.g. a
            # malformed provider `usage` object that isn't plain-JSON-serializable).
            # Either way, a usage-recording bug must never surface to whatever
            # actually did the LLM/embedding call this recorder was measuring.
            logger.exception(
                "UsageRecorder.close() failed to build/send its payload for requestId=%s - "
                "usage record LOST",
                self.request_id,
            )
