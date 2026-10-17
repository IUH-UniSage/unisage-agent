import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.database.models import ConversationClarificationState
from app.database.repositories.dialect import insert_builder
from app.schemas.clarification import LastCalculation, PendingRound

logger = logging.getLogger(__name__)

PendingStatus = Literal["OPEN", "PROCESSING"]


@dataclass(frozen=True)
class RoundState:
    """What a turn needs to know before it starts: is a panel open/being answered."""

    status: PendingStatus | None = None
    round: PendingRound | None = None
    recent_calculations: list[LastCalculation] = field(default_factory=list)


def _now() -> datetime:
    # Python-side clock, compared against `claim_expires_at` written by the same rule;
    # CLARIFICATION_LEASE_MARGIN_SECONDS covers skew between workers.
    return datetime.now(UTC)


RECENT_CALCULATIONS_MAX = 5


def _recent(raw: list[Any] | None) -> list[LastCalculation]:
    recent: list[LastCalculation] = []
    for item in raw or []:
        try:
            recent.append(LastCalculation.model_validate(item))
        except ValueError:
            logger.warning("clarification.unreadable_calculation")
    return recent


def _recent_json(recent: list[LastCalculation]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in recent[-RECENT_CALCULATIONS_MAX:]]


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


_CLEARED: dict[str, Any] = {
    "pending_status": None,
    "pending_clarification": None,
    "pending_panel_id": None,
    "claim_token": None,
    "claim_expires_at": None,
}


class ClarificationRoundRepository:
    """Panel v2 state machine on `conversation_clarification_states`
    (docs/specs/SPEC-clarification-panel.md §2).

    none ──upsert_open──► OPEN ──claim──► PROCESSING ──complete──► none | OPEN(new panel)
                           ▲                  │
                           └────restore───────┘

    Every transition out of PROCESSING is fenced by `claim_token`, so a request
    that lost its claim cannot write anything. Callers own the transaction and
    must commit right after `claim` so other requests see PROCESSING at once.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _row(self, conversation_id: str) -> ConversationClarificationState | None:
        result = await self._session.execute(
            select(ConversationClarificationState).where(
                ConversationClarificationState.conversation_id == conversation_id
            )
        )
        return result.scalar_one_or_none()

    async def _update(self, conversation_id: str, *conditions: Any, **values: Any) -> bool:
        stmt = (
            update(ConversationClarificationState)
            .where(ConversationClarificationState.conversation_id == conversation_id, *conditions)
            .values(**values, updated_at=func.now())
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(stmt)
        assert isinstance(result, CursorResult)
        return bool(result.rowcount == 1)

    async def get_round(self, conversation_id: str) -> RoundState:
        row = await self._row(conversation_id)
        if row is None:
            return RoundState()
        recent = _recent(row.recent_calculations)
        status = row.pending_status
        if status is None or row.pending_clarification is None:
            if row.pending_clarification is not None:
                # A pre-UNISAGE-99 (v1) round: no longer answerable, ignored until overwritten.
                logger.info("clarification.legacy_dropped conversation_id=%s", conversation_id)
            return RoundState(recent_calculations=recent)
        if (
            status == "PROCESSING"
            and row.claim_expires_at is not None
            and _as_utc(row.claim_expires_at) < _now()
        ):
            # The claimed turn died (its deadline is shorter than the lease): consume it.
            await self._update(
                conversation_id,
                ConversationClarificationState.claim_token == row.claim_token,
                ConversationClarificationState.pending_status == "PROCESSING",
                **_CLEARED,
            )
            logger.warning(
                "clarification.lease_expired conversation_id=%s panel_id=%s",
                conversation_id,
                row.pending_panel_id,
            )
            return RoundState(recent_calculations=recent)
        try:
            pending = PendingRound.model_validate(row.pending_clarification)
        except ValueError:
            logger.warning("clarification.unreadable_round conversation_id=%s", conversation_id)
            return RoundState(recent_calculations=recent)
        state: PendingStatus = "OPEN" if status == "OPEN" else "PROCESSING"
        return RoundState(status=state, round=pending, recent_calculations=recent)

    async def save_calculations(
        self, conversation_id: str, recent_calculations: list[LastCalculation] | None = None
    ) -> None:
        """Insert the row if needed; never touches the round columns of an existing row.
        `recent_calculations` is only written when given (a turn without a calculation
        keeps the list as it is)."""

        insert = insert_builder(self._session)
        recent = _recent_json(recent_calculations) if recent_calculations else None
        stmt = insert(ConversationClarificationState).values(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            pending_clarification=None,
            recent_calculations=recent,
        )
        updates: dict[str, Any] = {"updated_at": func.now()}
        if recent is not None:
            updates["recent_calculations"] = stmt.excluded.recent_calculations
        stmt = stmt.on_conflict_do_update(
            index_elements=[ConversationClarificationState.conversation_id], set_=updates
        )
        await self._session.execute(stmt)

    async def upsert_open(
        self,
        conversation_id: str,
        pending: PendingRound,
        recent_calculations: list[LastCalculation] | None = None,
    ) -> bool:
        """Open a round on a turn that held no claim. False (and nothing written to the
        round columns) if a round is already OPEN/PROCESSING."""

        await self.save_calculations(conversation_id, recent_calculations)
        opened = await self._update(
            conversation_id,
            ConversationClarificationState.pending_status.is_(None),
            pending_status="OPEN",
            pending_clarification=pending.model_dump(mode="json"),
            pending_panel_id=pending.panel.panel_id,
            claim_token=None,
            claim_expires_at=None,
        )
        if not opened:
            logger.warning("clarification.open_refused conversation_id=%s", conversation_id)
        return opened

    async def claim(
        self, conversation_id: str, panel_id: uuid.UUID, token: uuid.UUID, lease_seconds: float
    ) -> PendingRound | None:
        claimed = await self._update(
            conversation_id,
            ConversationClarificationState.pending_status == "OPEN",
            ConversationClarificationState.pending_panel_id == panel_id,
            pending_status="PROCESSING",
            claim_token=token,
            claim_expires_at=_now() + timedelta(seconds=lease_seconds),
        )
        if not claimed:
            return None
        row = await self._row(conversation_id)
        assert row is not None and row.pending_clarification is not None
        await self._session.refresh(row)
        return PendingRound.model_validate(row.pending_clarification)

    async def still_owner(self, conversation_id: str, token: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(ConversationClarificationState.id).where(
                ConversationClarificationState.conversation_id == conversation_id,
                ConversationClarificationState.pending_status == "PROCESSING",
                ConversationClarificationState.claim_token == token,
            )
        )
        return result.scalar_one_or_none() is not None

    async def restore(self, conversation_id: str, token: uuid.UUID) -> bool:
        """PROCESSING → OPEN, for a claim that failed before any side effect."""

        restored = await self._update(
            conversation_id,
            ConversationClarificationState.pending_status == "PROCESSING",
            ConversationClarificationState.claim_token == token,
            pending_status="OPEN",
            claim_token=None,
            claim_expires_at=None,
        )
        if not restored:
            logger.warning(
                "clarification.claim_lost conversation_id=%s on restore", conversation_id
            )
        return restored

    async def complete(
        self,
        conversation_id: str,
        token: uuid.UUID,
        new_round: PendingRound | None,
        recent_calculations: list[LastCalculation] | None = None,
    ) -> bool:
        """PROCESSING → none, or → OPEN with the next panel of a chain."""

        values: dict[str, Any] = dict(_CLEARED)
        if recent_calculations:
            values["recent_calculations"] = _recent_json(recent_calculations)
        if new_round is not None:
            values.update(
                pending_status="OPEN",
                pending_clarification=new_round.model_dump(mode="json"),
                pending_panel_id=new_round.panel.panel_id,
            )
        completed = await self._update(
            conversation_id,
            ConversationClarificationState.pending_status == "PROCESSING",
            ConversationClarificationState.claim_token == token,
            **values,
        )
        if not completed:
            logger.warning(
                "clarification.claim_lost conversation_id=%s on complete", conversation_id
            )
        return completed

    async def revoke_open(self, conversation_id: str, panel_id: uuid.UUID) -> bool:
        """OPEN → none, when the panel could not be projected to Java."""

        return await self._update(
            conversation_id,
            ConversationClarificationState.pending_status == "OPEN",
            ConversationClarificationState.pending_panel_id == panel_id,
            **_CLEARED,
        )
