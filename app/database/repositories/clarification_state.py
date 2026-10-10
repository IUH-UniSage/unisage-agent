import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import CursorResult, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.database.models import ConversationClarificationState
from app.schemas.clarification import LastCalculation, PendingRound

logger = logging.getLogger(__name__)

PendingStatus = Literal["OPEN", "PROCESSING"]


def _insert_builder(session: AsyncSession) -> Any:
    """Pick the dialect-specific `insert()` constructor for `session`'s bind.

    Production runs on Postgres; this project's default test suite runs
    against an in-memory SQLite engine (see tests/conftest.py's "no live
    external service in the default test run" rule) - both dialects expose
    the same `INSERT ... ON CONFLICT ... DO UPDATE` API shape
    (`on_conflict_do_update(index_elements=..., set_=...)`), only the
    entry-point constructor differs, so dispatch on the bind's dialect name
    rather than hard-coding one.
    """

    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        return sqlite.insert
    return postgresql.insert


@dataclass(frozen=True)
class RoundState:
    """What a turn needs to know before it starts: is a panel open/being answered."""

    status: PendingStatus | None = None
    round: PendingRound | None = None
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    last_calculation: LastCalculation | None = None


def _now() -> datetime:
    # Python-side clock, compared against `claim_expires_at` written by the same rule;
    # CLARIFICATION_LEASE_MARGIN_SECONDS covers skew between workers.
    return datetime.now(UTC)


def _last_calculation(raw: dict[str, Any] | None) -> LastCalculation | None:
    if raw is None:
        return None
    try:
        return LastCalculation.model_validate(raw)
    except ValueError:
        logger.warning("clarification.unreadable_last_calculation")
        return None


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
        confirmed = dict(row.confirmed_metadata)
        last = _last_calculation(row.last_calculation)
        status = row.pending_status
        if status is None or row.pending_clarification is None:
            if row.pending_clarification is not None:
                # A pre-UNISAGE-99 (v1) round: no longer answerable, ignored until overwritten.
                logger.info("clarification.legacy_dropped conversation_id=%s", conversation_id)
            return RoundState(confirmed_metadata=confirmed, last_calculation=last)
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
            return RoundState(confirmed_metadata=confirmed, last_calculation=last)
        try:
            pending = PendingRound.model_validate(row.pending_clarification)
        except ValueError:
            logger.warning("clarification.unreadable_round conversation_id=%s", conversation_id)
            return RoundState(confirmed_metadata=confirmed, last_calculation=last)
        state: PendingStatus = "OPEN" if status == "OPEN" else "PROCESSING"
        return RoundState(
            status=state, round=pending, confirmed_metadata=confirmed, last_calculation=last
        )

    async def save_confirmed_metadata(
        self,
        conversation_id: str,
        confirmed_metadata: dict[str, str],
        last_calculation: LastCalculation | None = None,
    ) -> None:
        """Insert the row if needed; never touches the round columns of an existing row.
        `last_calculation` is only written when given (a turn without a calculation
        keeps the previous one)."""

        insert = _insert_builder(self._session)
        last = last_calculation.model_dump(mode="json") if last_calculation else None
        stmt = insert(ConversationClarificationState).values(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            pending_clarification=None,
            confirmed_metadata=dict(confirmed_metadata),
            last_calculation=last,
        )
        updates: dict[str, Any] = {
            "confirmed_metadata": stmt.excluded.confirmed_metadata,
            "updated_at": func.now(),
        }
        if last is not None:
            updates["last_calculation"] = stmt.excluded.last_calculation
        stmt = stmt.on_conflict_do_update(
            index_elements=[ConversationClarificationState.conversation_id], set_=updates
        )
        await self._session.execute(stmt)

    async def upsert_open(
        self,
        conversation_id: str,
        pending: PendingRound,
        confirmed_metadata: dict[str, str],
        last_calculation: LastCalculation | None = None,
    ) -> bool:
        """Open a round on a turn that held no claim. False (and nothing written to the
        round columns) if a round is already OPEN/PROCESSING."""

        await self.save_confirmed_metadata(conversation_id, confirmed_metadata, last_calculation)
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
        confirmed_metadata: dict[str, str],
        last_calculation: LastCalculation | None = None,
    ) -> bool:
        """PROCESSING → none, or → OPEN with the next panel of a chain."""

        values: dict[str, Any] = dict(_CLEARED)
        if last_calculation is not None:
            values["last_calculation"] = last_calculation.model_dump(mode="json")
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
            confirmed_metadata=dict(confirmed_metadata),
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
