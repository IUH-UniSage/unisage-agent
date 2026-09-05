import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.database.models import ConversationClarificationState
from app.schemas.clarification import PendingClarification


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


class ClarificationStateRepository:
    """Persistence boundary for `ConversationClarificationState` (T1.1).

    Keyed by `conversation_id` (Java's id, no FK - see the model's
    docstring). One row per conversation; `get_or_default` never raises for
    a conversation with no state yet, since "nothing pending, nothing
    confirmed" is the normal starting state, not an error.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, conversation_id: str) -> ConversationClarificationState | None:
        result = await self._session.execute(
            select(ConversationClarificationState).where(
                ConversationClarificationState.conversation_id == conversation_id
            )
        )
        return result.scalar_one_or_none()

    async def get_pending_clarification(self, conversation_id: str) -> PendingClarification | None:
        row = await self.get(conversation_id)
        if row is None or row.pending_clarification is None:
            return None
        return PendingClarification.model_validate(row.pending_clarification)

    async def get_confirmed_metadata(self, conversation_id: str) -> dict[str, str]:
        row = await self.get(conversation_id)
        if row is None:
            return {}
        return dict(row.confirmed_metadata)

    async def upsert(
        self,
        conversation_id: str,
        *,
        pending_clarification: PendingClarification | None,
        confirmed_metadata: dict[str, str],
    ) -> ConversationClarificationState:
        """Replace both fields for `conversation_id` in one write.

        Callers (the Clarification Guard, node 12's post-processing step)
        always compute the full next value of each field themselves - this
        repository does not merge partial updates, to keep "who decided the
        next state" unambiguous (single-writer, per tasks/plan.md).

        Concurrency: two concurrent `upsert()` calls for the same
        `conversation_id` used to race a SELECT-then-INSERT (both could read
        "no row", both try to INSERT, the loser hits the column's unique
        constraint) - fixed by using a single atomic
        `INSERT ... ON CONFLICT (conversation_id) DO UPDATE` instead of
        branching on a prior SELECT. Semantics under concurrency: **last
        write wins** - whichever of the two `upsert()` calls' UPDATE commits
        last is what the row ends up holding, in full (not merged field by
        field), matching the "single writer decides the full next state"
        contract above. `populate_existing=True` is set so the ORM instance
        this returns always reflects what was actually written, even if an
        older copy of this row was already in the session's identity map.
        """

        pending_payload = (
            pending_clarification.model_dump() if pending_clarification is not None else None
        )
        insert = _insert_builder(self._session)
        stmt = insert(ConversationClarificationState).values(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            pending_clarification=pending_payload,
            confirmed_metadata=dict(confirmed_metadata),
        )
        stmt = (
            stmt.on_conflict_do_update(
                index_elements=[ConversationClarificationState.conversation_id],
                set_={
                    "pending_clarification": stmt.excluded.pending_clarification,
                    "confirmed_metadata": stmt.excluded.confirmed_metadata,
                    "updated_at": func.now(),
                },
            )
            .returning(ConversationClarificationState)
            .execution_options(populate_existing=True)
        )

        result = await self._session.execute(stmt)
        await self._session.flush()
        row: ConversationClarificationState = result.scalar_one()
        return row
