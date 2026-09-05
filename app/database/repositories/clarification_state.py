from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ConversationClarificationState
from app.schemas.clarification import PendingClarification


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
        """

        row = await self.get(conversation_id)
        pending_payload = (
            pending_clarification.model_dump() if pending_clarification is not None else None
        )
        if row is None:
            row = ConversationClarificationState(
                conversation_id=conversation_id,
                pending_clarification=pending_payload,
                confirmed_metadata=dict(confirmed_metadata),
            )
            self._session.add(row)
        else:
            row.pending_clarification = pending_payload
            row.confirmed_metadata = dict(confirmed_metadata)

        await self._session.flush()
        return row
