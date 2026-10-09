"""Start-of-turn gate and the cancel flow for the clarification panel.

The state of a panel lives in `conversation_clarification_states` (agent DB);
`messages.metadata.clarification` in Java is only its projection. Every 4xx
here is raised before `start_turn`, so a refused request creates no message
and costs no quota. Spec: docs/specs/SPEC-clarification-panel.md §2.3-§2.5.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors.exceptions import (
    BackendJavaUnavailableException,
    ClarificationPendingException,
    ClarificationProcessingException,
    ClarificationStaleException,
)
from app.database.repositories.clarification_state import ClarificationRoundRepository, RoundState
from app.integrations.backend_java_client import BackendJavaClient, BackendJavaError
from app.schemas.chat import ChatStreamRequest
from app.schemas.clarification import ClarificationCancel, PendingRound

logger = logging.getLogger(__name__)

JAVA_RETRY_DELAYS_SECONDS = (0.2, 0.4, 0.8)


def gate_clarification(state: RoundState, request: ChatStreamRequest) -> PendingRound | None:
    """Apply the §2.3 table. Returns the OPEN round a `clarification` request may act
    on, None for a plain message on a conversation with no round; raises otherwise."""

    action = request.clarification
    if action is None:
        if state.status == "OPEN":
            raise ClarificationPendingException()
        if state.status == "PROCESSING":
            raise ClarificationProcessingException()
        return None
    if state.status != "OPEN" or state.round is None:
        raise ClarificationStaleException()
    if state.round.panel.panel_id != action.panel_id:
        raise ClarificationStaleException()
    return state.round


async def with_java_retries(call: Callable[[], Awaitable[object]], *, what: str) -> bool:
    """Run a Java call up to 1 + len(JAVA_RETRY_DELAYS_SECONDS) times. True on success."""

    for attempt, delay in enumerate((0.0, *JAVA_RETRY_DELAYS_SECONDS), start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            await call()
            return True
        except BackendJavaError:
            logger.warning("%s failed (attempt %d)", what, attempt, exc_info=True)
    return False


async def _closed_stream(panel_id: uuid.UUID) -> AsyncGenerator[str, None]:
    payload = {"panel_id": str(panel_id), "status": "cancelled"}
    yield f"event: clarification_closed\ndata: {json.dumps(payload)}\n\n"
    yield "event: done\ndata: {}\n\n"


async def cancel_panel(
    *,
    db_session: AsyncSession,
    java_client: BackendJavaClient,
    conversation_id: str,
    pending: PendingRound,
    action: ClarificationCancel,
    confirmed_metadata: dict[str, str],
) -> AsyncGenerator[str, None]:
    """claim → PATCH projection (must succeed) → complete. No start_turn, no quota, no
    LLM. A projection that cannot be written restores the round and fails with 503,
    so state and projection both still say `open`."""

    rounds = ClarificationRoundRepository(db_session)
    token = uuid.uuid4()
    if (
        await rounds.claim(
            conversation_id, action.panel_id, token, settings.CHAT_CLARIFICATION_LEASE_SECONDS
        )
        is None
    ):
        raise ClarificationStaleException()
    await db_session.commit()

    message_id = pending.assistant_message_id
    projected = message_id is None or await with_java_retries(
        lambda: java_client.cancel_clarification(
            message_id=str(message_id), conversation_id=conversation_id
        ),
        what="clarification cancel projection",
    )
    if not projected:
        await rounds.restore(conversation_id, token)
        await db_session.commit()
        raise BackendJavaUnavailableException()

    await rounds.complete(conversation_id, token, None, confirmed_metadata)
    await db_session.commit()
    logger.info(
        "clarification.cancelled conversation_id=%s panel_id=%s", conversation_id, action.panel_id
    )
    return _closed_stream(action.panel_id)
