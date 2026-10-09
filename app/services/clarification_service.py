"""Clarification panel service: the start-of-turn gate, cancel, and preparing a
submit (validate the answers against the stored panel, then claim the round).

The state of a panel lives in `conversation_clarification_states` (agent DB);
`messages.metadata.clarification` in Java is only its projection. Every 4xx
here is raised before `start_turn`, so a refused request creates no message
and costs no quota. Spec: docs/specs/SPEC-clarification-panel.md §2.3-§2.5.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors.exceptions import (
    BackendJavaUnavailableException,
    ClarificationInvalidException,
    ClarificationPendingException,
    ClarificationProcessingException,
    ClarificationStaleException,
)
from app.database.repositories.clarification_state import ClarificationRoundRepository, RoundState
from app.graph.clarification_answers import (
    ClarificationInvalid,
    answers_metadata,
    answers_summary,
    validate_answers,
)
from app.graph.streaming_session import ClaimContext
from app.graph.streaming_state import ResumeInput
from app.integrations.backend_java_client import BackendJavaClient, with_java_retries
from app.schemas.chat import ChatStreamRequest
from app.schemas.clarification import ClarificationCancel, ClarificationSubmit, PendingRound

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class PreparedSubmit:
    """A validated, claimed panel submit - everything the turn needs to resume it."""

    claim: ClaimContext
    resume: ResumeInput
    summary: str  # the USER message content
    start_turn_metadata: dict[str, Any]  # metadata.clarification_answers for that message


async def prepare_submit(
    *,
    db_session: AsyncSession,
    conversation_id: str,
    pending: PendingRound,
    action: ClarificationSubmit,
) -> PreparedSubmit:
    """Validate against the stored panel (400 if wrong, panel stays open), then claim
    the round OPEN → PROCESSING and commit so other requests see it at once."""

    try:
        answers = validate_answers(pending.panel, action)
    except ClarificationInvalid as exc:
        raise ClarificationInvalidException(
            {error.question_id: error.reason for error in exc.errors}
        ) from exc
    token = uuid.uuid4()
    claimed = await ClarificationRoundRepository(db_session).claim(
        conversation_id, action.panel_id, token, settings.CHAT_CLARIFICATION_LEASE_SECONDS
    )
    if claimed is None:
        raise ClarificationStaleException()
    await db_session.commit()
    return PreparedSubmit(
        claim=ClaimContext(
            token=token,
            deadline=asyncio.get_running_loop().time()
            + settings.CHAT_CLAIMED_TURN_DEADLINE_SECONDS,
        ),
        resume=ResumeInput(pending_round=claimed, answers=answers),
        summary=answers_summary(answers),
        start_turn_metadata={"clarification_answers": answers_metadata(claimed.panel, answers)},
    )


async def release_claim(
    db_session: AsyncSession, conversation_id: str, claim: ClaimContext
) -> None:
    """start_turn failed - no message, no quota yet: hand the panel back (best effort)."""

    with contextlib.suppress(Exception):
        await ClarificationRoundRepository(db_session).restore(conversation_id, claim.token)
        await db_session.commit()
