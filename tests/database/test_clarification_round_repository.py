"""Panel v2 state machine: OPEN → PROCESSING(claim_token, lease) → complete/restore."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import ConversationClarificationState
from app.database.repositories.clarification_state import ClarificationRoundRepository
from app.schemas.clarification import (
    CalculationPlan,
    ClarificationPanel,
    LastCalculation,
    PendingAdvisoryTask,
    PendingRound,
)
from app.schemas.intent import ClassifiedTask

LEASE = 210.0


def _round(panel_id: uuid.UUID | None = None) -> PendingRound:
    panel = ClarificationPanel.model_validate(
        {
            "panel_id": str(panel_id or uuid.uuid4()),
            "questions": [
                {
                    "id": "q1",
                    "tab_label": "Khoá",
                    "prompt": "Bạn thuộc khoá nào?",
                    "kind": "choice",
                    "options": [{"id": "k19", "label": "K19"}, {"id": "k20", "label": "K20"}],
                    "allow_other": True,
                    "origin": "advisory",
                    "task_id": "T1",
                    "field": "cohort",
                }
            ],
        }
    )
    return PendingRound(
        panel=panel,
        original_query="Điều kiện tốt nghiệp?",
        tasks=[
            PendingAdvisoryTask(
                task_id="T1",
                origin_tasks=[
                    ClassifiedTask(intent="academic_advisory", query="Điều kiện tốt nghiệp?")
                ],
            )
        ],
        created_at=datetime.now(UTC),
    )


async def _open(session: AsyncSession, conversation_id: str = "c1") -> PendingRound:
    pending = _round()
    assert await ClarificationRoundRepository(session).upsert_open(
        conversation_id, pending, {"program": "CNTT"}
    )
    await session.commit()
    return pending


@pytest.mark.asyncio
async def test_no_row_means_no_round(db_session: AsyncSession) -> None:
    state = await ClarificationRoundRepository(db_session).get_round("nope")
    assert state.status is None and state.round is None and state.confirmed_metadata == {}


@pytest.mark.asyncio
async def test_open_then_read_back(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    state = await ClarificationRoundRepository(db_session).get_round("c1")
    assert state.status == "OPEN"
    assert state.round == pending
    assert state.confirmed_metadata == {"program": "CNTT"}


@pytest.mark.asyncio
async def test_open_refuses_to_overwrite_an_existing_round(db_session: AsyncSession) -> None:
    first = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)
    assert not await repo.upsert_open("c1", _round(), {"program": "KT"})
    await db_session.commit()
    state = await repo.get_round("c1")
    assert state.round == first
    assert state.confirmed_metadata == {"program": "KT"}  # metadata still saved


@pytest.mark.asyncio
async def test_claim_moves_to_processing_and_rejects_wrong_panel(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)

    assert await repo.claim("c1", uuid.uuid4(), uuid.uuid4(), LEASE) is None
    token = uuid.uuid4()
    assert await repo.claim("c1", pending.panel.panel_id, token, LEASE) == pending
    await db_session.commit()

    state = await repo.get_round("c1")
    assert state.status == "PROCESSING"
    assert await repo.still_owner("c1", token)
    # A second claim (double submit) cannot take a PROCESSING round.
    assert await repo.claim("c1", pending.panel.panel_id, uuid.uuid4(), LEASE) is None


@pytest.mark.asyncio
async def test_restore_and_complete_are_fenced_by_token(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)
    token = uuid.uuid4()
    await repo.claim("c1", pending.panel.panel_id, token, LEASE)

    assert not await repo.restore("c1", uuid.uuid4())
    assert not await repo.complete("c1", uuid.uuid4(), None, {})
    assert (await repo.get_round("c1")).status == "PROCESSING"

    assert await repo.restore("c1", token)
    assert (await repo.get_round("c1")).status == "OPEN"
    assert not await repo.still_owner("c1", token)


@pytest.mark.asyncio
async def test_complete_clears_or_opens_the_next_panel(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)
    token = uuid.uuid4()
    await repo.claim("c1", pending.panel.panel_id, token, LEASE)
    follow_up = _round()
    assert await repo.complete("c1", token, follow_up, {"program": "CNTT", "cohort": "k20"})
    state = await repo.get_round("c1")
    assert state.status == "OPEN" and state.round == follow_up
    assert state.confirmed_metadata == {"program": "CNTT", "cohort": "k20"}

    token2 = uuid.uuid4()
    await repo.claim("c1", follow_up.panel.panel_id, token2, LEASE)
    assert await repo.complete("c1", token2, None, {})
    assert (await repo.get_round("c1")).status is None


@pytest.mark.asyncio
async def test_expired_lease_is_consumed_on_next_read(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)
    token = uuid.uuid4()
    await repo.claim("c1", pending.panel.panel_id, token, LEASE)
    await db_session.execute(
        update(ConversationClarificationState).values(
            claim_expires_at=datetime.now(UTC) - timedelta(seconds=1)
        )
    )
    await db_session.commit()

    assert (await repo.get_round("c1")).status is None
    # The old request is fenced out for good.
    assert not await repo.complete("c1", token, None, {})
    assert not await repo.restore("c1", token)


@pytest.mark.asyncio
async def test_revoke_open_only_matches_its_panel(db_session: AsyncSession) -> None:
    pending = await _open(db_session)
    repo = ClarificationRoundRepository(db_session)
    assert not await repo.revoke_open("c1", uuid.uuid4())
    assert await repo.revoke_open("c1", pending.panel.panel_id)
    assert (await repo.get_round("c1")).status is None


@pytest.mark.asyncio
async def test_legacy_v1_row_reads_as_no_round(db_session: AsyncSession) -> None:
    # A pre-UNISAGE-99 row: v1 JSON in pending_clarification, no pending_status.
    db_session.add(
        ConversationClarificationState(
            conversation_id="c1",
            pending_clarification={
                "origin_node": "QueryTransformationNode",
                "missing_fields": ["x"],
            },
            confirmed_metadata={"program": "CNTT"},
        )
    )
    await db_session.commit()
    repo = ClarificationRoundRepository(db_session)
    state = await repo.get_round("c1")
    assert state.status is None and state.confirmed_metadata == {"program": "CNTT"}
    # ...and a new round may be opened over it.
    assert await repo.upsert_open("c1", _round(), {})


@pytest.mark.asyncio
async def test_concurrent_claims_have_one_winner(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session:
        pending = await _open(session)

    async def attempt() -> bool:
        async with db_session_factory() as session:
            claimed = await ClarificationRoundRepository(session).claim(
                "c1", pending.panel.panel_id, uuid.uuid4(), LEASE
            )
            await session.commit()
            return claimed is not None

    results = await asyncio.gather(attempt(), attempt())
    assert sorted(results) == [False, True]


@pytest.mark.asyncio
async def test_last_calculation_is_kept_until_a_new_one_is_written(
    db_session: AsyncSession,
) -> None:
    repo = ClarificationRoundRepository(db_session)
    first = LastCalculation(
        plan=CalculationPlan(formula_id="grade_conversion"), params={"score10": 8}
    )
    await repo.save_confirmed_metadata("c1", {}, first)
    await db_session.commit()
    # A turn without a calculation (None) keeps the stored one.
    await repo.save_confirmed_metadata("c1", {"program": "CNTT"})
    await db_session.commit()
    state = await repo.get_round("c1")
    assert state.last_calculation == first and state.confirmed_metadata == {"program": "CNTT"}

    pending = await _open(db_session, "c2")
    token = uuid.uuid4()
    assert await repo.claim("c2", pending.panel.panel_id, token, LEASE)
    await db_session.commit()
    second = LastCalculation(plan=CalculationPlan(formula_id="course_score"), params={"tclt": 3})
    assert await repo.complete("c2", token, None, {}, second)
    await db_session.commit()
    assert (await repo.get_round("c2")).last_calculation == second
