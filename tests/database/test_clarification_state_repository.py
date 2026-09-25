import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import ConversationClarificationState
from app.database.repositories.clarification_state import ClarificationStateRepository
from app.schemas.clarification import PendingClarification


@pytest.mark.asyncio
async def test_get_on_unknown_conversation_returns_defaults(db_session: AsyncSession) -> None:
    repo = ClarificationStateRepository(db_session)

    assert await repo.get("no-such-conv") is None
    assert await repo.get_pending_clarification("no-such-conv") is None
    assert await repo.get_confirmed_metadata("no-such-conv") == {}


@pytest.mark.asyncio
async def test_upsert_then_read_back_round_trips_pending_and_confirmed(
    db_session: AsyncSession,
) -> None:
    repo = ClarificationStateRepository(db_session)
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        pending_sub_query_id=None,
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong", "vlvh"]],
        retry_count=0,
    )

    await repo.upsert(
        "conv-1", pending_clarification=pending, confirmed_metadata={"program": "CNTT"}
    )
    await db_session.commit()

    read_back = await repo.get_pending_clarification("conv-1")
    assert read_back == pending
    assert await repo.get_confirmed_metadata("conv-1") == {"program": "CNTT"}


@pytest.mark.asyncio
async def test_upsert_overwrites_previous_row_not_creates_a_second_one(
    db_session: AsyncSession,
) -> None:
    repo = ClarificationStateRepository(db_session)

    await repo.upsert(
        "conv-1",
        pending_clarification=PendingClarification(
            origin_node="QueryTransformationNode",
            missing_fields=["training_type"],
            options=[["chinh_quy", "lien_thong"]],
            retry_count=1,
        ),
        confirmed_metadata={},
    )
    await db_session.commit()

    # Matched -> clear pending, write confirmed_metadata (guard's "match" branch).
    await repo.upsert(
        "conv-1", pending_clarification=None, confirmed_metadata={"training_type": "chinh_quy"}
    )
    await db_session.commit()

    assert await repo.get_pending_clarification("conv-1") is None
    assert await repo.get_confirmed_metadata("conv-1") == {"training_type": "chinh_quy"}
    row = await repo.get("conv-1")
    assert row is not None


@pytest.mark.asyncio
async def test_concurrent_upserts_for_same_conversation_both_succeed_last_write_wins(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two concurrent `upsert()` calls for the same `conversation_id` used to
    race a SELECT-then-INSERT: both could read "no row exists", both then
    try to INSERT, and the loser hit the column's unique constraint - an
    exception that `run_and_persist`'s broad `except Exception` silently
    swallowed, dropping that turn's clarification state with no visible
    error.

    Fixed via `INSERT ... ON CONFLICT (conversation_id) DO UPDATE`, so both
    concurrent calls succeed (neither raises), only one row ever exists for
    the conversation, and "last write wins" - whichever call's UPDATE
    commits last is what the row holds afterwards, in full.
    """

    async def upsert_via_own_session(*, value: str) -> None:
        async with db_session_factory() as session:
            repo = ClarificationStateRepository(session)
            await repo.upsert(
                "conv-race",
                pending_clarification=None,
                confirmed_metadata={"source": value},
            )
            await session.commit()

    # Both callers open their OWN session (simulating two truly independent
    # requests) and start from "no row exists yet" - run them concurrently
    # via asyncio.gather so both writers are in flight before either commits.
    await asyncio.gather(
        upsert_via_own_session(value="writer-a"),
        upsert_via_own_session(value="writer-b"),
    )

    async with db_session_factory() as session:
        repo = ClarificationStateRepository(session)
        final = await repo.get_confirmed_metadata("conv-race")

    # Neither writer raised (no unique-violation leaked out), exactly one
    # row exists, and it holds one writer's value whole (not a merge of
    # both) - "last write wins" semantics.
    assert final in ({"source": "writer-a"}, {"source": "writer-b"})


@pytest.mark.asyncio
async def test_get_pending_clarification_parses_a_legacy_row_without_origin_tasks(
    db_session: AsyncSession,
) -> None:
    """A row persisted before `origin_tasks`/`pending_sub_query_id` existed
    (bare JSONB, missing both keys entirely) must still `model_validate`."""

    legacy_json = {
        "origin_node": "QueryTransformationNode",
        "missing_fields": ["training_type"],
        "options": [["chinh_quy", "lien_thong"]],
        "retry_count": 0,
    }
    session = db_session
    session.add(
        ConversationClarificationState(
            conversation_id="conv-legacy",
            pending_clarification=legacy_json,
            confirmed_metadata={},
        )
    )
    await session.commit()

    repo = ClarificationStateRepository(session)
    pending = await repo.get_pending_clarification("conv-legacy")

    assert pending is not None
    assert pending.origin_tasks is None
    assert pending.pending_sub_query_id is None
    assert pending.missing_fields == ["training_type"]
