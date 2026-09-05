import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
