"""Profile: the caller's own attributes - per user when logged in (any role), per
conversation for a guest."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import UserProfile
from app.database.repositories.user_profile import ProfileOwner, UserProfileRepository
from app.schemas.security import AcademicSecurityContext


def test_owner_is_the_user_when_logged_in_else_the_conversation() -> None:
    user = AcademicSecurityContext(user_id="u1", role="SINH_VIEN")
    assert ProfileOwner.of(user, "c1") == ProfileOwner(user_id="u1")
    assert ProfileOwner.of(AcademicSecurityContext(), "c1") == ProfileOwner(conversation_id="c1")


def test_a_lecturer_or_staff_profile_follows_the_user_like_a_students() -> None:
    """It holds their own attributes (academic degree, contract type), not a student's."""

    for role in ("GIANG_VIEN", "CAN_BO", "SUPER_ADMIN"):
        caller = AcademicSecurityContext(user_id="u1", role=role)
        assert ProfileOwner.of(caller, "c1") == ProfileOwner(user_id="u1"), role


@pytest.mark.asyncio
async def test_no_row_means_an_empty_profile(db_session: AsyncSession) -> None:
    assert await UserProfileRepository(db_session).get(ProfileOwner(user_id="u1")) == {}


@pytest.mark.asyncio
async def test_save_inserts_then_replaces_the_whole_profile(db_session: AsyncSession) -> None:
    repo = UserProfileRepository(db_session)
    owner = ProfileOwner(user_id="u1")
    await repo.save(owner, {"khoa_nhap_hoc": "2022", "nganh": "CNTT"})
    await db_session.commit()
    await repo.save(owner, {"khoa_nhap_hoc": "2023"})
    await db_session.commit()
    assert await repo.get(owner) == {"khoa_nhap_hoc": "2023"}


@pytest.mark.asyncio
async def test_user_and_guest_profiles_are_kept_apart(db_session: AsyncSession) -> None:
    repo = UserProfileRepository(db_session)
    await repo.save(ProfileOwner(user_id="c1"), {"nganh": "CNTT"})
    await repo.save(ProfileOwner(conversation_id="c1"), {"nganh": "Kế toán"})
    await db_session.commit()
    assert await repo.get(ProfileOwner(user_id="c1")) == {"nganh": "CNTT"}
    assert await repo.get(ProfileOwner(conversation_id="c1")) == {"nganh": "Kế toán"}
    # A user's profile is the same in every conversation.
    user = AcademicSecurityContext(user_id="c1", role="SINH_VIEN")
    assert await repo.get(ProfileOwner.of(user, "another")) == {"nganh": "CNTT"}


@pytest.mark.asyncio
async def test_a_profile_has_exactly_one_owner(db_session: AsyncSession) -> None:
    db_session.add(UserProfile(id=uuid.uuid4(), user_id="u1", conversation_id="c1"))
    with pytest.raises(IntegrityError):
        await db_session.commit()
