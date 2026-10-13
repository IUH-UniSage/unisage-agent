import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import func

from app.database.models import StudentProfile
from app.database.repositories.dialect import insert_builder
from app.schemas.security import AcademicSecurityContext


@dataclass(frozen=True)
class ProfileOwner:
    """The user, or - for a guest, who has no user id - the conversation."""

    user_id: str | None = None
    conversation_id: str | None = None

    @classmethod
    def of(cls, security: AcademicSecurityContext, conversation_id: str) -> "ProfileOwner":
        if security.user_id:
            return cls(user_id=security.user_id)
        return cls(conversation_id=conversation_id)


class StudentProfileRepository:
    """Callers own the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, owner: ProfileOwner) -> dict[str, str]:
        column, key = self._key(owner)
        result = await self._session.execute(select(StudentProfile.attributes).where(column == key))
        attributes = result.scalar_one_or_none()
        return {str(name): str(value) for name, value in (attributes or {}).items()}

    async def save(self, owner: ProfileOwner, attributes: dict[str, str]) -> None:
        """Replace the whole profile (insert it on first write)."""

        column, _ = self._key(owner)
        insert = insert_builder(self._session)
        stmt = insert(StudentProfile).values(
            id=uuid.uuid4(),
            user_id=owner.user_id,
            conversation_id=owner.conversation_id,
            attributes=dict(attributes),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[column],
            set_={"attributes": stmt.excluded.attributes, "updated_at": func.now()},
        )
        await self._session.execute(stmt)

    @staticmethod
    def _key(owner: ProfileOwner) -> tuple[InstrumentedAttribute[str | None], str]:
        if owner.user_id:
            return StudentProfile.user_id, owner.user_id
        assert owner.conversation_id, "a profile needs a user or a conversation"
        return StudentProfile.conversation_id, owner.conversation_id
