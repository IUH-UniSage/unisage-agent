"""student_profiles; recent_calculations replaces last_calculation

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-10-13 12:00:00.000000+00:00

UNISAGE-95: the student's declared attributes move out of
conversation_clarification_states.confirmed_metadata into one place,
student_profiles, owned by the user (or, for a guest, the conversation). The old
per-conversation values are copied as conversation-owned profiles - the table
cannot tell which user a conversation belongs to - and the column is dropped. A
conversation keeps its latest calculations as a list instead of only the last one.
See changes/13-10-2026-Student-Profile-Metadata/plan.md.

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b5c6d7e8f9a0"
down_revision: str | None = "a4b5c6d7e8f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STATES = "conversation_clarification_states"
_PROFILES = "student_profiles"
_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    profiles = op.create_table(
        _PROFILES,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=True, unique=True),
        sa.Column("conversation_id", sa.String(), nullable=True, unique=True),
        sa.Column("attributes", _JSON, nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "(user_id IS NULL) <> (conversation_id IS NULL)", name="ck_student_profiles_one_owner"
        ),
    )
    states = sa.table(
        _STATES,
        sa.column("conversation_id", sa.String()),
        sa.column("confirmed_metadata", _JSON),
        sa.column("last_calculation", _JSON),
        sa.column("recent_calculations", _JSON),
    )
    bind = op.get_bind()
    rows = bind.execute(sa.select(states.c.conversation_id, states.c.confirmed_metadata)).all()
    copied = [
        {"id": uuid.uuid4(), "conversation_id": conversation_id, "attributes": attributes}
        for conversation_id, attributes in rows
        if attributes
    ]
    if copied:
        op.bulk_insert(profiles, copied)

    op.add_column(_STATES, sa.Column("recent_calculations", _JSON, nullable=True))
    for conversation_id, last in bind.execute(
        sa.select(states.c.conversation_id, states.c.last_calculation).where(
            states.c.last_calculation.is_not(None)
        )
    ).all():
        bind.execute(
            sa.update(states)
            .where(states.c.conversation_id == conversation_id)
            .values(recent_calculations=[last])
        )
    op.drop_column(_STATES, "last_calculation")
    op.drop_column(_STATES, "confirmed_metadata")


def downgrade() -> None:
    """Conversation-owned profiles go back to their conversation's row and the latest
    calculation back to `last_calculation`; user-owned profiles have no conversation
    to go back to and are lost."""

    op.add_column(
        _STATES,
        sa.Column("confirmed_metadata", _JSON, nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(_STATES, sa.Column("last_calculation", _JSON, nullable=True))
    states = sa.table(
        _STATES,
        sa.column("conversation_id", sa.String()),
        sa.column("confirmed_metadata", _JSON),
        sa.column("last_calculation", _JSON),
        sa.column("recent_calculations", _JSON),
    )
    profiles = sa.table(
        _PROFILES, sa.column("conversation_id", sa.String()), sa.column("attributes", _JSON)
    )
    bind = op.get_bind()
    for conversation_id, attributes in bind.execute(
        sa.select(profiles.c.conversation_id, profiles.c.attributes).where(
            profiles.c.conversation_id.is_not(None)
        )
    ).all():
        bind.execute(
            sa.update(states)
            .where(states.c.conversation_id == conversation_id)
            .values(confirmed_metadata=attributes)
        )
    for conversation_id, recent in bind.execute(
        sa.select(states.c.conversation_id, states.c.recent_calculations).where(
            states.c.recent_calculations.is_not(None)
        )
    ).all():
        if recent:
            bind.execute(
                sa.update(states)
                .where(states.c.conversation_id == conversation_id)
                .values(last_calculation=recent[-1])
            )
    op.drop_column(_STATES, "recent_calculations")
    op.drop_table(_PROFILES)
