"""add conversation_clarification_states

Revision ID: d1e2f3a4b5c6
Revises: c4d5e6f7a8b9
Create Date: 2026-09-05 12:00:00.000000+00:00

T1.1: missing-metadata clarification state (see tasks/plan.md and
missing_metadata_clarification_design.md section 5), keyed by Java's
conversation_id with no FK — same pattern as document_process_logs.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d1e2f3a4b5c6"
down_revision: str | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_clarification_states",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column(
            "pending_clarification",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "confirmed_metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_clarification_states_conversation_id"),
        "conversation_clarification_states",
        ["conversation_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_conversation_clarification_states_conversation_id"),
        table_name="conversation_clarification_states",
    )
    op.drop_table("conversation_clarification_states")
