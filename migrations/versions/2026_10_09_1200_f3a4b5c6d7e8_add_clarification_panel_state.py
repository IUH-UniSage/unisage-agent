"""add clarification panel state machine columns

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-10-09 12:00:00.000000+00:00

UNISAGE-99: a pending clarification round is OPEN or PROCESSING (claimed by
one request, fenced by claim_token, bounded by claim_expires_at). NULL status
means no round. See docs/specs/SPEC-clarification-panel.md §2.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3a4b5c6d7e8"
down_revision: str | None = "e2f3a4b5c6d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "conversation_clarification_states"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("pending_status", sa.String(length=16), nullable=True))
    op.add_column(_TABLE, sa.Column("pending_panel_id", sa.Uuid(), nullable=True))
    op.add_column(_TABLE, sa.Column("claim_token", sa.Uuid(), nullable=True))
    op.add_column(_TABLE, sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_check_constraint(
        "ck_clarification_pending_status",
        _TABLE,
        "pending_status IS NULL OR pending_status IN ('OPEN', 'PROCESSING')",
    )
    op.create_index(
        "ix_conversation_clarification_states_pending_panel_id", _TABLE, ["pending_panel_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_clarification_states_pending_panel_id", table_name=_TABLE)
    op.drop_constraint("ck_clarification_pending_status", _TABLE, type_="check")
    op.drop_column(_TABLE, "claim_expires_at")
    op.drop_column(_TABLE, "claim_token")
    op.drop_column(_TABLE, "pending_panel_id")
    op.drop_column(_TABLE, "pending_status")
