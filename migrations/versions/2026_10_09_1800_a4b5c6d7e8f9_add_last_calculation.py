"""add last_calculation to conversation_clarification_states

Revision ID: a4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-10-09 18:00:00.000000+00:00

UNISAGE-99: the conversation's latest computed calculation (formula plan and
parameters), so a follow-up target question reuses it instead of asking every
number again. See docs/specs/SPEC-calculation-node.md §8.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a4b5c6d7e8f9"
down_revision: str | None = "f3a4b5c6d7e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "conversation_clarification_states"


def upgrade() -> None:
    op.add_column(
        _TABLE,
        sa.Column(
            "last_calculation",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(_TABLE, "last_calculation")
