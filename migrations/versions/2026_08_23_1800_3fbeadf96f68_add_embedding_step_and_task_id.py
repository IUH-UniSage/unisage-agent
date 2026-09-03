"""add embedding step and celery task id

Revision ID: 3fbeadf96f68
Revises: 93676326d88b
Create Date: 2026-08-23 18:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3fbeadf96f68"
down_revision: str | None = "93676326d88b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE documentprocessstep ADD VALUE IF NOT EXISTS 'EMBEDDING'")
    op.add_column(
        "document_process_logs",
        sa.Column("celery_task_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    # Postgres cannot drop a single enum value in place; narrowing the type
    # back down would require rebuilding it (create a new type, cast every
    # row, drop the old type) and is not needed for this project's scope -
    # only the added column is reversed.
    op.drop_column("document_process_logs", "celery_task_id")
