"""add department_id to document_process_logs

Revision ID: c4d5e6f7a8b9
Revises: 3fbeadf96f68
Create Date: 2026-09-03 12:00:00.000000+00:00

Lets `GET /ingestion/jobs/{document_id}` authorize the caller against the
draft's owning department (`require_department_membership`). Nullable so
rows written before this migration keep working - those skip the
membership check with a logged warning.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "3fbeadf96f68"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_process_logs",
        sa.Column("department_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_process_logs", "department_id")
