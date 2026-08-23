"""add document process logs and chunks

Revision ID: 93676326d88b
Revises:
Create Date: 2026-08-23 03:58:43.569152+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "93676326d88b"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_process_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("object_key", sa.String(), nullable=False),
        sa.Column("current_step", sa.Enum("CHUNKED", name="documentprocessstep"), nullable=False),
        sa.Column("chunking_strategy", sa.String(), nullable=False),
        sa.Column(
            "chunking_params",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
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
        op.f("ix_document_process_logs_document_id"),
        "document_process_logs",
        ["document_id"],
        unique=True,
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("process_log_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("region_type", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["process_log_id"], ["document_process_logs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_index(op.f("ix_document_process_logs_document_id"), table_name="document_process_logs")
    op.drop_table("document_process_logs")
    # create_table's Enum column above auto-creates the `documentprocessstep` Postgres
    # enum type on upgrade; dropping the owning table does not drop the type itself,
    # so it must be dropped explicitly here.
    sa.Enum(name="documentprocessstep").drop(op.get_bind(), checkfirst=True)
