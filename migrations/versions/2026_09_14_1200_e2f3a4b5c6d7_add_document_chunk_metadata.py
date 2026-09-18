"""add metadata column to document_chunks

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-09-14 12:00:00.000000+00:00

Phase 0 (Task 0.2) of changes/13-09-2026-Chunking-Structural-Metadata: every
`Chunk` field beyond chunk_index/content/region_type (heading_path,
source_type, block_index, source_locator, column_names, has_header,
header_source, header_confidence, chunking_version) is persisted as JSON
into this column instead of getting its own column. `server_default='{}'`
so existing rows backfill to an empty object rather than NULL - read back
through `Chunk(**{})`, that resolves every new field to its own default,
including `chunking_version="legacy"` (distinguishing pre-migration data
from anything produced by the current chunking logic).

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e2f3a4b5c6d7"
down_revision: str | None = "d1e2f3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "metadata")
