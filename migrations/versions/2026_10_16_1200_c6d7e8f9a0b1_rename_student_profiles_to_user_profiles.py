"""rename student_profiles to user_profiles

Revision ID: c6d7e8f9a0b1
Revises: b5c6d7e8f9a0
Create Date: 2026-10-16 12:00:00.000000+00:00

UNISAGE-95: the profile is the asker's own declared attributes whatever their role
(a student's intake, a lecturer's academic degree), not only a student's. Only names
change - table, primary key, unique constraints and check constraint; rows are kept.

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c6d7e8f9a0b1"
down_revision: str | None = "b5c6d7e8f9a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Postgres' default names for the constraints `b5c6d7e8f9a0` created without a name,
# plus the check constraint it named.
_CONSTRAINTS = ("pkey", "user_id_key", "conversation_id_key")


def _rename(old: str, new: str) -> None:
    op.rename_table(old, new)
    for suffix in _CONSTRAINTS:
        op.execute(f"ALTER TABLE {new} RENAME CONSTRAINT {old}_{suffix} TO {new}_{suffix}")
    op.execute(f"ALTER TABLE {new} RENAME CONSTRAINT ck_{old}_one_owner TO ck_{new}_one_owner")


def upgrade() -> None:
    _rename("student_profiles", "user_profiles")


def downgrade() -> None:
    _rename("user_profiles", "student_profiles")
