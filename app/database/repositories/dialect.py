from typing import Any

from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession


def insert_builder(session: AsyncSession) -> Any:
    """Pick the dialect-specific `insert()` constructor for `session`'s bind.

    Production runs on Postgres; this project's default test suite runs
    against an in-memory SQLite engine (see tests/conftest.py's "no live
    external service in the default test run" rule) - both dialects expose
    the same `INSERT ... ON CONFLICT ... DO UPDATE` API shape
    (`on_conflict_do_update(index_elements=..., set_=...)`), only the
    entry-point constructor differs, so dispatch on the bind's dialect name
    rather than hard-coding one.
    """

    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        return sqlite.insert
    return postgresql.insert
