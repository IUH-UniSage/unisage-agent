from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ChatDeps:
    """Runtime dependencies shared by graph steps."""

    db_session: AsyncSession | None
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
