import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DepartmentAccessDeniedException,
    InsufficientDocumentPermissionException,
    InvalidTrustedContextException,
    MissingTrustedContextException,
)
from app.database.session import get_db_session
from app.graph.deps import ChatDeps

_DOCUMENT_WRITE_PERMISSIONS = {"DOCUMENT_ALL", "DOCUMENT_CREATE"}


async def get_chat_deps(
    db_session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[ChatDeps, None]:
    """Build graph dependencies from the request-scoped database session."""

    yield ChatDeps(
        db_session=db_session,
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
    )


@dataclass(frozen=True)
class DepartmentAccessEntry:
    """One department the caller is granted access to, and at what level."""

    department_id: str
    access_level: int


@dataclass(frozen=True)
class TrustedContext:
    """Caller identity trusted because the API Gateway injected it after JWT verification."""

    department_access: list[DepartmentAccessEntry]
    permissions: list[str]


async def get_trusted_context(
    x_user_department_access: str | None = Header(default=None),
    x_user_permissions: str | None = Header(default=None),
) -> TrustedContext:
    """Read the gateway-injected trusted headers, failing loudly if either is absent."""

    if not x_user_department_access:
        raise MissingTrustedContextException("X-User-Department-Access")
    if not x_user_permissions:
        raise MissingTrustedContextException("X-User-Permissions")

    return TrustedContext(
        department_access=_parse_department_access(x_user_department_access),
        permissions=_parse_permissions(x_user_permissions),
    )


def _parse_department_access(raw: str) -> list[DepartmentAccessEntry]:
    header_name = "X-User-Department-Access"
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc
    if not isinstance(entries, list):
        raise InvalidTrustedContextException(header_name, "expected a JSON array")
    try:
        return [
            DepartmentAccessEntry(
                department_id=str(entry["department_id"]),
                access_level=int(entry["access_level"]),
            )
            for entry in entries
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc


def _parse_permissions(raw: str) -> list[str]:
    header_name = "X-User-Permissions"
    try:
        permissions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc
    if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
        raise InvalidTrustedContextException(header_name, "expected a JSON array of strings")
    return permissions


async def require_document_permission(
    context: TrustedContext = Depends(get_trusted_context),
) -> TrustedContext:
    """FastAPI dependency: 403 unless the caller has DOCUMENT_ALL or DOCUMENT_CREATE.

    Returns the resolved context so callers that also need department
    membership checks (preview/chunking) don't have to depend on
    `get_trusted_context` a second time.
    """

    if _DOCUMENT_WRITE_PERMISSIONS.isdisjoint(context.permissions):
        raise InsufficientDocumentPermissionException()
    return context


def require_department_membership(department_id: str, context: TrustedContext) -> None:
    """Raise 403 unless `department_id` is present in the caller's department_access.

    Only checks membership, not `access_level` — that ceiling only has
    meaning at the embed step, which uses the fuller check in
    `app/api/v1/ingestion.py` instead of this helper.
    """

    if not any(entry.department_id == department_id for entry in context.department_access):
        raise DepartmentAccessDeniedException(department_id)
