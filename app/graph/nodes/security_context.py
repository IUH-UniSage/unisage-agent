"""Security context extraction: turn the 5 gateway-injected trusted headers
into an `AcademicSecurityContext`, or a guest context if the gateway sent none
of them (no token on the request).

Clarification answers no longer go through text matching here - they arrive
as a structured panel submit (app/graph/clarification_answers.py).
"""

import json

from fastapi import Header

from app.core.errors.exceptions import InvalidTrustedContextException
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


async def parse_security_headers(
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
    x_user_code: str | None = Header(default=None),
    x_user_department_access: str | None = Header(default=None),
    x_user_permissions: str | None = Header(default=None),
) -> AcademicSecurityContext:
    """FastAPI dependency: gateway-trusted headers -> `AcademicSecurityContext`.

    Absent `X-User-Id` (the gateway injects nothing when the request had no
    token) means guest — NOT a 401; that distinction is enforced by the
    gateway itself, which returns 401 before this service is ever reached
    for an invalid/expired token. A present-but-malformed header is a 400
    (input error), never a 401, since a present header means the gateway
    already validated the token.
    """

    if x_user_id is None:
        return AcademicSecurityContext(role="KHACH")

    department_access = _parse_department_access(x_user_department_access)
    permissions = _parse_permissions(x_user_permissions)

    return AcademicSecurityContext(
        user_id=x_user_id,
        role=x_user_role or "KHACH",
        user_code=x_user_code,
        department_access=department_access,
        permissions=permissions,
    )


def _parse_department_access(raw: str | None) -> list[DepartmentAccessEntry]:
    if not raw:
        return []
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


def _parse_permissions(raw: str | None) -> list[str]:
    if not raw:
        return []
    header_name = "X-User-Permissions"
    try:
        permissions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc
    if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
        raise InvalidTrustedContextException(header_name, "expected a JSON array of strings")
    return permissions
