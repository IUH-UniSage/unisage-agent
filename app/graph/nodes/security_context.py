"""Security context extraction and clarification-guard matching.

Two responsibilities, kept as two separate functions rather than split into
two graph nodes:

- `parse_security_headers` — turn the 5 gateway-injected trusted headers into
  an `AcademicSecurityContext`, or a guest context if the gateway sent none
  of them (no token on the request).
- `resolve_clarification_guard` — deterministic (no LLM) match of the
  current message against a `PendingClarification`'s `options`.
"""

import json
import unicodedata
from dataclasses import dataclass

from fastapi import Header

from app.core.exceptions import InvalidTrustedContextException
from app.schemas.clarification import PendingClarification
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


def _normalize_for_match(text: str) -> str:
    """Diacritic/case/underscore-insensitive normal form for hand-typed replies.

    "Chính quy ạ" and "chinh_quy" both normalize to "chinh quy a"/"chinh quy"
    (word boundaries collapse to single spaces) so a typed reply can match
    an option id without requiring the user to type the id verbatim.
    """

    normalized = unicodedata.normalize("NFD", text)
    without_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_diacritics.replace("_", " ").lower().split())


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


@dataclass(frozen=True)
class ClarificationGuardResult:
    """Outcome of matching one user reply against a `PendingClarification`."""

    matched: bool
    route_to_origin: bool
    origin_node: str | None
    confirmed_metadata: dict[str, str]
    pending_clarification: PendingClarification | None
    skip_classification: bool


def resolve_clarification_guard(
    *,
    user_message: str,
    pending: PendingClarification | None,
    confirmed_metadata: dict[str, str],
    max_retry: int,
) -> ClarificationGuardResult:
    """Deterministic match/no-match/discard.

    - No pending clarification: pass through unchanged, normal flow.
    - Reply matches one of `pending.options` (id, or diacritic/case-insensitive
      label match): write into `confirmed_metadata`, clear pending, route back
      to `pending.origin_node`, skip message classification.
    - No match, `retry_count + 1 < max_retry`: bump `retry_count`, keep
      asking (stay at the same origin — caller re-renders the same question).
    - No match, retry limit reached: discard pending, fall through to the
      normal flow (generation must pick a safe answer covering all branches).
    """

    if pending is None:
        return ClarificationGuardResult(
            matched=False,
            route_to_origin=False,
            origin_node=None,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=None,
            skip_classification=False,
        )

    matched_field, matched_value = _match_reply(user_message, pending)
    if matched_field is not None and matched_value is not None:
        new_confirmed = dict(confirmed_metadata)
        new_confirmed[matched_field] = matched_value
        return ClarificationGuardResult(
            matched=True,
            route_to_origin=True,
            origin_node=pending.origin_node,
            confirmed_metadata=new_confirmed,
            pending_clarification=None,
            skip_classification=True,
        )

    next_retry_count = pending.retry_count + 1
    if next_retry_count >= max_retry:
        return ClarificationGuardResult(
            matched=False,
            route_to_origin=False,
            origin_node=None,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=None,
            skip_classification=False,
        )

    return ClarificationGuardResult(
        matched=False,
        route_to_origin=False,
        origin_node=None,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending.model_copy(update={"retry_count": next_retry_count}),
        skip_classification=False,
    )


def _match_reply(user_message: str, pending: PendingClarification) -> tuple[str | None, str | None]:
    """Try to match `user_message` against any field's options.

    Only single-field forms (the common case) are matched by free text; a
    field with `options=None` (free-text field) is not resolved here —
    deterministic matching only covers option chips, free-text extraction is
    out of scope for this guard.
    """

    normalized_reply = _normalize_for_match(user_message)
    for field, options in zip(pending.missing_fields, pending.options, strict=True):
        if options is None:
            continue
        for option_id in options:
            normalized_option = _normalize_for_match(option_id)
            if not normalized_option:
                continue
            # Exact match (UI chip click sends the id verbatim) or the
            # option's words all appear as a contiguous phrase in the reply
            # (hand-typed reply, possibly with extra words like "... ạ").
            if normalized_reply == normalized_option or f" {normalized_option} " in (
                f" {normalized_reply} "
            ):
                return field, option_id
    return None, None
