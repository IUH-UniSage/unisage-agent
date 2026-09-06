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
import re
import unicodedata
from dataclasses import dataclass

from fastapi import Header

from app.core.exceptions import InvalidTrustedContextException
from app.schemas.clarification import PendingClarification
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry

_PUNCTUATION_PATTERN = re.compile(r"[,;:.!?]")
# U+0111/U+0110 ("đ"/"Đ") are standalone Unicode code points, not a base
# letter + combining mark - NFD does NOT decompose them, so the diacritic
# strip below leaves "đ" untouched (e.g. "đại" -> "đai", not "dai") unless
# handled explicitly here first.
_DJ_STROKE_PATTERN = re.compile(r"[đĐ]")


def _normalize_for_match(text: str) -> str:
    """Diacritic/case/underscore/punctuation-insensitive normal form for
    hand-typed replies.

    "Chính quy ạ" and "chinh_quy" both normalize to "chinh quy a"/"chinh quy"
    (word boundaries collapse to single spaces) so a typed reply can match
    an option id without requiring the user to type the id verbatim.
    Punctuation is stripped (not just collapsed to a boundary) so a reply
    answering several fields at once, e.g. "Chính quy, K21, Cử nhân", isn't
    broken into "k21," and "cu nhan" fragments that fail the surrounding-
    space substring check below.
    """

    without_dj_stroke = _DJ_STROKE_PATTERN.sub("d", text)
    normalized = unicodedata.normalize("NFD", without_dj_stroke)
    without_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_punctuation = _PUNCTUATION_PATTERN.sub(" ", without_diacritics)
    return " ".join(without_punctuation.replace("_", " ").lower().split())


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
    """Outcome of matching one user reply against a `PendingClarification`.

    `original_query` is the question that started this round (see
    `PendingClarification.original_query`), carried separately from
    `pending_clarification` because a fully-resolved reply clears the latter
    to `None` - exactly the turn retrieval most needs to know what topic to
    search for, since `user_message` this turn is just the reply, not a
    question. `None` only when there was no active round this turn at all.
    """

    matched: bool
    route_to_origin: bool
    origin_node: str | None
    confirmed_metadata: dict[str, str]
    pending_clarification: PendingClarification | None
    skip_classification: bool
    original_query: str | None = None


def resolve_clarification_guard(
    *,
    user_message: str,
    pending: PendingClarification | None,
    confirmed_metadata: dict[str, str],
    max_retry: int,
) -> ClarificationGuardResult:
    """Deterministic match/partial-match/no-match/discard.

    - No pending clarification: pass through unchanged, normal flow.
    - Reply resolves EVERY field still in `pending.missing_fields`: write
      them all into `confirmed_metadata`, clear pending, route back to
      `pending.origin_node`, skip message classification.
    - Reply resolves SOME but not all fields (a multi-field form's reply
      can answer several at once, e.g. "Chính quy, K21, Cử nhân" for a
      3-field form): write the resolved ones into `confirmed_metadata`,
      keep asking for the rest only (`retry_count` reset to 0 - partial
      progress isn't a failed attempt).
    - No field resolved, `retry_count + 1 < max_retry`: bump `retry_count`,
      keep asking (stay at the same origin - caller re-renders the same
      question).
    - No field resolved, retry limit reached: discard pending, fall through
      to the normal flow (generation must pick a safe answer covering all
      branches).
    """

    if pending is None:
        return ClarificationGuardResult(
            matched=False,
            route_to_origin=False,
            origin_node=None,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=None,
            skip_classification=False,
            original_query=None,
        )

    original_query = pending.original_query or None

    resolved = _match_reply(user_message, pending)
    if resolved:
        new_confirmed = dict(confirmed_metadata)
        new_confirmed.update(resolved)
        labels_per_field = _labels_aligned_to_fields(pending)
        remaining = [
            (field, options, labels)
            for field, options, labels in zip(
                pending.missing_fields, pending.options, labels_per_field, strict=True
            )
            if field not in resolved
        ]
        if not remaining:
            return ClarificationGuardResult(
                matched=True,
                route_to_origin=True,
                origin_node=pending.origin_node,
                confirmed_metadata=new_confirmed,
                pending_clarification=None,
                skip_classification=True,
                original_query=original_query,
            )
        return ClarificationGuardResult(
            matched=True,
            route_to_origin=True,
            origin_node=pending.origin_node,
            confirmed_metadata=new_confirmed,
            pending_clarification=pending.model_copy(
                update={
                    "missing_fields": [field for field, _options, _labels in remaining],
                    "options": [options for _field, options, _labels in remaining],
                    # Narrow the labels in lockstep - leaving the full list
                    # behind would desync it from `missing_fields` and break
                    # label matching (and the strict zip) on the next turn.
                    "option_labels": [labels for _field, _options, labels in remaining],
                    "retry_count": 0,
                }
            ),
            skip_classification=True,
            original_query=original_query,
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
            original_query=None,
        )

    return ClarificationGuardResult(
        matched=False,
        route_to_origin=False,
        origin_node=None,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending.model_copy(update={"retry_count": next_retry_count}),
        skip_classification=False,
        original_query=original_query,
    )


def _labels_aligned_to_fields(pending: PendingClarification) -> list[list[str] | None]:
    """`pending.option_labels` padded/truncated to line up with
    `missing_fields`.

    The field is optional (`None` on rows persisted before it existed) and a
    narrowing `model_copy` elsewhere could always leave it out of sync, so
    every read goes through here rather than zipping it raw - a length
    mismatch would otherwise raise on `strict=True` mid-conversation.
    """

    field_count = len(pending.missing_fields)
    labels = pending.option_labels or []
    if len(labels) == field_count:
        return list(labels)
    return list(labels[:field_count]) + [None] * max(0, field_count - len(labels))


def _match_reply(user_message: str, pending: PendingClarification) -> dict[str, str]:
    """Match `user_message` against every field still pending, returning ALL
    fields resolved by this one reply - a multi-field form's reply may
    answer several at once (e.g. "Chính quy, K21, Cử nhân" for a 3-field
    form), not just the first field checked.

    Each option is matched by BOTH its id and its display label (when
    `pending.option_labels` carries one): a hand-typed reply almost always
    uses the label the form showed - "Công nghệ Thông tin" - while the id is
    an internal slug like "cntt" that only a UI chip click sends back.
    Matching ids alone made every typed reply miss here, burn a retry, and
    depend on the LLM fallback to rescue the round. Whichever form matches,
    the canonical ID is what gets written into `confirmed_metadata`.

    A field with `options=None` (free-text field, e.g. a score) is not
    resolved here - deterministic matching only covers option chips,
    free-text extraction is out of scope for this guard.
    """

    normalized_reply = _normalize_for_match(user_message)
    padded_reply = f" {normalized_reply} "
    resolved: dict[str, str] = {}
    labels_per_field = _labels_aligned_to_fields(pending)
    for field, options, labels in zip(
        pending.missing_fields, pending.options, labels_per_field, strict=True
    ):
        if options is None:
            continue
        for index, option_id in enumerate(options):
            candidates = [option_id]
            if labels is not None and index < len(labels):
                candidates.append(labels[index])
            for candidate in candidates:
                normalized_option = _normalize_for_match(candidate)
                if not normalized_option:
                    continue
                # Exact match (UI chip click sends the id verbatim) or the
                # option's words all appear as a contiguous phrase in the
                # reply (hand-typed, possibly with extra words like "... ạ").
                if (
                    normalized_reply == normalized_option
                    or f" {normalized_option} " in padded_reply
                ):
                    resolved[field] = option_id
                    break
            if field in resolved:
                break
    return resolved
