"""One name per concept for everything a student declares or types on a form.

`known_metadata_fields.json` is the registry: each field is either `profile` (a
lasting attribute of the student, kept in `student_profiles`) or `calculation` (a
number of one calculation, kept with that calculation). Every field name - a form
question, an answer, a correction from the classifier - goes through
`canonical_field` first, so "khoa_tuyen_sinh" and "khoa_nhap_hoc" can never hold
two values for the same thing.
"""

import re
import unicodedata
from collections.abc import Mapping
from datetime import date
from functools import cache
from typing import Literal

from app.core.academic_calendar import INTAKE_FIELD, parse_intake_year
from app.rag.prompting.loader import get_known_metadata_fields

Scope = Literal["profile", "calculation"]
VALUE_MAX_LENGTH = 200


def slug(text: str) -> str:
    """ASCII id from Vietnamese text: "Đơn giá" -> "don_gia"."""

    ascii_text = (
        unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9_-]+", "_", ascii_text).strip("_")[:64]


def canonical_field(name: str) -> str:
    """The registry name for `name` or one of its key aliases; any other name slugged."""

    key = slug(name)
    return _alias_index().get(key, key)


def registered_name(name: str) -> str:
    """The registry name for a registered field or alias; `name` unchanged otherwise
    (a label such as "Số tín chỉ lý thuyết của một năm học" stays readable)."""

    return _alias_index().get(slug(name), name)


def concept(name: str) -> str:
    """The human name of a field ("Khóa nhập học / ..."), for the update notice."""

    entry = _entries().get(canonical_field(name))
    return str(entry["concept"]).split(" / ")[0] if entry else name


def field_scope(name: str, default: Scope) -> Scope:
    """`default` for a field the registry does not know: the advisory flow's own
    branch attributes are profile, an LLM calculation's own inputs are calculation."""

    entry = _entries().get(canonical_field(name))
    if entry is None:
        return default
    return "profile" if entry["scope"] == "profile" else "calculation"


def is_profile_field(name: str) -> bool:
    entry = _entries().get(canonical_field(name))
    return entry is not None and entry["scope"] == "profile"


def normalize_value(name: str, value: object, today: date) -> str | None:
    """A profile value as stored: the intake as a year, anything else as trimmed text.
    None when the value says nothing (empty, or an intake that cannot be read)."""

    field = canonical_field(name)
    if field == INTAKE_FIELD:
        intake = parse_intake_year(value, today)
        return str(intake) if intake is not None else None
    if value is None or isinstance(value, bool | dict | list):
        return None
    text = " ".join(str(value).split())[:VALUE_MAX_LENGTH]
    return text or None


def apply_updates(
    profile: Mapping[str, str], updates: Mapping[str, object], today: date
) -> tuple[dict[str, str], dict[str, str | None]]:
    """(the new profile, what changed: field -> new value, None = removed). Only
    profile fields of the registry are touched; anything else is ignored."""

    merged = dict(profile)
    changed: dict[str, str | None] = {}
    for name, raw in updates.items():
        field = canonical_field(name)
        if not is_profile_field(field):
            continue
        if raw is None:
            if field in merged:
                del merged[field]
                changed[field] = None
            continue
        value = normalize_value(field, raw, today)
        if value is not None and merged.get(field) != value:
            merged[field] = value
            changed[field] = value
    return merged, changed


def update_notice(changed: Mapping[str, str | None]) -> str:
    """The fixed line a reply opens with when the turn changed the profile, so the
    student sees what was saved (and can correct it)."""

    parts = [
        f"{_lower_first(concept(name))}: {value}"
        if value is not None
        else f"bỏ {_lower_first(concept(name))}"
        for name, value in changed.items()
    ]
    return f"Mình đã cập nhật thông tin của bạn - {'; '.join(parts)}."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def canonical_profile(profile: Mapping[str, str], today: date) -> dict[str, str]:
    """A stored profile under registry names with normalised values - rows written
    before the registry (or under an alias) read back the same as new ones."""

    result: dict[str, str] = {}
    for name, raw in profile.items():
        field = canonical_field(name)
        value = normalize_value(field, raw, today)
        if value is not None:
            result.setdefault(field, value)
    return result


def canonical_params(params: Mapping[str, object]) -> dict[str, object]:
    """Known values keyed by registry names (the first name wins on a clash)."""

    result: dict[str, object] = {}
    for name, value in params.items():
        result.setdefault(canonical_field(name), value)
    return result


@cache
def _entries() -> dict[str, Mapping[str, object]]:
    return {str(entry["field"]): entry for entry in get_known_metadata_fields()}


@cache
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for field, entry in _entries().items():
        index[field] = field
        aliases = entry.get("key_aliases")
        for alias in aliases if isinstance(aliases, list) else []:
            index.setdefault(slug(str(alias)), field)
    return index
