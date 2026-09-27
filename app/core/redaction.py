"""Redacts secrets from arbitrary text (provider exception messages, health-report
bodies, anything that might end up in a DB column / Slack payload / HTTP response)
before it leaves the process. Matches plan.md "Secret redaction" — must behave
identically to the Java side (`unisage-backend/.../utils/SecretRedactor.java`), and
the shared vector file `unisage-backend/contracts/redaction-vectors.json` (vendored
at `contracts/vendor/redaction-vectors.json` — see `tests/core/test_redaction.py`)
is the single source of truth both repos test against.

Order matters, same as Java: redact the exact known secret (and its substrings
>= 8 chars) first, then the generic header/prefix/query/userinfo patterns, and only
then truncate to 500 chars — truncating first could split a secret exactly at the
boundary and let half of it survive.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"
_MAX_LENGTH = 500
_MIN_SUBSTRING_LENGTH = 8

# Authorization: <scheme> <token> — case-insensitive header name, requires the colon
# so unrelated identifiers like "AuthorizationError" are never matched; stops at
# whitespace/quote/comma/end so it doesn't eat trailing prose.
_AUTHORIZATION_HEADER = re.compile(r"(?i)Authorization\s*:\s*(Bearer\s+)?[A-Za-z0-9._~+/-]+=*")
_BEARER_TOKEN = re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+=*")
# x-api-key / api-key header or JSON-ish field, with optional colon/equals separator
# and optional quoting.
_API_KEY_HEADER = re.compile(r'(?i)(x-api-key|api-key)\s*[:=]\s*"?[A-Za-z0-9._~+/-]+=*"?')
# Raw provider key prefixes (OpenAI-style `sk-...`, Anthropic `sk-ant-...`). Match the
# prefix plus the following token run so the key body is swallowed, not just the prefix.
_RAW_KEY_PREFIX = re.compile(r"sk-ant-[A-Za-z0-9._-]+|sk-[A-Za-z0-9._-]+")
# Query params: key=, api_key=, token= up to the next & / whitespace / quote.
_QUERY_PARAM = re.compile(r"""(?i)([?&])(key|api_key|token)=[^&\s"'#]*""")
# URL userinfo: scheme://user:pass@ — redact the user:pass portion, keep scheme:// and host.
_URL_USERINFO = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


def redact(text: str | None, known_secret: str | None = None, *, truncate: bool = True) -> str:
    """Redacts `text` and, by default, truncates to 500 chars.

    Args:
        text: raw text that may contain a secret (exception message, response
            body, log message, traceback, ...). `None`/empty returns `""`.
        known_secret: the exact API key of the credential currently being
            processed, or `None`/blank if not applicable. Redacted first,
            along with any of its substrings of length >= 8 — the strongest,
            format-independent layer.
        truncate: `False` skips the 500-char cap. The cap exists for text
            destined for bounded storage (a DB column, a Slack payload, an
            HTTP response) — every such call site keeps the default. A local
            console log line isn't bounded storage; truncating a multi-line
            traceback there just throws away the exception type/message that
            usually sits at the very end, which is the opposite of what
            logging it was for (`logging_config.py`'s `SecretRedactionFilter`
            passes `truncate=False` for exactly this reason).

    Returns:
        Redacted text, truncated to 500 chars unless `truncate=False`. Never `None`.
    """

    if not text:
        return ""

    result = text
    if known_secret and known_secret.strip():
        result = _redact_known_secret(result, known_secret)
    result = _redact_patterns(result)

    if truncate and len(result) > _MAX_LENGTH:
        result = result[:_MAX_LENGTH]
    return result


def _redact_known_secret(text: str, secret: str) -> str:
    """Replaces the exact secret, and every contiguous substring of it that is
    >= 8 chars, with `[REDACTED]`. Rather than enumerating every substring length
    (quadratic), this relies on the fact that any surviving substring of length
    >= 8 must contain at least one length-8 sliding window of the secret aligned
    to the same offsets — so redacting every such window is sufficient to
    guarantee no >=8-char fragment survives, in linear-in-secret-length passes.
    """

    trimmed = secret.strip()
    if not trimmed:
        return text

    if len(trimmed) < _MIN_SUBSTRING_LENGTH:
        # Too short to safely window-match (would nuke unrelated short text) —
        # still redact the exact secret itself; the generic patterns below
        # cover the rest.
        return text.replace(trimmed, REDACTED) if trimmed in text else text

    result = text.replace(secret, REDACTED) if secret in text else text
    for start in range(0, len(trimmed) - _MIN_SUBSTRING_LENGTH + 1):
        window = trimmed[start : start + _MIN_SUBSTRING_LENGTH]
        if window in result:
            result = result.replace(window, REDACTED)
    return result


def _redact_patterns(text: str) -> str:
    result = text
    result = _AUTHORIZATION_HEADER.sub(REDACTED, result)
    result = _BEARER_TOKEN.sub(REDACTED, result)
    result = _API_KEY_HEADER.sub(REDACTED, result)
    result = _RAW_KEY_PREFIX.sub(REDACTED, result)
    result = _QUERY_PARAM.sub(lambda m: f"{m.group(1)}{m.group(2)}={REDACTED}", result)
    result = _URL_USERINFO.sub(rf"\1{REDACTED}@", result)
    return result


def safe_error_message(exc: BaseException, credential: str | None = None) -> str:
    """The one sanctioned way to turn a provider exception into text that may
    reach a DB column, Slack, or an HTTP response/request body (plan.md
    "Secret redaction" — "Không bao giờ lưu/gửi `str(exc)` hay `repr(exc)` thô").

    `credential` should be the exact API key of the credential that produced
    `exc`, when known (e.g. the candidate key being verified, or the ACTIVE
    row's key) — passing it lets `redact()` catch a raw key even if it shows up
    in a shape none of the generic patterns recognize (a provider SDK echoing
    it back in a non-standard field, for instance).

    Every call site in `app/core/llm/`, `app/worker/`, `app/integrations/` that
    turns a provider exception into text destined for DB/Slack/HTTP must go
    through this function instead of `str(exc)`/`repr(exc)` —
    `tests/core/test_safe_error_message_usage.py` enforces this with a
    best-effort AST scan.
    """

    text = f"{type(exc).__name__}: {exc}"
    return redact(text, credential)
