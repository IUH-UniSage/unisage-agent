"""Keep the model's internal ```json blocks out of the token stream.

The advisory prompt has the model end with a ```json {"type": "ask_user_form"}```
(or "confirmed_metadata") block when it needs more information. Those blocks
are for the graph, not the student: they must never reach the client or the
`content` persisted in Java. Parsing after streaming can't take back tokens
already sent, so this filters while streaming - including a fence split
across chunks - and hands the captured blocks to the caller.

Only ``` up to 7 characters are ever held back, so the visible stream is
delayed by a few characters at most. Every other code block passes through
untouched. Spec: docs/specs/SPEC-clarification-panel.md §4.

A model sometimes drops the fence and writes the block bare (seen 21/10/2026: the raw
`{"type":"ask_user_form",...}` reached the student). So a `{` is held back for a few
characters too: if what follows is `"type": "<one of ours>"` the object is collected until
it parses and captured, otherwise the `{` is released (`_BARE_LOOKAHEAD`). Finished,
non-streamed text goes through `take_bare_blocks` instead.
"""

import json
import re
from typing import Any

FENCE = "```"
CAPTURED_TYPES = frozenset({"ask_user_form", "confirmed_metadata", "calc_request"})
# Longest language tag we wait for before deciding a fence is not ```json.
_LANG_LOOKAHEAD = 16
_JSON_LANG = re.compile(r"json(?![a-z0-9_-])", re.IGNORECASE)
# A bare block starts like this (whitespace ignored); how many characters after a `{` we wait
# before deciding it is ordinary text, and how long a bare block may grow before we give up.
_BARE_LITERAL = '{"type":"'
_BARE_LOOKAHEAD = 40
_BARE_MAX = 20_000
_WS = re.compile(r"\s")
_TYPE_NAME = re.compile(r'([a-z_]*)("?)')


class FenceRedactor:
    def __init__(self) -> None:
        self._pending = ""  # TEXT: tail that may still turn into a fence
        self._in_fence = False
        self._opening = ""  # the "```json…" text of the open fence, to re-emit if not ours
        self._content = ""  # IN_FENCE: body collected so far
        self._bare = ""  # BARE: an unfenced block of ours collected so far
        self.captured: list[dict[str, Any]] = []

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` (plus held-back text) that is safe to show now."""

        data = self._pending + chunk
        self._pending = ""
        visible: list[str] = []
        while data:
            if self._bare:
                data = self._absorb_bare(data, visible)
                continue
            if self._in_fence:
                end = data.find(FENCE)
                if end == -1:
                    # Keep a trailing ` or `` in case the closing fence is split.
                    keep = _partial_fence_suffix(data)
                    self._content += data[: len(data) - keep]
                    self._pending = data[len(data) - keep :]
                    return "".join(visible)
                self._content += data[:end]
                data = data[end + len(FENCE) :]
                visible.append(self._close_fence())
                continue

            start = data.find(FENCE)
            brace = data.find("{")
            if brace != -1 and (start == -1 or brace < start):
                visible.append(data[:brace])
                head = data[brace:]
                kind = _bare_kind(head)
                if kind in CAPTURED_TYPES:
                    self._bare = head[:1]
                    data = self._absorb_bare(head[1:], visible)
                    continue
                if kind == "maybe":
                    self._pending = head  # not enough characters yet to tell
                    return "".join(visible)
                visible.append("{")  # ordinary text
                data = head[1:]
                continue
            if start == -1:
                keep = _partial_fence_suffix(data)
                visible.append(data[: len(data) - keep])
                self._pending = data[len(data) - keep :]
                return "".join(visible)
            visible.append(data[:start])
            rest = data[start + len(FENCE) :]
            newline = rest.find("\n")
            head = rest if newline == -1 else rest[:newline]
            if newline == -1 and len(head.strip()) < _LANG_LOOKAHEAD and not _decided(head):
                self._pending = data[start:]  # language tag not complete yet
                return "".join(visible)
            match = _JSON_LANG.match(head.lstrip())
            if match is None:
                # A real code block (or a closing fence): pass the fence line through.
                visible.append(FENCE + head + ("\n" if newline != -1 else ""))
                data = "" if newline == -1 else rest[newline + 1 :]
                continue
            lang_end = len(head) - len(head.lstrip()) + match.end()
            self._in_fence = True
            self._opening = FENCE + rest[:lang_end]
            self._content = ""
            data = rest[lang_end:]
        return "".join(visible)

    def finish(self) -> str:
        """End of stream: flush what is left. An unclosed fence that looks like one of ours
        is dropped (and captured if it parses); anything else is shown as-is."""

        tail = self._pending
        self._pending = ""
        if self._bare:
            # Stream ended inside a bare block of ours: capture if it parses, never show it;
            # whatever followed a complete block is shown.
            text = self._bare + tail
            self._bare = ""
            value, end = _try_decode(text)
            if isinstance(value, dict) and value.get("type") in CAPTURED_TYPES:
                self.captured.append(value)
                return text[end:]
            return ""
        if not self._in_fence:
            return tail
        self._content += tail
        self._in_fence = False
        body = self._content
        if any(kind in body for kind in CAPTURED_TYPES):
            parsed = _parse(body)
            if parsed is not None:
                self.captured.append(parsed)
            return ""
        return self._opening + body

    def _absorb_bare(self, data: str, visible: list[str]) -> str:
        """Add `data` to the open bare block; when the block is complete, capture it (or
        show it, if it turns out not to be ours) and return the text after it."""

        self._bare += data
        value, end = _try_decode(self._bare)
        if value is None:
            if len(self._bare) > _BARE_MAX:
                visible.append(self._bare)
                self._bare = ""
            return ""
        if isinstance(value, dict) and value.get("type") in CAPTURED_TYPES:
            self.captured.append(value)
        else:
            visible.append(self._bare[:end])
        rest = self._bare[end:]
        self._bare = ""
        return rest

    def _close_fence(self) -> str:
        self._in_fence = False
        parsed = _parse(self._content)
        if parsed is not None and parsed.get("type") in CAPTURED_TYPES:
            self.captured.append(parsed)
            return ""
        return self._opening + self._content + FENCE


_BARE_TYPE = re.compile(r'"type"\s*:\s*"(?:' + "|".join(sorted(CAPTURED_TYPES)) + r')"')


def take_bare_blocks(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Our JSON blocks written without a ``` fence, cut out of a finished (non-streamed)
    text: `text` without them, and the blocks. A model sometimes drops the fence, and the
    raw JSON would otherwise be shown to the student."""

    blocks: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    search_from = 0
    while (match := _BARE_TYPE.search(text, search_from)) is not None:
        start = text.rfind("{", 0, match.start())
        try:
            value, end = decoder.raw_decode(text, start) if start != -1 else (None, 0)
        except ValueError:
            value = None
        if not isinstance(value, dict) or value.get("type") not in CAPTURED_TYPES:
            search_from = match.end()
            continue
        blocks.append(value)
        text = (text[:start].rstrip() + "\n\n" + text[end:].lstrip()).strip()
        search_from = start
    return text, blocks


def _bare_kind(head: str) -> str:
    """For text starting with `{`: the block type if it begins `{"type": "<x>"` (whitespace
    anywhere), "maybe" while too short to tell, "no" when it is ordinary text."""

    compact = _WS.sub("", head)
    if not _BARE_LITERAL.startswith(compact[: len(_BARE_LITERAL)]):
        return "no"
    if len(compact) <= len(_BARE_LITERAL):
        return "maybe" if len(head) <= _BARE_LOOKAHEAD else "no"
    rest = compact[len(_BARE_LITERAL) :]
    match = _TYPE_NAME.match(rest)
    assert match is not None  # the pattern always matches (possibly empty)
    if match.group(2):
        return match.group(1)
    if match.end() < len(rest):
        return "no"  # a character that cannot be part of a type name
    return "maybe" if len(head) <= _BARE_LOOKAHEAD else "no"


def _try_decode(text: str) -> tuple[Any, int]:
    """`(value, end)` of the JSON value at the start of `text`, or `(None, 0)` while
    incomplete or invalid."""

    try:
        return json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None, 0


def _partial_fence_suffix(text: str) -> int:
    """How many trailing characters could be the start of ``` (0-2)."""

    for size in (2, 1):
        if text.endswith(FENCE[:size]):
            return size
    return 0


def _decided(head: str) -> bool:
    """Enough of the language tag is visible to tell json from anything else."""

    stripped = head.lstrip().lower()
    if not stripped:
        return False
    if "json".startswith(stripped):
        return False  # "j", "js", "jso": wait
    return True


def _parse(body: str) -> dict[str, Any] | None:
    text = body.strip()
    for candidate in (text, re.sub(r"//.*$", "", text, flags=re.MULTILINE)):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    return None
