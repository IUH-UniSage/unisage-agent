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
"""

import json
import re
from typing import Any

FENCE = "```"
CAPTURED_TYPES = frozenset({"ask_user_form", "confirmed_metadata"})
# Longest language tag we wait for before deciding a fence is not ```json.
_LANG_LOOKAHEAD = 16
_JSON_LANG = re.compile(r"json(?![a-z0-9_-])", re.IGNORECASE)


class FenceRedactor:
    def __init__(self) -> None:
        self._pending = ""  # TEXT: tail that may still turn into a fence
        self._in_fence = False
        self._opening = ""  # the "```json…" text of the open fence, to re-emit if not ours
        self._content = ""  # IN_FENCE: body collected so far
        self.captured: list[dict[str, Any]] = []

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` (plus held-back text) that is safe to show now."""

        data = self._pending + chunk
        self._pending = ""
        visible: list[str] = []
        while data:
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

    def _close_fence(self) -> str:
        self._in_fence = False
        parsed = _parse(self._content)
        if parsed is not None and parsed.get("type") in CAPTURED_TYPES:
            self.captured.append(parsed)
            return ""
        return self._opening + self._content + FENCE


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
