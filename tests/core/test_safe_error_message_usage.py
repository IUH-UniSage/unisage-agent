"""Architecture test — plan.md "Secret redaction": `str(exc)`/`repr(exc)` must
never flow toward a DB column, Slack, or an HTTP call/response body. The one
sanctioned way to turn a provider exception into such text is
`app.core.redaction.safe_error_message()` (`app/core/redaction.py`).

This is a **best-effort, approximate** AST scan, not a sound dataflow
analysis — a pure syntax scan cannot fully prove where a string ends up once
it's assigned to a variable, returned from a function, or passed through
another layer of indirection. What it actually checks, scoped to
`app/core/llm/`, `app/worker/`, and `app/integrations/` (plan.md's named
surfaces):

  - Every `str(x)` / `repr(x)` call, where `x` is a name bound by an
    `except ... as x:` handler anywhere in the same file (approximate: not
    scope-precise — it doesn't verify `x` is still *in scope* at the call
    site, only that some handler in the file bound that name), is flagged
    UNLESS it is:
      - a direct argument to a `logger.<level>(...)`/`log.<level>(...)` call
        (any object whose attribute is a standard logging method name) — the
        logging.Filter in `app/core/logging_config.py` redacts these before
        they reach a handler, so this is a separate, already-covered surface;
      - a direct argument to `safe_error_message(...)`.
  - The same check for an exception name used inside an f-string
    (`f"...{exc}..."` / `f"...{exc!r}..."`), which stringifies it just as
    `str()`/`repr()` would, under the same two exemptions.

What this deliberately does NOT try to catch (false negatives, accepted as
the cost of staying sound rather than guessing): `exc.args`, `exc.message`,
custom `.detail`/`.body` attributes some SDKs attach, an exception object
passed whole into a dict/call and stringified later elsewhere, or a name
that only *transitively* aliases an exception (`err = exc; str(err)`). Any
new provider-exception-to-text call site added to these three directories
should route through `safe_error_message()` regardless of whether this scan
would have caught the alternative.
"""

from __future__ import annotations

import ast
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2] / "app"
_SCANNED_DIRS = (
    _APP_ROOT / "core" / "llm",
    _APP_ROOT / "worker",
    _APP_ROOT / "integrations",
)
_REDACTION_MODULE = _APP_ROOT / "core" / "redaction.py"

_LOGGING_METHOD_NAMES = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _iter_scanned_files() -> list[Path]:
    files: list[Path] = []
    for directory in _SCANNED_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return files


def _add_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _exception_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.name is not None
    }


def _is_exempt_call(node: ast.AST) -> bool:
    """True if `node` (a str()/repr() Call, or a FormattedValue) is a direct
    argument to a logging call or to `safe_error_message(...)`."""

    parent = getattr(node, "parent", None)
    if not isinstance(parent, ast.Call):
        # FormattedValue's parent is a JoinedStr, not a Call — walk one more
        # level up to the JoinedStr's own parent to find its usage context.
        if isinstance(parent, ast.JoinedStr):
            parent = getattr(parent, "parent", None)
        if not isinstance(parent, ast.Call):
            return False

    func = parent.func
    if isinstance(func, ast.Name) and func.id == "safe_error_message":
        return True
    if isinstance(func, ast.Attribute) and func.attr in _LOGGING_METHOD_NAMES:
        return True
    return False


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    _add_parents(tree)
    exc_names = _exception_names(tree)
    if not exc_names:
        return []

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id in ("str", "repr")
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in exc_names
            ):
                if not _is_exempt_call(node):
                    violations.append(
                        f"{path}:{node.lineno}: {func.id}({node.args[0].id}) used outside "
                        "logging/safe_error_message() — route provider exceptions through "
                        "app.core.redaction.safe_error_message() before they can reach "
                        "DB/Slack/HTTP"
                    )
            continue

        if isinstance(node, ast.FormattedValue):
            value = node.value
            if isinstance(value, ast.Name) and value.id in exc_names and not _is_exempt_call(node):
                violations.append(
                    f"{path}:{node.lineno}: f-string interpolates exception `{value.id}` "
                    "directly outside logging/safe_error_message() — route provider "
                    "exceptions through app.core.redaction.safe_error_message() before "
                    "they can reach DB/Slack/HTTP"
                )

    return violations


def test_no_raw_exception_text_outside_safe_error_message() -> None:
    all_violations: list[str] = []
    for path in _iter_scanned_files():
        all_violations.extend(_violations_in_file(path))

    assert not all_violations, "Unredacted exception text found:\n" + "\n".join(all_violations)


def test_redaction_module_exists() -> None:
    assert _REDACTION_MODULE.is_file()


def test_scanned_dirs_exist() -> None:
    """Guards against the scoped directories silently going stale/renamed and
    the scan quietly covering nothing."""

    for directory in _SCANNED_DIRS:
        assert directory.is_dir(), f"expected scanned directory to exist: {directory}"
