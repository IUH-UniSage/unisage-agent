"""Architecture test — every provider HTTP client must be built through
`app/core/llm/http_client.py`'s factory, never with the SDK's own default
transport.

AST-scans every `.py` file under `app/` for:
  - `OpenAI(`, `AsyncOpenAI(`, `Anthropic(` calls that don't pass an explicit
    `http_client=` keyword — that keyword is only ever satisfied by
    `build_provider_http_client`/`build_provider_http_client_sync`, so its
    presence is what "goes through the factory" means at a call site. A call
    with no `http_client=` gets the SDK's own unguarded default transport,
    which is exactly the bare-client pattern this feature bans.
  - `httpx.Client(`/`httpx.AsyncClient(` constructions outside the factory
    module itself. `app/integrations/backend_java_client.py` and
    `app/integrations/slack_notifier.py` are scoped exceptions: each builds a
    plain `httpx.AsyncClient` to call an operator-configured external
    service, not a provider — SSRF pinning is a defense for URLs the SA
    registers as a provider endpoint (see `http_client.py`'s own module
    docstring), and both `BACKEND_JAVA_BASE_URL`
    and `SLACK_APIKEY_ALERT_WEBHOOK_URL` are operator-configured infra, not
    registry data.
  - Any `import litellm` / `from litellm import ...` anywhere in `app/` — ADR
    0005 rejected LiteLLM as the provider-calling SDK (couldn't inject a
    pinned transport). Model prices come from backend-java
    (`app/core/pricing/snapshot.py`, ADR 0006 in unisage-backend), so there is
    no remaining reason to import it at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_APP_ROOT = Path(__file__).resolve().parents[2] / "app"
_FACTORY_MODULE = _APP_ROOT / "core" / "llm" / "http_client.py"

# Scoped, understood exception — see module docstring. Not a provider call site.
_HTTPX_CLIENT_ALLOWED_FILES = {
    _APP_ROOT / "integrations" / "backend_java_client.py",
    _APP_ROOT / "integrations" / "slack_notifier.py",
}

_SDK_CLIENT_NAMES = {"OpenAI", "AsyncOpenAI", "Anthropic"}
_HTTPX_CLIENT_ATTRS = {"Client", "AsyncClient"}


def _iter_app_py_files() -> list[Path]:
    return sorted(_APP_ROOT.rglob("*.py"))


def _call_name(call: ast.Call) -> str | None:
    """Best-effort name of the thing being called, e.g. `OpenAI` for both
    `OpenAI(...)` and `openai.OpenAI(...)`."""

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_httpx_client_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _HTTPX_CLIENT_ATTRS:
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id == "httpx"


def _has_http_client_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "http_client" for kw in call.keywords)


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            names = [alias.name for alias in node.names]
            if module == "litellm" or module.startswith("litellm.") or "litellm" in names:
                violations.append(f"{path}:{node.lineno}: `import litellm` is banned (ADR 0005)")
            continue

        if not isinstance(node, ast.Call):
            continue

        if _is_httpx_client_call(node):
            if path != _FACTORY_MODULE and path not in _HTTPX_CLIENT_ALLOWED_FILES:
                violations.append(
                    f"{path}:{node.lineno}: raw httpx.Client/AsyncClient construction "
                    "outside app/core/llm/http_client.py"
                )
            continue

        name = _call_name(node)
        if name in _SDK_CLIENT_NAMES and not _has_http_client_kwarg(node):
            violations.append(
                f"{path}:{node.lineno}: {name}(...) built without an explicit "
                "http_client= from the SSRF-guarded factory"
            )

    return violations


def test_no_raw_provider_clients_outside_factory() -> None:
    all_violations: list[str] = []
    for path in _iter_app_py_files():
        all_violations.extend(_violations_in_file(path))

    assert not all_violations, "Raw provider client construction found:\n" + "\n".join(
        all_violations
    )


def test_factory_module_exists() -> None:
    """Guards against the allowlist/exemption paths above silently going stale
    (e.g. the factory file gets renamed and every call site starts being flagged
    -- or worse, stops being checked at all)."""

    assert _FACTORY_MODULE.is_file()


@pytest.mark.parametrize("path", sorted(_HTTPX_CLIENT_ALLOWED_FILES))
def test_httpx_client_allowlist_entries_exist(path: Path) -> None:
    assert path.is_file(), f"allowlisted file no longer exists: {path}"
