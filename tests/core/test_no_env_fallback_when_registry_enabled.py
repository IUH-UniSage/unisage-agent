"""Architecture + runtime test — plan.md "Cutover khỏi cấu hình `.env` tĩnh" (Task 5):

Once the model registry is enabled, `get_graph_models()` (app/api/deps.py) must never fall
back to `settings.OPENAI_*` — not even when the snapshot has no ACTIVE CHAT credential. This
file has two halves:

  - An AST scan proving `settings.OPENAI_*` is referenced in exactly one place in the whole
    module: the `MODEL_REGISTRY_ENABLED == False` branch of `get_graph_models()`'s top-level
    `if`/`else`. Any new code path that reads `settings.OPENAI_*` from the registry-enabled
    branch (or anywhere else in the module) fails this test.
  - A runtime test: `MODEL_REGISTRY_ENABLED=true` + a snapshot with no CHAT credential ->
    `get_graph_models()` raises, rather than silently building a default `OpenAIChatModel`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import app.core.model_registry as model_registry
from app.api.deps import get_graph_models
from app.core.config import settings
from app.core.model_registry import ModelRegistryError, parse_snapshot

_DEPS_MODULE = Path(__file__).resolve().parents[2] / "app" / "api" / "deps.py"


def _openai_setting_lines(nodes: list[ast.AST]) -> set[int]:
    """Line numbers of every `settings.OPENAI_*` attribute access reachable from `nodes`."""

    lines: set[int] = set()
    for node in nodes:
        for candidate in ast.walk(node):
            if (
                isinstance(candidate, ast.Attribute)
                and candidate.attr.startswith("OPENAI_")
                and isinstance(candidate.value, ast.Name)
                and candidate.value.id == "settings"
            ):
                lines.add(candidate.lineno)
    return lines


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_DEPS_MODULE}")


def test_settings_openai_only_read_in_registry_disabled_branch() -> None:
    source = _DEPS_MODULE.read_text(encoding="utf-8")
    module_tree = ast.parse(source, filename=str(_DEPS_MODULE))

    all_refs = _openai_setting_lines([module_tree])
    assert all_refs, (
        "expected at least one settings.OPENAI_* reference somewhere in deps.py "
        "(the legacy MODEL_REGISTRY_ENABLED=false path) - if that path was removed, "
        "update/retire this test deliberately instead of letting it pass vacuously"
    )

    func = _find_function(module_tree, "get_graph_models")
    top_level_if = next(
        (stmt for stmt in func.body if isinstance(stmt, ast.If)), None
    )
    assert top_level_if is not None, "get_graph_models() must branch on MODEL_REGISTRY_ENABLED"

    refs_in_enabled_branch = _openai_setting_lines(top_level_if.body)
    refs_in_disabled_branch = _openai_setting_lines(top_level_if.orelse)

    assert not refs_in_enabled_branch, (
        "the MODEL_REGISTRY_ENABLED=true branch of get_graph_models() must never read "
        f"settings.OPENAI_* - found at line(s) {sorted(refs_in_enabled_branch)}"
    )
    assert refs_in_disabled_branch, (
        "the MODEL_REGISTRY_ENABLED=false branch should read settings.OPENAI_* "
        "(that's the legacy path this test is meant to fence in)"
    )
    assert all_refs == refs_in_disabled_branch, (
        "settings.OPENAI_* referenced outside get_graph_models()'s disabled branch, at line(s) "
        f"{sorted(all_refs - refs_in_disabled_branch)}"
    )


_EMPTY_SNAPSHOT_PAYLOAD: dict[str, Any] = {
    "version": 1,
    "generatedAt": "2026-09-25T03:00:00Z",
    "purposes": {"CHAT": [], "EMBEDDING": [], "EXTRACTION": []},
    "embeddingIndexIdentity": None,
}


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


def test_registry_enabled_with_no_chat_credential_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(
        model_registry, "_current_snapshot", parse_snapshot(_EMPTY_SNAPSHOT_PAYLOAD)
    )

    with pytest.raises(ModelRegistryError):
        get_graph_models()


def test_registry_enabled_with_no_snapshot_loaded_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same refusal even if `init_model_registry()` never ran at all (should not happen in
    practice - it gates startup - but `get_graph_models()` must not paper over it either)."""

    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)

    with pytest.raises(ModelRegistryError):
        get_graph_models()
