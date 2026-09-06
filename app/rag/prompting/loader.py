"""YAML template loader for prompt templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .schema import PromptTemplates

_templates_dir: Path | None = None
_templates_cache: PromptTemplates | None = None


def reset_templates_cache() -> None:
    """Reset the templates cache (useful for testing)."""
    global _templates_cache
    _templates_cache = None


def get_templates() -> PromptTemplates:
    """Get loaded templates (lazy load with caching)."""
    global _templates_cache
    if _templates_cache is None:
        _templates_cache = _load_all_templates()
    return _templates_cache


def _get_templates_dir() -> Path:
    global _templates_dir
    if _templates_dir is None:
        _templates_dir = Path(__file__).parent / "prompt_templates"
    return _templates_dir


def _load_yaml_file(file_path: Path) -> Any:
    content = file_path.read_text(encoding="utf-8")
    return yaml.safe_load(content)


def _load_yaml_template(file_path: Path) -> str:
    """Read a template YAML, taking its `template`/`content` field."""

    data = _load_yaml_file(file_path)
    if not isinstance(data, dict):
        raise ValueError(f"Template YAML không phải dict: {file_path.name}")
    value = data.get("template") or data.get("content")
    if not isinstance(value, str):
        raise ValueError(f"Template YAML thiếu key 'template'/'content': {file_path.name}")
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def _load_all_templates() -> PromptTemplates:
    templates_dir = _get_templates_dir()
    common = templates_dir / "common"
    main = templates_dir / "main"

    return PromptTemplates(
        chat_academic_advisory=_load_yaml_template(main / "chat_academic_advisory.yaml"),
        chat_direct_llm=_load_yaml_template(main / "chat_direct_llm.yaml"),
        header=_load_yaml_template(common / "header.yaml"),
        academic_metadata=_load_yaml_template(common / "academic_metadata.yaml"),
        security_access_control=_load_yaml_template(common / "security_access_control.yaml"),
        academic_domain_rules=_load_yaml_template(common / "academic_domain_rules.yaml"),
        response_style=_load_yaml_template(common / "response_style.yaml"),
        citation_rules=_load_yaml_template(common / "citation_rules.yaml"),
        prepared_context=_load_yaml_template(common / "prepared_context.yaml"),
        task_1=_load_yaml_template(common / "task_1.yaml"),
        task_2=_load_yaml_template(common / "task_2.yaml"),
        ask_user_form_guide=_load_yaml_template(common / "ask_user_form_guide.yaml"),
    )
