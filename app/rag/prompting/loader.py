"""YAML template loader for prompt templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .schema import PromptTemplates

_templates_dir: Path | None = None
_templates_cache: PromptTemplates | None = None
_known_metadata_fields_cache: list[dict[str, Any]] | None = None


def reset_templates_cache() -> None:
    """Reset the templates cache (useful for testing)."""
    global _templates_cache, _known_metadata_fields_cache
    _templates_cache = None
    _known_metadata_fields_cache = None


def get_known_metadata_fields() -> list[dict[str, Any]]:
    """Load `known_metadata_fields.json` (lazy load with caching) - the
    common-concept-to-canonical-field-name vocabulary rendered into
    `ask_user_form_guide.yaml` (see `builder.build_known_metadata_fields_section`).
    Not part of `PromptTemplates`: it's structured data, not a format-string
    template."""

    global _known_metadata_fields_cache
    if _known_metadata_fields_cache is None:
        path = Path(__file__).parent / "known_metadata_fields.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        _known_metadata_fields_cache = data["fields"]
    return _known_metadata_fields_cache


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
    agents = templates_dir / "agents"

    return PromptTemplates(
        chat_academic_advisory=_load_yaml_template(main / "chat_academic_advisory.yaml"),
        chat_multi_intent_synthesis=_load_yaml_template(main / "chat_multi_intent_synthesis.yaml"),
        chat_ticket_fallback=_load_yaml_template(main / "chat_ticket_fallback.yaml"),
        json_repair=_load_yaml_template(main / "json_repair.yaml"),
        agent_hyde_generator=_load_yaml_template(agents / "hyde_generator.yaml"),
        agent_message_classification=_load_yaml_template(agents / "message_classification.yaml"),
        agent_multi_query_decomposer=_load_yaml_template(agents / "multi_query_decomposer.yaml"),
        agent_calculation_extractor=_load_yaml_template(agents / "calculation_extractor.yaml"),
        agent_reranker_compressor=_load_yaml_template(agents / "reranker_compressor.yaml"),
        agent_multi_representation_enricher=_load_yaml_template(
            agents / "multi_representation_enricher.yaml"
        ),
        header=_load_yaml_template(common / "header.yaml"),
        academic_metadata=_load_yaml_template(common / "academic_metadata.yaml"),
        history_message=_load_yaml_template(common / "history_message.yaml"),
        security_access_control=_load_yaml_template(common / "security_access_control.yaml"),
        academic_domain_rules=_load_yaml_template(common / "academic_domain_rules.yaml"),
        response_style=_load_yaml_template(common / "response_style.yaml"),
        citation_rules=_load_yaml_template(common / "citation_rules.yaml"),
        prepared_context=_load_yaml_template(common / "prepared_context.yaml"),
        task_1=_load_yaml_template(common / "task_1.yaml"),
        task_2=_load_yaml_template(common / "task_2.yaml"),
        ask_user_form_guide=_load_yaml_template(common / "ask_user_form_guide.yaml"),
        confirmed_metadata_guide=_load_yaml_template(common / "confirmed_metadata_guide.yaml"),
        ticket_fallback=_load_yaml_template(common / "ticket_fallback.yaml"),
    )
