"""Prompt templates schema definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplates:
    """Loaded YAML templates, cached in memory by `loader.get_templates()`."""

    # Main (per-node system prompts)
    chat_academic_advisory: str
    chat_direct_llm: str
    json_repair: str

    # Common components
    header: str
    academic_metadata: str
    security_access_control: str
    academic_domain_rules: str
    response_style: str
    citation_rules: str
    prepared_context: str
    task_1: str
    task_2: str
    ask_user_form_guide: str
    confirmed_metadata_guide: str
