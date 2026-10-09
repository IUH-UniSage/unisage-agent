"""Prompt templates schema definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptTemplates:
    """Loaded YAML templates, cached in memory by `loader.get_templates()`."""

    # Main (per-node system prompts)
    chat_academic_advisory: str
    chat_multi_intent_synthesis: str
    chat_ticket_fallback: str
    chat_calculation: str
    json_repair: str

    # Agents (system prompts of single-purpose LLM nodes)
    agent_hyde_generator: str
    agent_message_classification: str
    agent_message_classification_retrieval: str
    agent_multi_query_decomposer: str
    agent_calculation_extractor: str
    agent_calculation_formula: str
    agent_calculation_formula_verifier: str
    agent_reranker_compressor: str
    agent_multi_representation_enricher: str

    # Common components
    header: str
    academic_metadata: str
    history_message: str
    security_access_control: str
    academic_domain_rules: str
    response_style: str
    citation_rules: str
    prepared_context: str
    web_search_context: str
    task_1: str
    task_2: str
    ask_user_form_guide: str
    ticket_fallback: str

    # sha256[:12] of each template's text, by attribute name - the prompt version
    # recorded in calculation traces.
    versions: Mapping[str, str] = field(default_factory=dict)
