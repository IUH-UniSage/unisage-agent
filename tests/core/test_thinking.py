"""`resolve_thinking` asks always-reasoning models for their lowest effort instead of `False`,
which PydanticAI would otherwise drop silently."""

from __future__ import annotations

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.llm.thinking import resolve_thinking


def _openai(model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key="test-key"))


@pytest.mark.parametrize("model_name", ["gpt-5-mini", "gpt-5", "o3-mini"])
def test_always_reasoning_model_gets_minimal_instead_of_false(model_name: str) -> None:
    assert resolve_thinking(_openai(model_name), False) == "minimal"


@pytest.mark.parametrize("model_name", ["gpt-5.1", "gpt-4o"])
def test_model_that_can_turn_reasoning_off_keeps_false(model_name: str) -> None:
    assert resolve_thinking(_openai(model_name), False) is False


def test_wrapped_model_reads_the_inner_profile() -> None:
    assert resolve_thinking(WrapperModel(_openai("gpt-5-mini")), False) == "minimal"


@pytest.mark.parametrize("thinking", [True, "low", "high"])
def test_explicit_level_is_left_alone(thinking: bool | str) -> None:
    assert resolve_thinking(_openai("gpt-5-mini"), thinking) == thinking


def test_model_name_string_and_test_double_are_left_alone() -> None:
    assert resolve_thinking("openai:gpt-5-mini", False) is False
    assert resolve_thinking(TestModel(), False) is False
