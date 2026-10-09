"""Unified `thinking` setting, adjusted per model.

PydanticAI silently drops `thinking=False` for models whose profile says they always reason
(`thinking_always_enabled`: the original gpt-5 family incl. -mini, the o-series, deepseek-r1,
...): the request then goes out with no effort at all and the model reasons at its default
(`medium` for gpt-5-mini). Asking for `'minimal'` instead gets the lowest level it accepts -
PydanticAI itself maps `'minimal'` down to `'low'` for models that lack it.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ThinkingLevel


def resolve_thinking(model: Model | str, thinking: ThinkingLevel) -> ThinkingLevel:
    """`thinking` for this model: `False` becomes `'minimal'` when the model can't turn
    reasoning off. A bare model name string has no resolved profile, so it is left as-is."""

    if thinking is False and isinstance(model, Model):
        if model.profile.get("thinking_always_enabled", False):
            return "minimal"
    return thinking
