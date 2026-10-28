"""Judge providers: one adapter per provider behind the `base.JudgeAdapter` interface, picked
by the environment.

    JUDGE_PROVIDER            gemini (default) | openai_compat   (--judge-provider overrides)
    JUDGE_EMBEDDING_PROVIDER  who embeds for answer_relevancy (--embedding-provider overrides);
                              default: the judge provider if it can embed, else gemini

Settings are read from the process environment first, then from `--env-file` (`.env.eval`).
Adapters are imported only when picked: the Gemini one needs `google-genai`, which only the
judge environment has.
"""

import importlib
import os
from collections.abc import Mapping
from pathlib import Path

from evals.judge.providers.base import JudgeAdapter, MissingSetting, Models, Options

PROVIDER_ENV = "JUDGE_PROVIDER"
EMBEDDING_PROVIDER_ENV = "JUDGE_EMBEDDING_PROVIDER"
DEFAULT_PROVIDER = "gemini"
ADAPTERS = {
    "gemini": "evals.judge.providers.gemini:GeminiAdapter",
    "openai_compat": "evals.judge.providers.openai_compat:OpenAICompatAdapter",
}

__all__ = [
    "ADAPTERS",
    "JudgeAdapter",
    "MissingSetting",
    "Models",
    "Options",
    "adapter_class",
    "embedding_provider",
    "judge_provider",
    "read_env",
]


def read_env(env_file: Path | None) -> dict[str, str]:
    """`env_file` (KEY=value lines, # comments) overlaid by the process environment."""

    values: dict[str, str] = {}
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name and not name.startswith("#"):
                values[name.strip()] = value.strip().strip('"').strip("'")
    values.update({name: value for name, value in os.environ.items() if value})
    return values


def _check(name: str) -> str:
    if name not in ADAPTERS:
        raise ValueError(f"unknown judge provider {name!r} (known: {', '.join(ADAPTERS)})")
    return name


def judge_provider(env: Mapping[str, str], override: str | None = None) -> str:
    return _check((override or env.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip())


def adapter_class(name: str) -> type[JudgeAdapter]:
    module, _, attr = ADAPTERS[_check(name)].partition(":")
    return getattr(importlib.import_module(module), attr)


def embedding_provider(
    env: Mapping[str, str], judge: str, judge_models: Models, override: str | None = None
) -> str:
    chosen = override or env.get(EMBEDDING_PROVIDER_ENV)
    if chosen:
        return _check(chosen.strip())
    return judge if judge_models.embedding else DEFAULT_PROVIDER
