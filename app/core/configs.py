"""`configs/` at the repository root: the YAML the code reads at runtime (web search
vocabulary, eval label map), kept apart from the Python that reads it so it
can be edited without touching code. Prompt templates stay beside their loader
(app/rag/prompting/prompt_templates/), and files a tool expects in its own place
(Taskfile, docker compose, GitHub workflows) stay there."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@cache
def load_yaml(name: str) -> Any:
    """`configs/<name>` parsed, read once per process."""

    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))
