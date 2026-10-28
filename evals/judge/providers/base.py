"""The interface every judge adapter implements.

An adapter does two things for the RAGAS metrics: return the JSON a prompt asks for (fitting a
JSON schema), and embed texts. Retries, key rotation and rate limits stay inside the adapter
(through `pool.PooledCaller`); model fallback, cost booking and parsing into the metric's
pydantic model are done once, for every adapter, by `ragas_metrics.JudgeLLM`.

Adapters know nothing of RAGAS, so they are tested in the service environment.
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar


class MissingSetting(Exception):
    """An environment variable the adapter needs is not set."""

    def __init__(self, provider: str, names: Sequence[str]) -> None:
        super().__init__(
            f"judge provider {provider!r} needs {', '.join(names)} (environment or --env-file)"
        )
        self.names = tuple(names)


@dataclass(frozen=True)
class Options:
    """Run settings shared by every adapter (from the CLI)."""

    rpm: int = 12
    rpd_recheck_s: float = 0.0
    timeout_s: float = 90.0
    thinking: str = "minimal"  # Gemini only: lowest thinking level the model allows


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    key: str  # masked
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    model: str
    key: str  # masked
    input_tokens: int = 0


@dataclass(frozen=True)
class Models:
    """The models an adapter scores with when neither the CLI nor the environment names one."""

    judge: str
    fallback: str | None = None
    embedding: str | None = None  # None: the provider cannot embed (use another adapter)


class JudgeAdapter(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def default_models(cls, env: Mapping[str, str]) -> Models: ...

    @classmethod
    @abstractmethod
    def from_env(cls, env: Mapping[str, str], options: Options) -> "JudgeAdapter":
        """Build the adapter from its settings; `MissingSetting` names what is missing."""

    @abstractmethod
    async def generate_json(self, model: str, prompt: str, schema: dict[str, Any]) -> Completion:
        """The model's answer to `prompt`, meant to be JSON fitting `schema` (not validated
        here). Raises the `pool` errors: `ModelUnavailable`, `CallFailed`, `QuotaExhausted`."""

    async def embed(self, model: str, texts: list[str]) -> Embedded:
        raise NotImplementedError(f"judge provider {self.name!r} cannot embed")

    @abstractmethod
    def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        """{model: {masked key: {calls, rpm, timeout, ...}}}"""

    async def aclose(self) -> None:
        return None


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(text: str) -> str:
    """The JSON object in a model's answer: as is, without a ```json fence, or the outermost
    `{...}` when the model wrote words around it."""

    text = _FENCE.sub("", (text or "").strip())
    try:
        json.loads(text)
        return text
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        return text[start : end + 1] if 0 <= start < end else text


def split_keys(raw: str | None) -> list[str]:
    return [key for key in re.split(r"[\s,;]+", raw or "") if key]
