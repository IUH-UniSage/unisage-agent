"""Gemini adapter (`JUDGE_PROVIDER=gemini`, the default): `google-genai` with structured output,
keys `GEMINI_API_KEYS`. The SDK is in the judge environment only (`evals/judge/requirements.txt`),
so it is imported when a call is made, not when the adapter is picked."""

from collections.abc import Callable, Mapping
from typing import Any

from evals.judge.providers.base import (
    Completion,
    Embedded,
    JudgeAdapter,
    MissingSetting,
    Models,
    Options,
    split_keys,
)
from evals.judge.providers.pool import KeyPool, PooledCaller

KEYS_ENV = "GEMINI_API_KEYS"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_FALLBACK = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING = "gemini-embedding-001"


def _thinking(model: str, level: str) -> Any:
    from google.genai import types

    if not level:
        return None
    if "gemini-3" in model:
        return types.ThinkingConfig(thinking_level=level)
    # 2.x models take a token budget; 0 turns thinking off where the model allows it
    return types.ThinkingConfig(thinking_budget=0)


def _client_cache() -> Callable[[str], Any]:
    clients: dict[str, Any] = {}

    def client_for(key: str) -> Any:
        if key not in clients:
            from google import genai

            clients[key] = genai.Client(api_key=key)
        return clients[key]

    return client_for


class GeminiAdapter(JudgeAdapter):
    name = "gemini"

    def __init__(self, keys: list[str], options: Options) -> None:
        self.keys = keys
        self.options = options
        self.caller = PooledCaller(
            pools={},
            client_for=_client_cache(),
            timeout_s=options.timeout_s,
            rpd_recheck_s=options.rpd_recheck_s,
        )

    @classmethod
    def default_models(cls, env: Mapping[str, str]) -> Models:
        return Models(DEFAULT_MODEL, DEFAULT_FALLBACK, DEFAULT_EMBEDDING)

    @classmethod
    def from_env(cls, env: Mapping[str, str], options: Options) -> "GeminiAdapter":
        keys = split_keys(env.get(KEYS_ENV))
        if not keys:
            raise MissingSetting(cls.name, [KEYS_ENV])
        return cls(keys, options)

    def _pool(self, model: str) -> None:
        # Gemini quotas are per project and per model: one pool (RPM window) per model
        if model not in self.caller.pools:
            self.caller.pools[model] = KeyPool(self.keys, rpm=self.options.rpm)

    async def generate_json(self, model: str, prompt: str, schema: dict[str, Any]) -> Completion:
        from google.genai import types

        self._pool(model)
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=_thinking(model, self.options.thinking),
        )
        response, key = await self.caller.call(
            model,
            lambda client: client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return Completion(
            text=response.text or "",
            model=model,
            key=key.masked,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            # thinking tokens are billed as output
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
            + int(getattr(usage, "thoughts_token_count", 0) or 0),
        )

    async def embed(self, model: str, texts: list[str]) -> Embedded:
        self._pool(model)
        response, key = await self.caller.call(
            model, lambda client: client.aio.models.embed_content(model=model, contents=texts)
        )
        usage = getattr(response, "usage_metadata", None)
        return Embedded(
            vectors=[list(embedding.values) for embedding in response.embeddings],
            model=model,
            key=key.masked,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        )

    def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return self.caller.stats()
