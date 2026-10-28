"""OpenAI-compatible adapter (`JUDGE_PROVIDER=openai_compat`): any endpoint speaking the Chat
Completions API (Qwen through a proxy is the first one), over `httpx`.

Settings (environment or `.env.eval`, never committed):

    JUDGE_OPENAI_BASE_URL         e.g. https://api.example.com/qwen/v1
    JUDGE_OPENAI_API_KEYS         one or more keys, comma separated
    JUDGE_OPENAI_MODEL            judge model, e.g. Qwen3.8-Max-fast-agent
    JUDGE_OPENAI_FALLBACK_MODEL   optional, tried when the main model stays unavailable
    JUDGE_OPENAI_EMBEDDING_MODEL  optional; unset = no /embeddings (answer_relevancy then
                                  embeds through JUDGE_EMBEDDING_PROVIDER, Gemini by default)
    JUDGE_OPENAI_RESPONSE_FORMAT  json_object (default) | json_schema | none
    JUDGE_OPENAI_RPM              requests per minute per key (default: --rpm)

Some proxies ignore `response_format` and wrap the JSON in a ```json fence, so the schema is
also written into the system message and the answer is read with `extract_json`. Requests
carry their own User-Agent: some proxies (Cloudflare) refuse the default one of HTTP libraries.
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import httpx

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

BASE_URL_ENV = "JUDGE_OPENAI_BASE_URL"
KEYS_ENV = "JUDGE_OPENAI_API_KEYS"
MODEL_ENV = "JUDGE_OPENAI_MODEL"
FALLBACK_ENV = "JUDGE_OPENAI_FALLBACK_MODEL"
EMBEDDING_ENV = "JUDGE_OPENAI_EMBEDDING_MODEL"
FORMAT_ENV = "JUDGE_OPENAI_RESPONSE_FORMAT"
RPM_ENV = "JUDGE_OPENAI_RPM"
FORMATS = ("json_object", "json_schema", "none")
USER_AGENT = "unisage-eval-judge/1.0 (python-httpx)"
SYSTEM_PROMPT = (
    "You are a strict evaluation judge. Reply with ONE JSON object that validates against this "
    "JSON Schema, and nothing else (no markdown, no explanation):\n{schema}"
)
POOL = "all"  # one RPM window per key, whatever the model


class HTTPError(Exception):
    """A non-2xx answer, shaped for `pool.classify` (`.code`, `.details`)."""

    def __init__(self, code: int, details: Any) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.details = details


class _Client:
    """One key's view of the endpoint."""

    def __init__(self, http: httpx.AsyncClient, key: str) -> None:
        self.http = http
        self.key = key

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.http.post(
                path, json=body, headers={"Authorization": f"Bearer {self.key}"}
            )
        except httpx.TimeoutException as error:
            raise HTTPError(504, {"error": type(error).__name__}) from None
        except httpx.TransportError as error:
            raise HTTPError(503, {"error": type(error).__name__}) from None
        if response.status_code >= 400:
            raise HTTPError(response.status_code, _details(response))
        return response.json()


def _details(response: httpx.Response) -> dict[str, Any]:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:300]
    details: dict[str, Any] = {"body": body}
    retry_after = response.headers.get("retry-after", "")
    if retry_after.replace(".", "", 1).isdigit():
        details["retryDelay"] = f"{float(retry_after):g}s"  # read by pool.classify
    return details


class OpenAICompatAdapter(JudgeAdapter):
    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        keys: list[str],
        options: Options,
        *,
        response_format: str = "json_object",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if response_format not in FORMATS:
            raise ValueError(f"{FORMAT_ENV} must be one of {', '.join(FORMATS)}")
        self.response_format = response_format
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
            # longer than the pool's own deadline, so a slow call is cut by the pool
            timeout=options.timeout_s + 30,
            transport=transport,
        )
        clients = {key: _Client(self.http, key) for key in keys}
        self.caller = PooledCaller(
            pools={POOL: KeyPool(keys, rpm=options.rpm)},
            client_for=clients.__getitem__,
            timeout_s=options.timeout_s,
            rpd_recheck_s=options.rpd_recheck_s,
        )
        self.models_used: set[str] = set()

    @classmethod
    def default_models(cls, env: Mapping[str, str]) -> Models:
        return Models(
            judge=env.get(MODEL_ENV, "").strip(),
            fallback=env.get(FALLBACK_ENV, "").strip() or None,
            embedding=env.get(EMBEDDING_ENV, "").strip() or None,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str], options: Options) -> "OpenAICompatAdapter":
        base_url = env.get(BASE_URL_ENV, "").strip()
        keys = split_keys(env.get(KEYS_ENV))
        missing = [
            name for name, value in ((BASE_URL_ENV, base_url), (KEYS_ENV, keys)) if not value
        ]
        if missing:
            raise MissingSetting(cls.name, missing)
        response_format = env.get(FORMAT_ENV, "").strip() or "json_object"
        rpm = env.get(RPM_ENV, "").strip()
        if rpm:
            if not rpm.isdigit() or int(rpm) < 1:
                raise ValueError(f"{RPM_ENV} must be a positive integer")
            options = replace(options, rpm=int(rpm))
        return cls(base_url, keys, options, response_format=response_format)

    def _format(self, schema: dict[str, Any]) -> dict[str, Any] | None:
        if self.response_format == "json_schema":
            name = str(schema.get("title") or "output")[:64]
            return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
        if self.response_format == "json_object":
            return {"type": "json_object"}
        return None

    async def _call(self, model: str, path: str, body: dict[str, Any]) -> tuple[dict, str]:
        self.models_used.add(model)
        # the pool is per key (not per model), so a fallback model shares the RPM window
        data, key = await self.caller.call(POOL, lambda client: client.post(path, body))
        return data, key.masked

    async def generate_json(self, model: str, prompt: str, schema: dict[str, Any]) -> Completion:
        body: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False)),
                },
                {"role": "user", "content": prompt},
            ],
        }
        response_format = self._format(schema)
        if response_format:
            body["response_format"] = response_format
        data, key = await self._call(model, "chat/completions", body)
        choices = data.get("choices") or [{}]
        usage = data.get("usage") or {}
        return Completion(
            text=str((choices[0].get("message") or {}).get("content") or ""),
            model=model,
            key=key,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def embed(self, model: str, texts: list[str]) -> Embedded:
        data, key = await self._call(model, "embeddings", {"model": model, "input": texts})
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        return Embedded(
            vectors=[list(item["embedding"]) for item in items],
            model=model,
            key=key,
            input_tokens=int((data.get("usage") or {}).get("prompt_tokens") or 0),
        )

    def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        keys = self.caller.stats()[POOL]
        return {", ".join(sorted(self.models_used)) or POOL: keys}

    async def aclose(self) -> None:
        await self.http.aclose()
