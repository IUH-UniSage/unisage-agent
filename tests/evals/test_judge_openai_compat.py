"""The OpenAI-compatible judge adapter and provider selection, against a fake HTTP endpoint
(`httpx.MockTransport`) - no network, no RAGAS."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.judge import cli, providers
from evals.judge.providers.base import MissingSetting, Options, extract_json
from evals.judge.providers.openai_compat import USER_AGENT, OpenAICompatAdapter
from evals.judge.providers.pool import CallFailed

KEY_A = "sk-test-aaaaaaaaaaaa1111"
KEY_B = "sk-test-bbbbbbbbbbbb2222"
SCHEMA = {"title": "Verdict", "type": "object", "properties": {"verdict": {"type": "integer"}}}


def _answer(content: str, prompt_tokens: int = 30, completion_tokens: int = 7) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def _adapter(handler: Any, keys: list[str] | None = None, **kwargs: Any) -> OpenAICompatAdapter:
    adapter = OpenAICompatAdapter(
        "https://proxy.test/qwen/v1",
        keys or [KEY_A],
        Options(rpm=100, timeout_s=5),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )

    # a fake clock: backoff and RPM waits pass at once
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    pool = adapter.caller.pools["all"]
    pool._clock = pool._wall = lambda: now[0]
    pool._sleep = sleep
    return adapter


def _run(adapter: OpenAICompatAdapter, coro: Any) -> Any:
    async def go() -> Any:
        try:
            return await coro
        finally:
            await adapter.aclose()

    return asyncio.run(go())


def test_a_fenced_json_answer_is_read_and_the_request_carries_schema_and_user_agent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _answer('```json\n{"verdict": 1}\n```')

    adapter = _adapter(handler)
    answer = _run(adapter, adapter.generate_json("Qwen-fast", "Đúng không?", SCHEMA))

    assert json.loads(extract_json(answer.text)) == {"verdict": 1}
    assert (answer.input_tokens, answer.output_tokens, answer.model) == (30, 7, "Qwen-fast")
    assert KEY_A not in answer.key  # masked
    request = seen[0]
    assert request.url.path == "/qwen/v1/chat/completions"
    assert request.headers["user-agent"] == USER_AGENT
    assert request.headers["authorization"] == f"Bearer {KEY_A}"
    body = json.loads(request.content)
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert '"verdict"' in body["messages"][0]["content"]  # schema in the system message
    assert body["messages"][1] == {"role": "user", "content": "Đúng không?"}


def test_json_schema_and_no_response_format_are_selectable() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _answer('{"verdict": 0}')

    strict = _adapter(handler, response_format="json_schema")
    _run(strict, strict.generate_json("m", "p", SCHEMA))
    plain = _adapter(handler, response_format="none")
    _run(plain, plain.generate_json("m", "p", SCHEMA))

    assert bodies[0]["response_format"]["json_schema"]["name"] == "Verdict"
    assert "response_format" not in bodies[1]


def test_a_429_with_retry_after_moves_on_to_another_key() -> None:
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"])
        if len(used) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"})
        return _answer('{"verdict": 1}')

    adapter = _adapter(handler, keys=[KEY_A, KEY_B])
    _run(adapter, adapter.generate_json("m", "p", SCHEMA))

    assert used == [f"Bearer {KEY_A}", f"Bearer {KEY_B}"]
    pool = adapter.caller.pools["all"]
    assert pool.keys[0].stats["rpm"] == 1 and pool.keys[0].cooldown_until > 0


def test_a_rejected_key_is_dropped_and_no_key_left_fails_without_leaking_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    adapter = _adapter(handler)
    with pytest.raises(CallFailed) as failed:
        _run(adapter, adapter.generate_json("m", "p", SCHEMA))

    assert adapter.caller.pools["all"].keys[0].denied
    assert KEY_A not in str(failed.value)


def test_server_errors_and_timeouts_are_retried_then_succeed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        if calls["n"] == 2:
            raise httpx.ReadTimeout("slow", request=request)
        return _answer('{"verdict": 1}')

    adapter = _adapter(handler)
    answer = _run(adapter, adapter.generate_json("m", "p", SCHEMA))

    assert calls["n"] == 3 and json.loads(answer.text) == {"verdict": 1}
    stats = adapter.stats()
    assert list(stats) == ["m"]
    (counts,) = stats["m"].values()
    assert counts["unavailable"] == 1 and counts["timeout"] == 1 and counts["calls"] == 3


def test_embeddings_come_back_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4},
            },
        )

    adapter = _adapter(handler)
    result = _run(adapter, adapter.embed("emb", ["a", "b"]))

    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]] and result.input_tokens == 4


def test_extract_json_finds_the_object_inside_words() -> None:
    assert extract_json('Kết quả: {"verdict": 1} hết.') == '{"verdict": 1}'
    assert extract_json('{"a": [1, 2]}') == '{"a": [1, 2]}'


# --- picking the provider from the environment ------------------------------------------------


def _args(*extra: str) -> Any:
    return cli.parse_args(["--out", "runs/x", *extra])


QWEN_ENV = {
    "JUDGE_PROVIDER": "openai_compat",
    "JUDGE_OPENAI_BASE_URL": "https://proxy.test/qwen/v1",
    "JUDGE_OPENAI_API_KEYS": KEY_A,
    "JUDGE_OPENAI_MODEL": "Qwen-fast",
}


def test_the_provider_comes_from_the_environment_and_the_flag_overrides_it() -> None:
    plan = cli.resolve(_args(), QWEN_ENV)
    assert (plan["provider"], plan["model"], plan["fallback"]) == (
        "openai_compat",
        "Qwen-fast",
        None,
    )
    # no /embeddings model configured: answer_relevancy embeds through Gemini
    assert (plan["embedding_provider"], plan["embedding_model"]) == (
        "gemini",
        "gemini-embedding-001",
    )

    gemini = cli.resolve(_args("--judge-provider", "gemini"), QWEN_ENV)
    assert (gemini["provider"], gemini["model"]) == ("gemini", "gemini-3.1-flash-lite-preview")
    assert gemini["fallback"] == "gemini-3.5-flash-lite"

    assert cli.resolve(_args(), {})["provider"] == "gemini"  # nothing set: Gemini
    only_refusal = cli.resolve(_args("--metrics", "refusal"), QWEN_ENV)
    assert only_refusal["embedding_provider"] is None


def test_cli_values_win_and_an_empty_fallback_disables_it() -> None:
    plan = cli.resolve(
        _args(
            "--judge-model", "Qwen-max", "--judge-fallback", "", "--embedding-provider", "gemini"
        ),
        {**QWEN_ENV, "JUDGE_OPENAI_FALLBACK_MODEL": "Qwen-plus"},
    )
    assert (plan["model"], plan["fallback"]) == ("Qwen-max", None)


def test_missing_settings_are_named() -> None:
    with pytest.raises(MissingSetting) as missing:
        OpenAICompatAdapter.from_env({"JUDGE_PROVIDER": "openai_compat"}, Options())
    assert missing.value.names == ("JUDGE_OPENAI_BASE_URL", "JUDGE_OPENAI_API_KEYS")
    with pytest.raises(MissingSetting):
        cli.resolve(_args(), {"JUDGE_PROVIDER": "openai_compat"})  # no model
    with pytest.raises(ValueError):
        providers.judge_provider({"JUDGE_PROVIDER": "nope"})


def test_main_exits_2_naming_the_missing_keys(tmp_path: Path, capsys) -> None:
    row = {
        "id": "q1",
        "variant": "original",
        "asked": "Học phí?",
        "category": "unanswerable",
        "response": "Không tìm thấy.",
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    env = tmp_path / ".env.eval"
    env.write_text(
        "JUDGE_PROVIDER=openai_compat\nJUDGE_OPENAI_MODEL=Qwen-fast\n"
        "JUDGE_OPENAI_BASE_URL=https://proxy.test/v1\n",
        encoding="utf-8",
    )

    code = cli.main(["--out", str(tmp_path), "--env-file", str(env), "--metrics", "refusal"])

    assert code == 2
    assert "JUDGE_OPENAI_API_KEYS" in capsys.readouterr().err


def test_the_rpm_per_key_can_be_set_for_the_endpoint() -> None:
    adapter = OpenAICompatAdapter.from_env({**QWEN_ENV, "JUDGE_OPENAI_RPM": "60"}, Options(rpm=12))
    assert adapter.caller.pools["all"].rpm == 60
    asyncio.run(adapter.aclose())
    with pytest.raises(ValueError):
        OpenAICompatAdapter.from_env({**QWEN_ENV, "JUDGE_OPENAI_RPM": "0"}, Options())
