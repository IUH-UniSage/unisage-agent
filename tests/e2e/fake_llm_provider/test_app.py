"""Route-logic tests for the fake LLM provider - runnable with no docker/network.

Not marked `integration`: this only exercises the fake provider's own FastAPI
app in-process via `TestClient`, same as any other unit test in this repo. The
`integration` marker is reserved for tests that need the actual compose stack.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.e2e.fake_llm_provider.app import Mode, _state, app

client = TestClient(app)


def _reset() -> None:
    client.post("/admin/reset")


def test_ok_mode_returns_well_formed_completion() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "ok", "expected_api_key": "sk-test-key"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test-key"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert body["object"] == "chat.completion"


def test_wrong_key_is_401_even_in_ok_mode() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "ok", "expected_api_key": "sk-test-key"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-wrong"},
        json={"model": "fake-model", "messages": []},
    )

    assert resp.status_code == 401


def test_invalid_key_mode_rejects_every_key() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "invalid_key"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-fake-provider-default-key"},
        json={"model": "fake-model", "messages": []},
    )

    assert resp.status_code == 401


def test_rate_limited_mode_returns_429_with_retry_after() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "rate_limited", "retry_after_seconds": 7})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-fake-provider-default-key"},
        json={"model": "fake-model", "messages": []},
    )

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "7"


def test_out_of_credit_mode_returns_402() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "out_of_credit"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-fake-provider-default-key"},
        json={"model": "fake-model", "messages": []},
    )

    assert resp.status_code == 402


def test_echo_secrets_mode_returns_raw_authorization_and_body() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "echo_secrets"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-canary-deadbeef"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "marker-xyz"}]},
    )

    assert resp.status_code == 500
    message = resp.json()["error"]["message"]
    assert "sk-canary-deadbeef" in message
    assert "marker-xyz" in message


def test_mid_stream_error_sends_n_chunks_then_drops_connection() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "mid_stream_error", "error_after_chunks": 2})

    chunks: list[str] = []
    raised = False
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-fake-provider-default-key"},
            json={"model": "fake-model", "messages": [], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
    except Exception:
        # TestClient runs the ASGI app in-process, so the generator's exception
        # propagates directly here; over a real network it would instead surface to
        # the httpx client as a dropped/reset connection (RemoteProtocolError etc).
        raised = True

    # TestClient's in-process ASGI transport can drop already-buffered chunks
    # when the producer task is cancelled by the exception, so this only
    # asserts the drop itself (no more than `error_after_chunks` chunks, and it
    # never reaches the `[DONE]` sentinel) - the "exactly N clean chunks first"
    # half of this behavior is what the real docker-network integration test
    # (test_model_registry_smoke.py's peers) exercises against a real socket.
    assert len(chunks) <= 2
    assert "data: [DONE]" not in chunks
    assert raised, "mid_stream_error mode should tear the connection down, not finish cleanly"


def test_ok_mode_stream_completes_normally_with_done_sentinel() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "ok", "expected_api_key": "sk-test-key"})

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test-key"},
        json={"model": "fake-model", "messages": [], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    assert lines[-1] == "data: [DONE]"


def test_worker_pid_probe_header_is_echoed_back() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "ok", "expected_api_key": "sk-test-key"})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test-key", "X-Worker-Pid-Probe": "worker-42"},
        json={"model": "fake-model", "messages": []},
    )

    assert resp.headers["X-Worker-Pid"] == "worker-42"


def test_embeddings_endpoint_returns_one_vector_per_input() -> None:
    _reset()
    client.post("/admin/mode", json={"mode": "ok", "expected_api_key": "sk-test-key"})

    resp = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer sk-test-key"},
        json={"model": "fake-embedding-model", "input": ["a", "b", "c"]},
    )

    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 3


def test_reset_restores_default_mode_and_key() -> None:
    client.post("/admin/mode", json={"mode": "out_of_credit"})
    client.post("/admin/reset")

    assert _state.mode == Mode.OK
    assert _state.expected_api_key == "sk-fake-provider-default-key"
