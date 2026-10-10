"""A small OpenAI-compatible fake LLM provider for the integration harness.

Exists so the integration stack (docker-compose.integration.yml) never calls a
real LLM vendor. It is admin-controllable at runtime (`POST /admin/mode`) so
one running instance can play every failure mode the registry/failover code
needs to see: healthy, invalid key, rate limited, out of credit, an error
partway through a stream, and (for the secret-redaction canary test, plan.md
"Secret redaction") a mode that echoes back the raw `Authorization` header and
request body in an error response.

Not a real LLM: `/v1/chat/completions` and `/v1/embeddings` return canned,
deterministic content shaped like the real OpenAI response schema - just
enough for callers that only check "did a well-formed response come back" and
"did failover/circuit-breaker logic react correctly to this error".
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="fake-llm-provider")


class Mode(StrEnum):
    OK = "ok"
    INVALID_KEY = "invalid_key"
    RATE_LIMITED = "rate_limited"
    OUT_OF_CREDIT = "out_of_credit"
    MID_STREAM_ERROR = "mid_stream_error"
    ECHO_SECRETS = "echo_secrets"


class AdminState(BaseModel):
    mode: Mode = Mode.OK
    # The only key this instance accepts when mode == OK (or any mode that
    # still checks the key first, i.e. everything except INVALID_KEY, which
    # rejects every key on purpose).
    expected_api_key: str = "sk-fake-provider-default-key"
    retry_after_seconds: int = 2
    # mid_stream_error: how many SSE chunks go out normally before the
    # connection is torn down mid-response.
    error_after_chunks: int = 2
    request_count: int = 0


class SetModeRequest(BaseModel):
    mode: Mode
    expected_api_key: str | None = None
    retry_after_seconds: int | None = None
    error_after_chunks: int | None = None


_state = AdminState()


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization[len("bearer ") :].strip()


# --- Admin API - controls this instance's behavior for the next request(s).
# Never mounted on anything but the harness network; not part of the
# OpenAI-compatible surface. ---


@app.post("/admin/mode")
def set_mode(req: SetModeRequest) -> dict[str, Any]:
    _state.mode = req.mode
    if req.expected_api_key is not None:
        _state.expected_api_key = req.expected_api_key
    if req.retry_after_seconds is not None:
        _state.retry_after_seconds = req.retry_after_seconds
    if req.error_after_chunks is not None:
        _state.error_after_chunks = req.error_after_chunks
    return {"ok": True, "state": _state.model_dump()}


@app.get("/admin/state")
def get_state() -> dict[str, Any]:
    return _state.model_dump()


@app.post("/admin/reset")
def reset_state() -> dict[str, Any]:
    global _state
    _state = AdminState()
    return {"ok": True, "state": _state.model_dump()}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok"}


# --- OpenAI-compatible surface ---


def _rate_limited_response() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(_state.retry_after_seconds)},
        content={
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limited",
            }
        },
    )


def _out_of_credit_response() -> JSONResponse:
    return JSONResponse(
        status_code=402,
        content={
            "error": {
                "message": "Insufficient credit balance",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        },
    )


def _invalid_key_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
    )


async def _echo_secrets_response(request: Request, authorization: str | None) -> JSONResponse:
    # Deliberately puts the raw Authorization header + raw request body into the
    # error message - the canary test (plan.md "Secret redaction") seeds a
    # `sk-canary-<uuid>` key, drives a call through this mode, then asserts that
    # string never reaches DB/logs/Redis/Slack unredacted. This route is the one
    # place in the harness allowed to produce that raw text.
    raw_body = (await request.body()).decode("utf-8", errors="replace")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"echo_secrets: Authorization={authorization!r} body={raw_body!r}",
                "type": "internal_error",
                "code": "echo_secrets",
            }
        },
    )


def _chat_completion_payload(model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-fake-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "fake-llm-provider: canned response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
    }


def _sse_chunk(model: str, content: str, finish_reason: str | None) -> str:
    payload = {
        "id": f"chatcmpl-fake-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_response(model: str, worker_pid_header: dict[str, str]) -> StreamingResponse:
    error_after = _state.error_after_chunks

    async def _gen() -> Any:
        for i in range(error_after):
            yield _sse_chunk(model, f"chunk-{i} ", None)
        if _state.mode == Mode.MID_STREAM_ERROR:
            # No clean SSE error event - a mid-stream provider failure is an
            # abrupt connection drop, not a well-formed payload. Raising here
            # tears the response down; the client sees a truncated stream.
            raise RuntimeError("fake-llm-provider: simulated mid-stream failure")
        yield _sse_chunk(model, "final chunk", "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=worker_pid_header)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_worker_pid_probe: str | None = Header(default=None, alias="X-Worker-Pid-Probe"),
) -> Any:
    _state.request_count += 1
    body = await request.json()

    # Fixed, test-profile-only response header so the harness can assert "both
    # gunicorn workers actually served requests" (plan.md Task 0.5 acceptance:
    # "Mỗi request báo được worker nào xử lý"). The *agent's* outbound client
    # sets this from its own worker PID; this fake provider just echoes it back
    # unchanged so the test can read it off the final HTTP response too.
    worker_pid_header = {"X-Worker-Pid": x_worker_pid_probe} if x_worker_pid_probe else {}

    if _state.mode == Mode.INVALID_KEY:
        return _invalid_key_response()

    api_key = _extract_bearer(authorization)
    if api_key != _state.expected_api_key and _state.mode != Mode.ECHO_SECRETS:
        return _invalid_key_response()

    if _state.mode == Mode.RATE_LIMITED:
        return _rate_limited_response()
    if _state.mode == Mode.OUT_OF_CREDIT:
        return _out_of_credit_response()
    if _state.mode == Mode.ECHO_SECRETS:
        return await _echo_secrets_response(request, authorization)

    model = body.get("model", "fake-model")
    if body.get("stream"):
        return await _stream_response(model, worker_pid_header)

    return JSONResponse(content=_chat_completion_payload(model), headers=worker_pid_header)


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str | None = Header(default=None)) -> Any:
    _state.request_count += 1
    body = await request.json()

    if _state.mode == Mode.INVALID_KEY:
        return _invalid_key_response()
    api_key = _extract_bearer(authorization)
    if api_key != _state.expected_api_key and _state.mode != Mode.ECHO_SECRETS:
        return _invalid_key_response()
    if _state.mode == Mode.RATE_LIMITED:
        return _rate_limited_response()
    if _state.mode == Mode.OUT_OF_CREDIT:
        return _out_of_credit_response()
    if _state.mode == Mode.ECHO_SECRETS:
        return await _echo_secrets_response(request, authorization)

    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [
        {"object": "embedding", "index": i, "embedding": [0.001 * (i + 1)] * 8}
        for i in range(len(inputs))
    ]
    return JSONResponse(
        content={
            "object": "list",
            "data": data,
            "model": body.get("model", "fake-embedding-model"),
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )
