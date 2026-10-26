"""Start the cross-encoder service for a benchmark and make sure it is gone afterwards.

The harness never loads the model in its own process. When nothing answers at
`CROSS_ENCODER_URL` it spawns `uv run --project reranker uvicorn reranker.app:app`,
waits for `/health` to report the model loaded, and `stop()` sends SIGINT, waits for the
process to exit and then for nvidia-smi's used VRAM to come back to (about) the level
seen before the start - what "the benchmark does not keep VRAM" means. A service that
was already running is left alone and the report says so.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from app.integrations.cross_encoder_client import CrossEncoderClient, CrossEncoderUnavailableError
from evals.benchmark.resource_sampler import vram_used_mb

logger = logging.getLogger("evals.benchmark")

START_TIMEOUT_SECONDS = 600  # first start downloads the model
STOP_TIMEOUT_SECONDS = 30
VRAM_TOLERANCE_MB = 150


@dataclass
class ServiceHandle:
    process: subprocess.Popen[bytes] | None
    url: str
    spawned: bool
    vram_before_mb: int | None
    note: str = ""
    vram_after_stop_mb: int | None = None
    released: bool | None = None

    def stop(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        if process.poll() is None:
            logger.info("stopping the reranker service (pid %d)", process.pid)
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("reranker service did not exit on SIGINT - killing it")
                process.kill()
                process.wait(timeout=10)
        self.vram_after_stop_mb, self.released = _wait_for_vram(self.vram_before_mb)
        logger.info(
            "reranker service stopped; VRAM before=%s after=%s released=%s",
            self.vram_before_mb,
            self.vram_after_stop_mb,
            self.released,
        )


def _wait_for_vram(before_mb: int | None) -> tuple[int | None, bool | None]:
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    after = vram_used_mb()
    if before_mb is None or after is None:
        return after, None
    while (
        after is not None and after > before_mb + VRAM_TOLERANCE_MB and time.monotonic() < deadline
    ):
        time.sleep(1)
        after = vram_used_mb()
    return after, after is not None and after <= before_mb + VRAM_TOLERANCE_MB


async def ensure_service(
    client: CrossEncoderClient, *, url: str, serve: bool, port: int | None = None
) -> ServiceHandle:
    """A handle on a service answering at `url`: the one already there, or one spawned
    here (when `serve`). Raises `CrossEncoderUnavailableError` when neither is possible."""

    vram_before = vram_used_mb()
    try:
        health = await client.health()
        if health.get("loaded"):
            note = f"service already running at {url} ({health.get('device')}), left as is"
            logger.info(note)
            return ServiceHandle(None, url, spawned=False, vram_before_mb=vram_before, note=note)
    except CrossEncoderUnavailableError:
        pass
    if not serve:
        raise CrossEncoderUnavailableError(
            f"no reranker service at {url} and --no-serve given: run `task reranker:serve`"
        )
    port = port or urlparse(url).port or 8403
    command = [
        "uv",
        "run",
        "--project",
        "reranker",
        "uvicorn",
        "reranker.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = _spawn(command)
    handle = ServiceHandle(process, url, spawned=True, vram_before_mb=vram_before)
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CrossEncoderUnavailableError(
                f"reranker service exited with code {process.returncode} while starting"
            )
        try:
            health = await client.health()
            if health.get("loaded"):
                handle.note = (
                    f"service spawned by the benchmark on port {port} "
                    f"({health.get('device')}, vram_used_mb={health.get('vram_used_mb')})"
                )
                logger.info(handle.note)
                return handle
        except CrossEncoderUnavailableError:
            pass
        await asyncio.sleep(1)
    handle.stop()
    raise CrossEncoderUnavailableError(
        f"reranker service did not report loaded within {START_TIMEOUT_SECONDS}s"
    )


def _spawn(command: list[str]) -> subprocess.Popen[bytes]:
    logger.info("spawning the reranker service: %s", " ".join(command))
    return subprocess.Popen(
        command,
        stdout=sys.stderr,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
