"""Background thread sampling CPU %, RAM and VRAM while a -p / -c loop runs.

The cross-encoder runs in another process, so CPU and RAM are machine-wide (psutil) and
VRAM is the GPU's used memory as nvidia-smi reports it. Every probe is wrapped: no GPU,
no nvidia-smi or no psutil means that column is None, never a crash.

    with ResourceSampler() as sampler:
        ...
    sampler.stats
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

SAMPLE_INTERVAL_SECONDS = 0.25


@dataclass
class ResourceStats:
    avg_cpu_pct: float | None = None
    peak_ram_gb: float | None = None
    peak_vram_mb: int | None = None
    avg_gpu_util_pct: float | None = None
    samples: int = 0


def gpu_snapshot() -> tuple[int | None, float | None]:
    """(used VRAM in MiB, GPU utilisation %) of GPU 0, or (None, None)."""

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        used, util = (part.strip() for part in out.split(",")[:2])
        return int(float(used)), float(util)
    except Exception:
        return None, None


def vram_used_mb() -> int | None:
    return gpu_snapshot()[0]


def gpu_name() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


class ResourceSampler:
    def __init__(self, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu: list[float] = []
        self._ram: list[float] = []
        self._vram: list[int] = []
        self._util: list[float] = []
        self.stats = ResourceStats()

    def __enter__(self) -> ResourceSampler:
        try:
            import psutil  # type: ignore[import-untyped]

            psutil.cpu_percent()  # first call always 0: prime it
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.stats = ResourceStats(
            avg_cpu_pct=round(sum(self._cpu) / len(self._cpu), 1) if self._cpu else None,
            peak_ram_gb=round(max(self._ram), 2) if self._ram else None,
            peak_vram_mb=max(self._vram) if self._vram else None,
            avg_gpu_util_pct=round(sum(self._util) / len(self._util), 1) if self._util else None,
            samples=max(len(self._cpu), len(self._vram)),
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                import psutil

                self._cpu.append(psutil.cpu_percent())
                self._ram.append(psutil.virtual_memory().used / (1024**3))
            except Exception:
                pass
            used, util = gpu_snapshot()
            if used is not None:
                self._vram.append(used)
            if util is not None:
                self._util.append(util)
            time.sleep(self._interval)
