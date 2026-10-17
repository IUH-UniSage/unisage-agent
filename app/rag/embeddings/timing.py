"""Phase timing of one self-contained embedding call (`embed()` / `embed_async()`), logged
once per call - what to read when chat retrieval is slow: the provider call itself, or the
budget reservation and usage recording around it (Redis round trips)."""

import logging
import time

logger = logging.getLogger(__name__)


class EmbedTiming:
    def __init__(self, text_count: int) -> None:
        self._text_count = text_count
        self._started_at = time.monotonic()
        self._last_mark = self._started_at
        self._phases: list[tuple[str, int]] = []

    def mark(self, phase: str) -> None:
        """Closes `phase`: the time since the previous mark (or since construction)."""

        now = time.monotonic()
        self._phases.append((phase, int((now - self._last_mark) * 1000)))
        self._last_mark = now

    def log(self, status: str) -> None:
        phases = " ".join(f"{phase}={elapsed_ms}" for phase, elapsed_ms in self._phases)
        logger.info(
            "Embedding timing: texts=%d status=%s %s total_ms=%d",
            self._text_count,
            status,
            phases,
            int((time.monotonic() - self._started_at) * 1000),
        )
