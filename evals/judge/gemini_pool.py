"""A pool of Gemini API keys for the judge (Task 11, `spec.md` §5).

Free-tier quotas apply per project and per model: each key gets its own client-side request
limit (≤ 12 RPM by default) and the keys are used round-robin. What happens on an error:

- 429 out of requests per minute: wait the `retryDelay` the error gives, on another key;
- 429 out of requests per day: the key sits out until the daily reset (midnight Pacific time),
  or only `rpd_recheck_s` when set - Google's free tier sometimes answers per-day 429 for a key
  that still has quota, so the key is tried again later;
- 503 / 500 / 504 / client timeout: the key cools down 2-32 s (backoff with jitter) and the call
  moves on to another key; after `max_unavailable` / `max_timeouts` tries `ModelUnavailable`
  (the caller can switch to the fallback model) or `CallFailed`;
- 401 / 403 (the key's project is blocked, "Your project has been denied access"): the key is
  dropped from the pool for the rest of the run and the call moves on to another key;
  `CallFailed` once no key is left;
- every key out for the day: `QuotaExhausted` with the earliest reset time.

Nothing here imports `google.genai`: errors are read by duck typing (`.code`, `.details`), so
the pool is tested with a fake client.
"""

import asyncio
import json
import random
import re
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

WINDOW_S = 60.0
_RETRY_DELAY = re.compile(r'"?retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s')
# Phrases Google puts in a per-day quota violation (`quotaId` like
# `GenerateRequestsPerDayPerProjectPerModel-FreeTier`).
_PER_DAY = ("perday", "per_day", "daily")


class QuotaExhausted(Exception):
    """Every key is out of requests for the day."""

    def __init__(self, reset_at: float) -> None:
        super().__init__(f"all keys exhausted until {time.ctime(reset_at)}")
        self.reset_at = reset_at


class ModelUnavailable(Exception):
    """The model stayed overloaded (503/5xx) through every retry."""


class CallFailed(Exception):
    """The call kept timing out, or failed in a way a retry cannot fix."""


def mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 10 else "…"


def next_daily_reset(now: float | None = None) -> float:
    """Epoch seconds of the next midnight in Pacific time, when Gemini's daily quota resets."""

    moment = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        pacific = moment.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:  # no tz database: assume daylight time, the common case for a run
        pacific = moment.astimezone(timezone(timedelta(hours=-7)))
    midnight = (pacific + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


@dataclass
class KeyState:
    key: str
    recent: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0  # monotonic
    exhausted_until: float = 0.0  # wall clock
    denied: bool = False  # 401/403: the key's project cannot call the model
    stats: Counter[str] = field(default_factory=Counter)

    @property
    def masked(self) -> str:
        return mask(self.key)


class KeyPool:
    """Keys of one model, each held to `rpm` calls in any 60 s window."""

    def __init__(
        self,
        keys: list[str],
        *,
        rpm: int = 12,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if not keys:
            raise ValueError("no API keys: set GEMINI_API_KEYS")
        self.keys = [KeyState(key) for key in dict.fromkeys(keys)]
        self.rpm = rpm
        self._clock, self._wall, self._sleep = clock, wall, sleep
        self._next = 0

    def _ready_at(self, state: KeyState, now: float) -> float:
        while state.recent and state.recent[0] <= now - WINDOW_S:
            state.recent.popleft()
        slot = state.recent[0] + WINDOW_S if len(state.recent) >= self.rpm else now
        return max(slot, state.cooldown_until)

    async def acquire(self) -> KeyState:
        """The next key allowed to call, waiting for a free slot if none is free now."""

        while True:
            now, wall = self._clock(), self._wall()
            usable = [state for state in self.keys if not state.denied]
            if not usable:
                raise CallFailed("every API key was denied (401/403)")
            live = [state for state in usable if state.exhausted_until <= wall]
            if not live:
                raise QuotaExhausted(min(state.exhausted_until for state in usable))
            start = self._next % len(self.keys)
            order = [state for state in self.keys[start:] + self.keys[:start] if state in live]
            ready = [(self._ready_at(state, now), state) for state in order]
            for at, state in ready:
                if at <= now:
                    state.recent.append(now)
                    state.stats["calls"] += 1
                    self._next = self.keys.index(state) + 1
                    return state
            await self._sleep(max(0.01, min(at for at, _state in ready) - now))

    def cool_down(self, state: KeyState, seconds: float) -> None:
        state.cooldown_until = max(state.cooldown_until, self._clock() + seconds)

    def exhaust(self, state: KeyState, until: float) -> None:
        state.exhausted_until = until

    def deny(self, state: KeyState) -> None:
        state.denied = True


def classify(error: BaseException) -> tuple[str, float | None]:
    """(`rpm` | `rpd` | `unavailable` | `timeout` | `denied` | `fatal`, retry delay in seconds)."""

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", None
    code = getattr(error, "code", None)
    text = _error_text(error)
    if code == 429:
        if any(hint in text.lower().replace(" ", "") for hint in _PER_DAY):
            return "rpd", None
        match = _RETRY_DELAY.search(text)
        return "rpm", float(match.group(1)) if match else None
    if code in (500, 502, 503):
        return "unavailable", None
    if code == 504:
        return "timeout", None
    if code in (401, 403):
        return "denied", None
    return "fatal", None


def _error_text(error: BaseException) -> str:
    details = getattr(error, "details", None)
    try:
        extra = json.dumps(details) if details is not None else ""
    except (TypeError, ValueError):
        extra = str(details)
    return f"{error} {extra}"


@dataclass
class GeminiCaller:
    """Runs one request against a model through its key pool, retrying as the module says."""

    pools: dict[str, KeyPool]
    client_for: Callable[[str], Any]
    timeout_s: float = 90.0
    max_unavailable: int = 30
    max_timeouts: int = 30
    rpd_recheck_s: float = 0.0  # 0 = a per-day 429 benches the key until the daily reset
    rng: random.Random = field(default_factory=random.Random)
    next_reset: Callable[[], float] = next_daily_reset
    wall: Callable[[], float] = time.time

    async def call(
        self, model: str, request: Callable[[Any], Awaitable[Any]]
    ) -> tuple[Any, KeyState]:
        pool = self.pools[model]
        unavailable = timeouts = 0
        while True:
            state = await pool.acquire()
            try:
                response = await asyncio.wait_for(
                    request(self.client_for(state.key)), self.timeout_s
                )
                return response, state
            except Exception as error:
                kind, delay = classify(error)
                if kind == "fatal":
                    raise CallFailed(f"{type(error).__name__}: {error}") from error
                state.stats[kind] += 1
                if kind == "rpm":
                    pool.cool_down(state, delay if delay is not None else WINDOW_S)
                elif kind == "rpd":
                    until = self.next_reset()
                    if self.rpd_recheck_s > 0:
                        until = min(until, self.wall() + self.rpd_recheck_s)
                    pool.exhaust(state, until)
                elif kind == "denied":
                    if not state.denied:
                        pool.deny(state)
                        print(
                            f"{model}: key {state.masked} denied ({_error_text(error)[:160]}),"
                            " dropped from the pool"
                        )
                elif kind == "unavailable":
                    unavailable += 1
                    if unavailable > self.max_unavailable:
                        raise ModelUnavailable(f"{model}: {error}") from error
                    pool.cool_down(state, self._backoff(unavailable))
                else:  # timeout
                    timeouts += 1
                    if timeouts > self.max_timeouts:
                        raise CallFailed(f"{model}: timed out {timeouts} times") from error
                    pool.cool_down(state, self._backoff(timeouts))

    def _backoff(self, attempt: int) -> float:
        base = min(32.0, 2.0**attempt)
        return base + self.rng.uniform(0, base / 2)

    def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            model: {state.masked: dict(state.stats) for state in pool.keys}
            for model, pool in self.pools.items()
        }
