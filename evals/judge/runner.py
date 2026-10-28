"""The judge run: score every task with bounded concurrency, wait out a daily quota when asked,
and score again with the main model only the rows the fallback model scored too often."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.judge.checkpoint import fallback_rows, load_records, record_of
from evals.judge.providers.pool import QuotaExhausted
from evals.judge.tasks import NO_LLM, Scorer, Task


@dataclass
class JudgeRun:
    out: Path
    scorer_factory: Callable[[bool], Scorer]
    fallback_model: str | None = None
    concurrency: int = 8
    quota_wait_s: float = 0.0
    max_quota_waits: int = 48
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    clock: Callable[[], float] = time.time

    @property
    def checkpoint(self) -> Path:
        return self.out / "judge.jsonl"

    async def run(self, tasks: Sequence[Task]) -> bool:
        """Score every task; False when stopped by quota with tasks left (checkpoint kept)."""

        scorer = self.scorer_factory(False)
        if not await self._until_done(scorer, tasks):
            return False
        records = load_records(self.checkpoint)
        redo = fallback_rows(records, self.fallback_model)
        if redo:
            print(f"{len(redo)} row(s) needed the fallback model (> 5%): scoring them again")
            again = [task for task in tasks if task.key in redo and task.metric not in NO_LLM]
            primary = self.scorer_factory(True)
            return await self._until_done(primary, again)
        return True

    async def _until_done(self, scorer: Scorer, tasks: Sequence[Task]) -> bool:
        """Every task tried once (a failed metric is recorded as NaN and left for the next
        run); only the tasks a quota stop never reached are tried again after the wait."""

        waits = 0
        todo = list(tasks)
        while todo:
            reset_at, tried = await self._pass(scorer, todo)
            todo = [task for task in todo if (task.key, task.metric) not in tried]
            if reset_at is None or not todo:
                return True
            if self.quota_wait_s <= 0 or waits >= self.max_quota_waits:
                print(
                    f"quota exhausted: {len(todo)} score(s) left - rerun the same command "
                    "once the quota is back"
                )
                return False
            waits += 1
            wait = min(max(60.0, reset_at - self.clock() + 60.0), self.quota_wait_s)
            resume = time.strftime("%H:%M", time.localtime(self.clock() + wait))
            print(f"every key is out for the day: waiting until ~{resume} ({waits})")
            await self.sleep(wait)
        return True

    async def _pass(
        self, scorer: Scorer, tasks: Sequence[Task]
    ) -> tuple[float | None, set[tuple[str, str]]]:
        """One pass over `tasks`: the reset time if the quota ran out (else None) and the
        tasks that got a record."""

        gate = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        stopped: list[float] = []
        tried: set[tuple[str, str]] = set()
        done = 0

        async def one(task: Task) -> None:
            nonlocal done
            async with gate:
                if stopped:
                    return
                try:
                    score = await scorer.score(task)
                except QuotaExhausted as error:
                    stopped.append(error.reset_at)
                    return
                record = record_of(task, score)
                async with lock:
                    with self.checkpoint.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    tried.add((task.key, task.metric))
                    done += 1
                    shown = "NaN" if record["value"] is None else record["value"]
                    print(f"[{done}/{len(tasks)}] {task.key} {task.metric} = {shown}")

        await asyncio.gather(*(one(task) for task in tasks))
        return (min(stopped) if stopped else None), tried
