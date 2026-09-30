"""Moves every item sitting in the usage outbox's dead-letter queue back onto
the main outbox, so the next drain run retries them.

Run by hand after fixing whatever made backend-java reject these payloads with
a 4xx (a schema mismatch, usually) - see `app.worker.usage_outbox_tasks` for
when an item lands in the dead-letter queue in the first place.

    python -m app.tools.replay_dead_letter

Prints how many items it moved. Safe to run with an empty dead-letter queue
(prints 0, does nothing) or while the drain worker is running concurrently -
`RPOPLPUSH` is atomic, so no item is ever duplicated or dropped mid-move.
"""

from __future__ import annotations

import redis

from app.core.config import settings
from app.worker.usage_outbox_tasks import DEAD_KEY, OUTBOX_KEY


def replay_dead_letter() -> int:
    conn = redis.Redis.from_url(settings.REDIS_URL)
    try:
        moved = 0
        while conn.rpoplpush(DEAD_KEY, OUTBOX_KEY) is not None:
            moved += 1
        return moved
    finally:
        conn.close()


def main() -> int:
    moved = replay_dead_letter()
    print(f"Replayed {moved} item(s) from {DEAD_KEY} back onto {OUTBOX_KEY}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
