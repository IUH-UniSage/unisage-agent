import redis

from app.tools.replay_dead_letter import replay_dead_letter
from app.worker.usage_outbox_tasks import DEAD_KEY, OUTBOX_KEY


def _redis() -> redis.Redis:
    return redis.Redis.from_url("redis://localhost:6379/0")


def test_replay_dead_letter_moves_every_item_to_the_outbox() -> None:
    conn = _redis()
    conn.flushdb()
    conn.lpush(DEAD_KEY, "a", "b", "c")

    moved = replay_dead_letter()

    assert moved == 3
    assert conn.llen(DEAD_KEY) == 0
    assert conn.llen(OUTBOX_KEY) == 3
    conn.flushdb()
    conn.close()


def test_replay_dead_letter_is_a_no_op_on_an_empty_queue() -> None:
    conn = _redis()
    conn.flushdb()

    moved = replay_dead_letter()

    assert moved == 0
    conn.close()
