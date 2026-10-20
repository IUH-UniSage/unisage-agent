"""Look at, measure or clear the retrieval cache (docs/specs/SPEC-retrieval-cache.md).

Reads RETRIEVAL_CACHE_REDIS_URL of the env the task loads (`task db:retrieval-cache`,
`task db:retrieval-cache env=eval`). Subcommands:

    list                 every entry: layer, time left, documents, the key's parts
    show --key KEY       one entry's bundle (chunk ids and scores)
    stats [--days N]     hit / miss_first / miss_second / stale / bypass / error per day and
                         layer, hit rate, entry count and memory per prefix
    clear [--document ID] [--layer l1|l2]   delete entries (all, one document's, one layer's)
    clear --disable | --enable               kill switch: every lookup becomes a bypass

Entries are JSON of `app.schemas.retrieval_cache` models; the companion `q:` key holds the
`describe_key()` text, which is the only readable form of a hashed key.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, cast

import redis

from app.core.config import settings
from app.core.timezone import now_ict
from app.integrations.retrieval_cache import (
    DISABLED_KEY,
    KEY_PREFIX,
    OUTCOMES,
    Layer,
    description_key,
    document_key,
    stats_key,
)
from app.schemas.retrieval_cache import CachedRetrievalBundle, CachedTurnBundle

LAYERS: tuple[Layer, ...] = ("l1", "l2")


def _text(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _left(conn: redis.Redis, key: str) -> str:
    ttl = cast(int, conn.ttl(key))
    if ttl < 0:
        return "no expiry"
    return f"{ttl // 3600}h{ttl % 3600 // 60:02d}m"


def _entries(conn: redis.Redis, layer: Layer) -> list[str]:
    prefix = f"{KEY_PREFIX}{layer}:"
    return sorted(_text(key) for key in conn.scan_iter(f"{prefix}*", count=1000))


def _split(key: str) -> tuple[Layer, str]:
    rest = key.removeprefix(KEY_PREFIX)
    layer, _, digest = rest.partition(":")
    return cast(Layer, layer), digest


def cmd_list(conn: redis.Redis, args: argparse.Namespace) -> int:
    total = 0
    for layer in LAYERS:
        if args.layer and args.layer != layer:
            continue
        for key in _entries(conn, layer):
            total += 1
            _, digest = _split(key)
            raw = cast(bytes | None, conn.get(key))
            bundle = _parse(layer, raw)
            documents = ", ".join(bundle.document_ids()) if bundle else "(unreadable)"
            print(f"\n{key}")
            print(
                f"  layer={layer} left={_left(conn, key)} "
                f"chunks={len(bundle.merged) if bundle else '?'} docs={documents}"
            )
            print(
                "  "
                + (
                    _text(cast(bytes | None, conn.get(description_key(layer, digest))))
                    or "(no description)"
                )
            )
    print(f"\n{total} entr(y/ies).")
    return 0


def _parse(layer: Layer, raw: bytes | None) -> CachedRetrievalBundle | None:
    if not raw:
        return None
    try:
        model = CachedTurnBundle if layer == "l1" else CachedRetrievalBundle
        return model.model_validate_json(raw)
    except Exception:
        return None


def cmd_show(conn: redis.Redis, args: argparse.Namespace) -> int:
    key = args.key
    layer, digest = _split(key)
    raw = cast(bytes | None, conn.get(key))
    if raw is None:
        print(f"{key}: not found")
        return 1
    print(f"{key}  left={_left(conn, key)}")
    print(_text(cast(bytes | None, conn.get(description_key(layer, digest)))))
    bundle = _parse(layer, raw)
    if bundle is None:
        print("(entry unreadable - the graph ignores it)")
        print(_text(raw))
        return 1
    print(f"created_at={bundle.created_at.isoformat()} questions={bundle.questions}")
    for index, refs in enumerate(bundle.per_query):
        print(f"  query {index} best={bundle.best_scores[index]:.3f}")
        for ref in refs:
            print(f"    {ref.score:.3f}  {ref.chunk_id}")
    print(f"  merged: {bundle.merged}")
    if isinstance(bundle, CachedTurnBundle):
        for task in bundle.tasks:
            print(
                f"  task intent={task.get('intent')} routing={task.get('routing_mode')} "
                f"query={task.get('query')!r}"
            )
        for sub_query in bundle.sub_queries:
            print(f"  sub_query[{sub_query.task_index}] {sub_query.question!r}")
    return 0


def cmd_stats(conn: redis.Redis, args: argparse.Namespace) -> int:
    today = now_ict().date()
    days = [today - timedelta(days=offset) for offset in range(args.days)]
    print(
        f"{'day':<12}{'layer':<6}"
        + "".join(f"{outcome:>12}" for outcome in OUTCOMES)
        + f"{'hit rate':>10}"
    )
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for day in days:
        for layer in LAYERS:
            keys = [stats_key(day, layer, outcome) for outcome in OUTCOMES]
            values = [int(value or 0) for value in cast(list[bytes | None], conn.mget(keys))]
            if not any(values):
                continue
            counts: dict[str, int] = dict(zip(OUTCOMES, values, strict=True))
            for outcome, value in counts.items():
                totals[layer][outcome] += value
            print(
                f"{day.isoformat():<12}{layer:<6}"
                + "".join(f"{counts[o]:>12}" for o in OUTCOMES)
                + f"{_rate(counts):>10}"
            )
    for layer in LAYERS:
        counts = totals[layer]
        if counts:
            print(
                f"{'total':<12}{layer:<6}"
                + "".join(f"{counts[o]:>12}" for o in OUTCOMES)
                + f"{_rate(counts):>10}"
            )
    print()
    for layer in LAYERS:
        keys = _entries(conn, layer)
        memory = 0
        for key in keys[:200]:
            memory += int(cast(int | None, conn.memory_usage(key)) or 0)
        estimate = memory * len(keys) / len(keys[:200]) if keys else 0
        print(f"{layer}: {len(keys)} entr(y/ies), ~{estimate / 1024:.0f} KB (sampled)")
    seen = sum(1 for _ in conn.scan_iter(f"{KEY_PREFIX}seen:*", count=1000))
    docs = sum(1 for _ in conn.scan_iter(f"{KEY_PREFIX}doc:*", count=1000))
    print(
        f"seen markers: {seen}, document indexes: {docs}, "
        f"kill switch: {'ON' if conn.exists(DISABLED_KEY) else 'off'}"
    )
    return 0


def _rate(counts: Mapping[str, int]) -> str:
    lookups = counts["hit"] + counts["miss_first"] + counts["miss_second"] + counts["stale"]
    return f"{counts['hit'] / lookups:.1%}" if lookups else "-"


def cmd_clear(conn: redis.Redis, args: argparse.Namespace) -> int:
    if args.disable:
        conn.set(DISABLED_KEY, "1")
        print("Kill switch ON: every lookup is a bypass until `clear --enable`.")
        return 0
    if args.enable:
        conn.delete(DISABLED_KEY)
        print("Kill switch off.")
        return 0
    if args.document:
        members = cast(set[bytes], conn.smembers(document_key(args.document)))
        keys = [_text(member) for member in members]
        deleted = _delete_entries(conn, keys)
        conn.delete(document_key(args.document))
        print(f"Deleted {deleted} entr(y/ies) referencing document {args.document}.")
        return 0
    layers = [args.layer] if args.layer else list(LAYERS)
    keys = [key for layer in layers for key in _entries(conn, layer)]
    deleted = _delete_entries(conn, keys)
    if not args.layer:
        extra = [_text(k) for k in conn.scan_iter(f"{KEY_PREFIX}doc:*", count=1000)]
        extra += [_text(k) for k in conn.scan_iter(f"{KEY_PREFIX}seen:*", count=1000)]
        if extra:
            conn.delete(*extra)
    print(f"Deleted {deleted} entr(y/ies) from {', '.join(layers)}.")
    return 0


def _delete_entries(conn: redis.Redis, keys: list[str]) -> int:
    deleted = 0
    for key in keys:
        layer, digest = _split(key)
        deleted += int(cast(int, conn.delete(key)))
        conn.delete(description_key(layer, digest), f"{KEY_PREFIX}seen:{layer}:{digest}")
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Show, measure or clear the retrieval cache.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="every entry with its key parts")
    p_list.add_argument("--layer", choices=LAYERS)
    p_show = sub.add_parser("show", help="one entry's bundle")
    p_show.add_argument("--key", required=True)
    p_stats = sub.add_parser("stats", help="daily counters and hit rate")
    p_stats.add_argument("--days", type=int, default=14)
    p_clear = sub.add_parser("clear", help="delete entries or flip the kill switch")
    p_clear.add_argument("--document")
    p_clear.add_argument("--layer", choices=LAYERS)
    p_clear.add_argument("--disable", action="store_true")
    p_clear.add_argument("--enable", action="store_true")
    args = parser.parse_args()

    # Windows consoles default to a legacy code page that cannot print Vietnamese.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    url = settings.RETRIEVAL_CACHE_REDIS_URL.strip()
    if not url:
        print("RETRIEVAL_CACHE_REDIS_URL is empty - the retrieval cache is off.")
        return 1
    conn = redis.Redis.from_url(url)
    handlers: dict[str, Any] = {
        "list": cmd_list,
        "show": cmd_show,
        "stats": cmd_stats,
        "clear": cmd_clear,
    }
    return int(handlers[args.command](conn, args))


if __name__ == "__main__":
    raise SystemExit(main())
