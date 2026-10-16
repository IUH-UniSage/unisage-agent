"""Shows what the Redis web search cache holds (see `app.integrations.web_search_cache`):
every cached Tavily answer with its pages, score, academic year and time left.

    task db:web-cache env=eval                       # list every entry
    task db:web-cache env=eval content=1             # also print each page's snippet
    task db:web-cache env=eval query="học phí năm học 2025-2026"   # one query's entry
    task db:web-cache env=eval flush=1               # delete every entry

The key is a hash of the query, so a listed entry never says which query it came from.
`--query` finds one by recomputing its key, so it needs the exact query text that the agent
log's `Web search query=... cache=hit|miss` lines print. Reads WEB_SEARCH_CACHE_REDIS_URL of
the env the task loads (eval: DB 7, dev: DB 6).
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

import redis
from pydantic import ValidationError

from app.core.config import settings
from app.integrations.web_search_cache import _RESULTS, KEY_PREFIX, cache_key
from app.schemas.web_search import WebSearchResult


def _print_entry(conn: redis.Redis, key: bytes, *, content: bool) -> None:
    # The sync client's stubs type every reply as `Awaitable | Any`.
    raw = cast(bytes | None, conn.get(key))
    ttl = cast(int, conn.ttl(key))
    try:
        pages: list[WebSearchResult] = _RESULTS.validate_json(raw) if raw else []
    except ValidationError:
        print(f"\n{key.decode()}  (entry unreadable, ignored by the agent)")
        return
    left = f"{ttl // 3600}h{ttl % 3600 // 60:02d}m" if ttl >= 0 else "no expiry"
    print(f"\n{key.decode()[len(KEY_PREFIX) :][:12]}  {left} left  {len(pages)} page(s)")
    for page in pages:
        print(f"  {page.score:.2f}  {page.academic_year or '-':9}  {page.url}")
        print(f"        {page.title[:100]}")
        if content:
            snippet = " ".join(page.content.split())
            print(f"        > {snippet[:500]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show or clear the web search cache.")
    parser.add_argument("--content", action="store_true", help="print each page's snippet")
    parser.add_argument("--query", help="show only this query's entry")
    parser.add_argument(
        "--max-results",
        type=int,
        default=settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB,
        help="max_results the query was searched with (part of the key)",
    )
    parser.add_argument("--flush", action="store_true", help="delete every cached entry")
    args = parser.parse_args()

    # Windows consoles default to a legacy code page that cannot print Vietnamese.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    url = settings.WEB_SEARCH_CACHE_REDIS_URL.strip()
    if not url:
        print("WEB_SEARCH_CACHE_REDIS_URL is empty - the web search cache is off.")
        return 1
    conn = redis.Redis.from_url(url)
    try:
        if args.query:
            key = cache_key(args.query, max_results=args.max_results).encode()
            if not conn.exists(key):
                print(f"No cached entry for {args.query!r} (max_results={args.max_results}).")
                return 1
            _print_entry(conn, key, content=args.content)
            return 0

        keys = sorted(conn.scan_iter(f"{KEY_PREFIX}*", count=1000))
        if args.flush:
            deleted = conn.delete(*keys) if keys else 0
            print(f"Deleted {deleted} cached web search entr(y/ies) from {url}.")
            return 0
        print(f"{len(keys)} cached web search entr(y/ies) in {url}")
        for key in keys:
            _print_entry(conn, key, content=args.content)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
