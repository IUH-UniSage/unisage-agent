"""Readable titles for `manifest.csv` (Task 7a, spec §3.5).

The citation chip shows the uploaded file name (`citations.source_title` of the
object key) and the viewer/admin show `documents.title`, so every document is
uploaded as `<title>.pdf` with that same title. Crawled files are stored as
`<file_id>.pdf`, and many original names are codes (`CTDT.pdf`), tool output
(`ilovepdf_merged(4).pdf`) or lack diacritics - so titles are written from the
document itself:

1. `extract` writes, per selected row, page-1 text and its largest-font lines
   (`title_inputs.jsonl`) for a reader to name the document.
2. A reader (person or LLM) writes `{file_id, title, basis, confidence}` lines.
3. `apply` merges them into the manifest (`title`, `title_source = basis`),
   never touching `title_source=manual` rows, and lists rows to review.
4. `rename` renames each titled PDF on disk to `<title>.pdf` (same folder) and
   updates `local_path`; `file_id` stays the key everything else joins on.

Usage:
    python -m evals.titles extract --dataset ../unisage-gateway/dataset/official --out inputs.jsonl
    python -m evals.titles apply --dataset ../unisage-gateway/dataset/official titles.jsonl
    python -m evals.titles rename --dataset ../unisage-gateway/dataset/official
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf

from evals.crawl.download import DownloadState, read_csv, write_csv

EXCERPT_CHARS = 1500
HEADING_LINES = 6
MAX_TITLE_CHARS = 150
TITLE_SOURCES = {"heading", "content", "filename"}
REVIEW_FIELDS = ["file_id", "department_id", "file_name", "title", "title_source", "reason"]

_FORBIDDEN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_UUID_PREFIX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_"
)
_HEX_ID = re.compile(r"^[0-9a-f]{12}$")


def clean_title(raw: str) -> str:
    """NFC, one line, no uuid prefix, at most `MAX_TITLE_CHARS` (cut at a word)."""

    title = unicodedata.normalize("NFC", raw)
    title = _UUID_PREFIX.sub("", " ".join(title.split()))
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
    return title.strip(" .-_")


def upload_name(title: str) -> str:
    """File name to upload as: the title with characters no OS allows in a file
    name replaced (a decree number `1035/QĐ-ĐHCN` becomes `1035-QĐ-ĐHCN`)."""

    name = _FORBIDDEN.sub("-", clean_title(title))
    name = re.sub(r"-{2,}", "-", name).strip(" .-")
    return f"{name}.pdf"


def title_problem(title: str, file_id: str) -> str | None:
    """Why a title can't be used as is, or None."""

    if not title:
        return "empty"
    if title == file_id or _HEX_ID.match(title) or _UUID_PREFIX.match(title):
        return "looks like an id"
    if len(title) < 8:
        return "too short"
    return None


def page_inputs(pdf_path: Path) -> dict[str, Any]:
    """Page-1 text (page 2 too when page 1 is nearly empty) and the lines set in
    the largest fonts - usually the document's own title."""

    with pymupdf.open(pdf_path) as document:
        pages = [document[0]] if document.page_count else []
        if pages and len(pages[0].get_text().strip()) < 200 and document.page_count > 1:
            pages.append(document[1])
        text = "\n".join(page.get_text() for page in pages)
        sized: list[tuple[float, str]] = []
        for page in pages:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    if spans:
                        line_text = " ".join(span["text"].strip() for span in spans)
                        sized.append((max(span["size"] for span in spans), line_text))
    sized.sort(key=lambda item: -item[0])
    return {
        "text": " ".join(text.split())[:EXCERPT_CHARS],
        "headings": [line for _size, line in sized[:HEADING_LINES]],
    }


def extract(state: DownloadState, out: Path) -> int:
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in state.manifest:
            if row.get("selected") != "true" or row.get("title_source") == "manual":
                continue
            item = {
                "file_id": row["file_id"],
                "file_name": row["file_name"],
                "unit": row["unit"],
                "department_id": row["department_id"],
                "pages": row["pages"],
                **page_inputs(state.dataset / row["local_path"]),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def apply_titles(rows: list[dict[str, str]], titles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Set `title`/`title_source` from reader output; returns rows to review."""

    by_id = {row["file_id"]: row for row in rows}
    review: list[dict[str, str]] = []
    for item in titles:
        row = by_id.get(item["file_id"])
        if row is None or row.get("title_source") == "manual":
            continue
        title = clean_title(item["title"])
        basis = item.get("basis", "content")
        row["title"] = title
        row["title_source"] = basis if basis in TITLE_SOURCES else "content"
        problem = title_problem(title, row["file_id"])
        if problem or item.get("confidence") == "low":
            review.append({**row, "reason": problem or f"low confidence: {item.get('note', '')}"})

    seen: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("selected") == "true" and row.get("title"):
            seen[(row["department_id"], upload_name(row["title"]).casefold())].append(row)
    for duplicates in seen.values():
        if len(duplicates) > 1:
            review.extend({**row, "reason": "same title in department"} for row in duplicates)
    return review


def rename_files(dataset: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Move every titled row's PDF to `<folder>/<upload_name(title)>` and update
    `local_path`. Names are compared case-insensitively (Windows/macOS); a name
    already taken in the folder gets ` (<file_id>)` appended. Returns
    `{file_id, old_path, new_path}` for each file moved."""

    taken: dict[Path, set[str]] = defaultdict(set)
    for row in rows:
        path = Path(row["local_path"])
        taken[path.parent].add(path.name.casefold())

    moved: list[dict[str, str]] = []
    for row in rows:
        if not row.get("title"):
            continue
        old = Path(row["local_path"])
        name = upload_name(row["title"])
        if old.name == name:
            continue
        if name.casefold() in taken[old.parent]:
            name = f"{name.removesuffix('.pdf')} ({row['file_id']}).pdf"
            if old.name == name:
                continue
        new = old.with_name(name)
        (dataset / old).rename(dataset / new)
        taken[old.parent].discard(old.name.casefold())
        taken[old.parent].add(name.casefold())
        row["local_path"] = new.as_posix()
        moved.append(
            {"file_id": row["file_id"], "old_path": old.as_posix(), "new_path": new.as_posix()}
        )
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract_cmd = commands.add_parser("extract")
    extract_cmd.add_argument("--dataset", type=Path, required=True)
    extract_cmd.add_argument("--out", type=Path, required=True)
    apply_cmd = commands.add_parser("apply")
    apply_cmd.add_argument("--dataset", type=Path, required=True)
    apply_cmd.add_argument("titles", type=Path, nargs="+")
    rename_cmd = commands.add_parser("rename")
    rename_cmd.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    state = DownloadState(args.dataset)
    if args.command == "extract":
        print(f"wrote {extract(state, args.out)} rows to {args.out}")
        return
    if args.command == "rename":
        moved = rename_files(args.dataset, state.manifest)
        state.save()
        # Merge into the record of earlier runs (keyed by file_id), so a re-run or a later
        # batch never drops the old -> new paths already pushed elsewhere.
        record_path = args.dataset / "renamed_files.csv"
        record = {row["file_id"]: row for row in read_csv(record_path)}
        record.update({row["file_id"]: row for row in moved})
        write_csv(record_path, ["file_id", "old_path", "new_path"], list(record.values()))
        print(f"renamed {len(moved)} files -> {args.dataset / 'renamed_files.csv'}")
        return

    titles = [
        json.loads(line)
        for path in args.titles
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    review = apply_titles(state.manifest, titles)
    state.save()
    write_csv(args.dataset / "titles_review.csv", REVIEW_FIELDS, review)
    selected = [row for row in state.manifest if row.get("selected") == "true"]
    missing = [row["file_id"] for row in selected if not row.get("title")]
    print(f"titled {len(selected) - len(missing)}/{len(selected)} selected rows")
    print(f"missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    print(f"{len(review)} rows to review -> {args.dataset / 'titles_review.csv'}")


if __name__ == "__main__":
    main()
